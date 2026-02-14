"""LiteLLM wrapper with text and vision support."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import litellm
from litellm import ModelResponse

from .config import TEMPERATURE, MAX_TOKENS, LLM_TIMEOUT, Config

logger = logging.getLogger(__name__)

# Known LiteLLM provider prefixes that handle their own routing
_KNOWN_PREFIXES = (
    "openai/", "ollama/", "azure/", "gemini/", "anthropic/",
    "xai/", "openrouter/", "groq/", "mistral/", "deepseek/",
    "cohere/", "together_ai/", "zai/",
)

# Model prefix → env var name (for providers that need it in the env)
_PREFIX_TO_ENV_VAR = {
    "openai/": "OPENAI_API_KEY",
    "anthropic/": "ANTHROPIC_API_KEY",
    "gemini/": "GEMINI_API_KEY",
    "xai/": "XAI_API_KEY",
    "openrouter/": "OPENROUTER_API_KEY",
    "groq/": "GROQ_API_KEY",
    "mistral/": "MISTRAL_API_KEY",
    "deepseek/": "DEEPSEEK_API_KEY",
    "cohere/": "COHERE_API_KEY",
    "together_ai/": "TOGETHER_API_KEY",
    "zai/": "ZAI_API_KEY",
}

# Models that don't use a prefix but can be identified by name start
_NAME_TO_ENV_VAR = {
    "gpt": "OPENAI_API_KEY",
    "o1": "OPENAI_API_KEY",
    "o3": "OPENAI_API_KEY",
    "o4": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
}


def _detect_env_var(model: str) -> str | None:
    """Detect the correct env var for a model string."""
    for prefix, env_var in _PREFIX_TO_ENV_VAR.items():
        if model.startswith(prefix):
            return env_var
    for name_start, env_var in _NAME_TO_ENV_VAR.items():
        if model.startswith(name_start):
            return env_var
    return None


def _set_provider_key(model: str, api_key: str) -> None:
    """Set API key in the correct env var for the model's provider.

    Many LiteLLM providers (especially OpenRouter) only read from env vars,
    not from the api_key parameter. Belt-and-suspenders: we set both.
    """
    env_var = _detect_env_var(model)
    if env_var:
        os.environ[env_var] = api_key


def _prepare_model(model: str, api_base: str | None) -> str:
    """Add openai/ prefix for custom base URLs with unknown model names."""
    if api_base and not any(model.startswith(p) for p in _KNOWN_PREFIXES):
        return f"openai/{model}"
    return model


def _should_pass_api_base(model: str, api_base: str | None) -> bool:
    """Only pass api_base for models that need it.

    OpenRouter is handled internally by LiteLLM — never pass api_base for it.
    """
    if not api_base:
        return False
    return not model.startswith("openrouter/")


class LLMClient:
    """Unified LLM client for text generation and vision analysis.

    Supports separate providers/keys for text and vision models.
    """

    def __init__(self, config: Config):
        cfg = config.llm

        # Text model setup
        self.text_model = _prepare_model(cfg.model, cfg.api_base)
        self.text_api_key = cfg.api_key
        self.text_api_base = cfg.api_base

        # Vision model setup (falls back to text credentials if not set)
        vision_api_key = cfg.vision_api_key or cfg.api_key
        vision_api_base = cfg.vision_api_base or cfg.api_base
        self.vision_model = _prepare_model(cfg.vision_model, vision_api_base)
        self.vision_api_key = vision_api_key
        self.vision_api_base = vision_api_base

        # Set env vars for both providers (required by OpenRouter, others)
        if self.text_api_key:
            _set_provider_key(self.text_model, self.text_api_key)
        if self.vision_api_key:
            _set_provider_key(self.vision_model, self.vision_api_key)

        litellm.set_verbose = False
        litellm.suppress_debug_info = True
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("litellm").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Text-only generation."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.text_model,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "timeout": LLM_TIMEOUT,
        }

        if self.text_api_key:
            kwargs["api_key"] = self.text_api_key
        if _should_pass_api_base(self.text_model, self.text_api_base):
            kwargs["api_base"] = self.text_api_base

        response = litellm.completion(**kwargs)
        return response.choices[0].message.content

    def generate_with_images(
        self,
        prompt: str,
        image_paths: List[str],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Vision generation with base64-encoded images."""
        content: list = [{"type": "text", "text": prompt}]

        for img_path in image_paths:
            b64 = base64.b64encode(Path(img_path).read_bytes()).decode()
            ext = Path(img_path).suffix.lstrip(".").lower()
            mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(ext, "jpeg")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{mime};base64,{b64}"},
                }
            )

        messages: list = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        kwargs = {
            "model": self.vision_model,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "timeout": LLM_TIMEOUT,
        }

        if self.vision_api_key:
            kwargs["api_key"] = self.vision_api_key
        if _should_pass_api_base(self.vision_model, self.vision_api_base):
            kwargs["api_base"] = self.vision_api_base

        response = litellm.completion(**kwargs)
        return response.choices[0].message.content

    def generate_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
    ) -> ModelResponse:
        """LLM completion with tool calling support via LiteLLM.

        Args:
            messages: Full conversation history (system/user/assistant/tool messages).
            tools: Tool definitions in OpenAI function calling format.
            tool_choice: "auto", "required", or "none".

        Returns:
            Raw ModelResponse — caller manages message history.
        """
        kwargs = {
            "model": self.text_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "timeout": LLM_TIMEOUT,
        }

        if self.text_api_key:
            kwargs["api_key"] = self.text_api_key
        if _should_pass_api_base(self.text_model, self.text_api_base):
            kwargs["api_base"] = self.text_api_base

        return litellm.completion(**kwargs)

    def generate_json(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> dict | list:
        """Generate and parse a JSON response, stripping markdown fences."""
        raw = self.generate(prompt, system_prompt=system_prompt)
        cleaned = self._extract_json(raw)
        return json.loads(cleaned)

    @staticmethod
    def _extract_json(text: str) -> str:
        """Strip markdown code fences if present."""
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()
