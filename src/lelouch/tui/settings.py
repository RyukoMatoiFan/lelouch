"""Interactive settings screen — simplified 3-option menu."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

import questionary
from questionary import Style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..config import Config

MENU_STYLE = Style([
    ("qmark", "fg:ansibrightcyan bold"),
    ("question", "fg:ansiwhite bold"),
    ("answer", "fg:ansicyan bold"),
    ("pointer", "fg:ansibrightmagenta bold"),
    ("highlighted", "fg:ansibrightcyan bold"),
    ("selected", "fg:ansicyan"),
])

# ── Predefined models by provider ──────────────────────────────────────────

PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "models": ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o3-mini"],
        "vision_models": ["gpt-4o", "gpt-4o-mini"],
        "env_var": "OPENAI_API_KEY",
    },
    "anthropic": {
        "name": "Anthropic",
        "models": [
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-6",
            "claude-haiku-4-5-20251001",
        ],
        "vision_models": [
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-6",
            "claude-haiku-4-5-20251001",
        ],
        "env_var": "ANTHROPIC_API_KEY",
    },
    "gemini": {
        "name": "Google Gemini",
        "models": [
            "gemini/gemini-2.5-pro",
            "gemini/gemini-2.5-flash",
            "gemini/gemini-2.0-flash",
        ],
        "vision_models": [
            "gemini/gemini-2.5-pro",
            "gemini/gemini-2.5-flash",
            "gemini/gemini-2.0-flash",
        ],
        "env_var": "GEMINI_API_KEY",
    },
    "xai": {
        "name": "xAI (Grok)",
        "models": [
            "xai/grok-4",
            "xai/grok-3",
            "xai/grok-3-mini",
            "xai/grok-2",
        ],
        "vision_models": ["xai/grok-2-vision-latest"],
        "env_var": "XAI_API_KEY",
    },
    "groq": {
        "name": "Groq",
        "models": [
            "groq/llama-3.3-70b-versatile",
            "groq/llama-3.1-8b-instant",
            "groq/qwen/qwen3-32b",
        ],
        "vision_models": ["groq/meta-llama/llama-4-scout-17b-16e-instruct"],
        "env_var": "GROQ_API_KEY",
    },
    "deepseek": {
        "name": "DeepSeek",
        "models": [
            "deepseek/deepseek-chat",
            "deepseek/deepseek-reasoner",
        ],
        "vision_models": [],
        "env_var": "DEEPSEEK_API_KEY",
    },
    "mistral": {
        "name": "Mistral",
        "models": [
            "mistral/mistral-large-latest",
            "mistral/mistral-small-latest",
            "mistral/codestral-latest",
        ],
        "vision_models": [],
        "env_var": "MISTRAL_API_KEY",
    },
    "zai": {
        "name": "Z.AI (Zhipu)",
        "models": ["zai/glm-5", "zai/glm-4.7", "zai/glm-4.5-flash"],
        "vision_models": [],
        "env_var": "ZAI_API_KEY",
    },
    "ollama": {
        "name": "Ollama (Local)",
        "models": ["ollama/llama3.2", "ollama/mistral", "ollama/qwen2.5", "ollama/gemma2"],
        "vision_models": ["ollama/llava", "ollama/llama3.2-vision"],
        "env_var": None,
    },
    "openrouter": {
        "name": "OpenRouter",
        "models": ["openrouter/auto"],
        "vision_models": ["openrouter/auto"],
        "env_var": "OPENROUTER_API_KEY",
    },
}


def fetch_openrouter_free_models() -> list[str]:
    """Fetch free models from OpenRouter API."""
    try:
        import urllib.request
        import ssl

        ctx = ssl.create_default_context()
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"User-Agent": "Lelouch/1.0"}
        )
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        free_models = []
        for model in data.get("data", []):
            model_id = model.get("id", "")
            if model_id.endswith(":free"):
                free_models.append(f"openrouter/{model_id}")

        free_models.sort()
        return free_models if free_models else ["openrouter/auto"]
    except Exception:
        return ["openrouter/auto"]


def fetch_ollama_models() -> list[str]:
    """Fetch locally available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return ["ollama/llama3.2", "ollama/mistral", "ollama/qwen2.5"]

        models = []
        for line in result.stdout.strip().split("\n")[1:]:
            if line.strip():
                model_name = line.split()[0].split(":")[0]
                models.append(f"ollama/{model_name}")

        return models if models else ["ollama/llama3.2", "ollama/mistral"]
    except Exception:
        return ["ollama/llama3.2", "ollama/mistral", "ollama/qwen2.5", "ollama/gemma2"]


class SettingsScreen:
    """Interactive settings — just 3 options: text model, vision model, API key."""

    def __init__(self, config: Config) -> None:
        self.config = config.model_copy(deep=True)
        self.console = Console()

    def _mask_key(self, key: Optional[str]) -> str:
        if not key:
            return "not set"
        if len(key) > 12:
            return f"{key[:4]}...{key[-4:]}"
        return "***"

    def _print_header(self) -> None:
        """Print current settings summary."""
        cfg = self.config
        table = Table(show_header=False, show_edge=False, box=None, padding=(0, 2), expand=True)
        table.add_column("Setting", min_width=20)
        table.add_column("Value", style="green")

        table.add_row("  Text model", cfg.llm.model)
        table.add_row("  Text API key", self._mask_key(cfg.llm.api_key))
        table.add_row("  Vision model", cfg.llm.vision_model)
        table.add_row("  Vision API key", self._mask_key(cfg.llm.vision_api_key))
        table.add_row("  Agentic mode", "ON" if cfg.llm.agentic_mode else "OFF")

        self.console.print(
            Panel(
                table,
                title="[bold cyan]Lelouch Settings[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
        )
        self.console.print()

    def _configure_model(self, field: str) -> None:
        """Configure model or vision_model field."""
        is_vision = field == "vision_model"
        model_key = "vision_models" if is_vision else "models"
        current = getattr(self.config.llm, field)

        # Provider selection with status indicators
        provider_choices = []
        for key, info in PROVIDERS.items():
            env_var = info.get("env_var")
            has_key = bool(os.environ.get(env_var)) if env_var else True
            status = "\u2713" if has_key else "\u25cb"
            provider_choices.append(
                questionary.Choice(f"{status} {info['name']}", value=key)
            )
        provider_choices.append(questionary.Choice("\u2190 Back", value="_back"))

        provider = questionary.select(
            f"Select provider for {field.replace('_', ' ')}:",
            choices=provider_choices,
            style=MENU_STYLE,
        ).ask()

        if provider is None or provider == "_back":
            return

        # Fetch models dynamically for certain providers
        if provider == "openrouter":
            self.console.print("[dim]Fetching free models from OpenRouter...[/dim]")
            models = fetch_openrouter_free_models()
        elif provider == "ollama":
            self.console.print("[dim]Fetching local Ollama models...[/dim]")
            if is_vision:
                models = list(PROVIDERS[provider][model_key])
            else:
                models = fetch_ollama_models()
        else:
            models = list(PROVIDERS[provider][model_key])

        if "Custom..." not in models:
            models.append("Custom...")
        models.append("\u2190 Back")

        model = questionary.select(
            "Select model:",
            choices=models,
            style=MENU_STYLE,
        ).ask()

        if model is None or model == "\u2190 Back":
            return

        if model == "Custom...":
            model = questionary.text(
                "Enter model identifier:",
                default=current,
                style=MENU_STYLE,
            ).ask()

        if not model:
            return

        is_vision = field == "vision_model"
        setattr(self.config.llm, field, model)

        # Set api_base per model
        api_base_field = "vision_api_base" if is_vision else "api_base"
        if provider in PROVIDERS and "api_base" in PROVIDERS[provider]:
            setattr(self.config.llm, api_base_field, PROVIDERS[provider]["api_base"])
            self.console.print(f"[green]API base set to {PROVIDERS[provider]['api_base']}[/green]")
        else:
            if getattr(self.config.llm, api_base_field):
                setattr(self.config.llm, api_base_field, None)
                self.console.print("[green]API base cleared (not needed)[/green]")

        self.console.print(f"[green]Updated {field.replace('_', ' ')} to {model}[/green]")

        # Prompt for API key right after model selection
        env_var = PROVIDERS.get(provider, {}).get("env_var")
        if env_var:
            api_key_field = "vision_api_key" if is_vision else "api_key"
            current = getattr(self.config.llm, api_key_field)
            hint = f" (Enter to keep current)" if current else ""
            key = questionary.password(
                f"API key for {PROVIDERS[provider]['name']}{hint}:",
                style=MENU_STYLE,
            ).ask()
            if key:
                setattr(self.config.llm, api_key_field, key)
                self.console.print(f"[green]{field.replace('_', ' ').title()} API key updated[/green]")

    def _configure_api_key(self, which: str) -> None:
        """Configure API key for text or vision model."""
        is_vision = which == "vision"
        api_key_field = "vision_api_key" if is_vision else "api_key"
        label = "Vision" if is_vision else "Text"

        key = questionary.password(
            f"{label} model API key (leave blank to keep current):",
            style=MENU_STYLE,
        ).ask()
        if key:
            setattr(self.config.llm, api_key_field, key)
            self.console.print(f"[green]{label} API key updated[/green]")

    def run(self) -> Config:
        """Run settings menu."""
        while True:
            self.console.clear()
            self._print_header()

            agentic_label = "ON" if self.config.llm.agentic_mode else "OFF"
            choice = questionary.select(
                "What would you like to configure?",
                choices=[
                    questionary.Choice("  Text model", value="model"),
                    questionary.Choice("  Text API key", value="text_api_key"),
                    questionary.Choice("  Vision model", value="vision_model"),
                    questionary.Choice("  Vision API key", value="vision_api_key"),
                    questionary.Choice(f"  Agentic mode [{agentic_label}]", value="agentic_mode"),
                    questionary.Choice("  Done", value="_done"),
                ],
                style=MENU_STYLE,
            ).ask()

            if choice is None or choice == "_done":
                break
            elif choice == "model":
                self._configure_model("model")
            elif choice == "vision_model":
                self._configure_model("vision_model")
            elif choice == "text_api_key":
                self._configure_api_key("text")
            elif choice == "vision_api_key":
                self._configure_api_key("vision")
            elif choice == "agentic_mode":
                self.config.llm.agentic_mode = not self.config.llm.agentic_mode
                state = "ON" if self.config.llm.agentic_mode else "OFF"
                self.console.print(f"[green]Agentic mode {state}[/green]")

        return self.config
