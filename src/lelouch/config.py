"""Configuration management for Lelouch."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

# ── Hardcoded defaults (previously HarvestConfig) ─────────────────────────
TEMPERATURE = 0.7
MAX_TOKENS = 4000
LLM_TIMEOUT = 60
MAX_ITERATIONS = 20
MAX_CANDIDATES_PER_ITERATION = 10
MIN_CLIP_DURATION = 3.0
MIN_CONFIDENCE = 0.35
DOWNLOAD_TIMEOUT = 120
OUTPUT_DIR = "./harvest_output"
SAFESEARCH = "moderate"
SEARCH_REGION = "wt-wt"


class LLMConfig(BaseModel):
    """LLM provider configuration — separate credentials per model."""

    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    api_base: Optional[str] = None

    vision_model: str = "gpt-4o"
    vision_api_key: Optional[str] = None
    vision_api_base: Optional[str] = None


class Config(BaseModel):
    """Top-level application configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)

    @classmethod
    def load_from_file(cls, path: str | Path) -> Config:
        """Load config from a YAML file."""
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    @classmethod
    def load_default(cls) -> Config:
        """Load config from settings.yaml in CWD, or return defaults."""
        path = Path("settings.yaml")
        if path.exists():
            return cls.load_from_file(path)
        return cls()

    def save_to_file(self, path: str | Path) -> None:
        """Save config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump()
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
