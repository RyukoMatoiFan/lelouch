"""Lelouch - Agentic video harvester.

Describe what footage you need in natural language and Lelouch will
autonomously search, evaluate, and clip matching videos.
"""

__version__ = "1.0.0"

from .config import Config
from .llm_client import LLMClient
from .models.harvest import (
    ClipResult,
    ContentAnalysis,
    HarvestRequest,
    HarvestResult,
    VideoCandidate,
)

__all__ = [
    "Config",
    "LLMClient",
    "HarvestRequest",
    "VideoCandidate",
    "ContentAnalysis",
    "ClipResult",
    "HarvestResult",
]
