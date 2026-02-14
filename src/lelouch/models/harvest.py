"""Data models for the video harvesting pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class HarvestRequest:
    """Parsed user request for video harvesting."""

    description: str
    target_duration: float  # seconds
    content_criteria: List[str] = field(default_factory=list)
    style_criteria: Dict[str, str] = field(default_factory=dict)
    constraints: Dict[str, str] = field(default_factory=dict)
    max_iterations: int = 20
    output_dir: str = "./harvest_output"


@dataclass
class VideoCandidate:
    """A video found via search that hasn't been evaluated yet."""

    url: str
    title: str
    description: Optional[str] = None
    duration: Optional[float] = None  # seconds
    platform: str = "unknown"
    source_query: str = ""
    thumbnail_url: Optional[str] = None


@dataclass
class ContentAnalysis:
    """Result of LLM vision analysis on video keyframes."""

    matches: bool
    confidence: float  # 0.0-1.0
    reason: str
    relevant_segments: List[Tuple[float, float]] = field(default_factory=list)
    description: str = ""


@dataclass
class ClipResult:
    """A successfully extracted video clip."""

    source_url: str
    source_title: str
    clip_path: str
    duration: float
    start_time: float
    end_time: float
    confidence: float


@dataclass
class HarvestResult:
    """Final result of a harvest run."""

    clips: List[ClipResult] = field(default_factory=list)
    total_duration: float = 0.0
    iterations_used: int = 0
    candidates_evaluated: int = 0
    candidates_rejected: int = 0
    request: Optional[HarvestRequest] = None
