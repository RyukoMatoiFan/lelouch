"""Persistent state and memory for the harvest agent loop."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

from ..models.harvest import ClipResult, VideoCandidate

logger = logging.getLogger(__name__)


@dataclass
class HarvestMemory:
    """Tracks state across harvest iterations to inform future decisions."""

    tried_queries: List[str] = field(default_factory=list)
    found_candidates: List[VideoCandidate] = field(default_factory=list)
    accepted_clips: List[ClipResult] = field(default_factory=list)
    rejected: Dict[str, str] = field(default_factory=dict)  # url -> reason
    _query_to_clips: Dict[str, int] = field(default_factory=dict)  # query -> accepted count
    iteration: int = 0
    total_duration: float = 0.0

    def add_accepted(self, clip: ClipResult, source_query: str = "") -> None:
        """Record an accepted clip."""
        self.accepted_clips.append(clip)
        self.total_duration += clip.duration
        if source_query:
            self._query_to_clips[source_query] = (
                self._query_to_clips.get(source_query, 0) + 1
            )

    def add_rejected(self, url: str, reason: str) -> None:
        """Record a rejected candidate."""
        self.rejected[url] = reason

    def is_url_seen(self, url: str) -> bool:
        """Check if we already tried this URL."""
        seen_urls = {c.url for c in self.found_candidates}
        seen_urls.update(self.rejected.keys())
        return url in seen_urls

    def add_candidates(self, candidates: List[VideoCandidate]) -> None:
        """Track candidates we've seen."""
        self.found_candidates.extend(candidates)

    def successful_queries(self) -> List[str]:
        """Queries that led to at least one accepted clip."""
        return [q for q, count in self._query_to_clips.items() if count > 0]

    def common_rejection_reasons(self, top_n: int = 5) -> List[str]:
        """Most frequent rejection reasons."""
        counter = Counter(self.rejected.values())
        return [reason for reason, _ in counter.most_common(top_n)]

    def get_llm_context(self) -> str:
        """Format memory state as context for LLM query generation."""
        lines = [
            f"Iteration: {self.iteration}",
            f"Total footage collected: {self.total_duration:.1f}s",
            f"Candidates evaluated: {len(self.found_candidates)}",
            f"Clips accepted: {len(self.accepted_clips)}",
            f"Clips rejected: {len(self.rejected)}",
        ]

        if self.tried_queries:
            lines.append(f"\nPrevious queries tried: {self.tried_queries}")

        successful = self.successful_queries()
        if successful:
            lines.append(f"Queries that found good content: {successful}")

        reasons = self.common_rejection_reasons(3)
        if reasons:
            lines.append(f"Common rejection reasons: {reasons}")
            lines.append("Avoid queries likely to produce similar rejected content.")

        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist memory to a JSON file."""
        data = {
            "tried_queries": self.tried_queries,
            "accepted_clips": [asdict(c) for c in self.accepted_clips],
            "rejected": self.rejected,
            "query_to_clips": self._query_to_clips,
            "iteration": self.iteration,
            "total_duration": self.total_duration,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> HarvestMemory:
        """Load memory from a JSON file."""
        raw = json.loads(Path(path).read_text())
        mem = cls()
        mem.tried_queries = raw.get("tried_queries", [])
        mem.accepted_clips = [
            ClipResult(**c) for c in raw.get("accepted_clips", [])
        ]
        mem.rejected = raw.get("rejected", {})
        mem._query_to_clips = raw.get("query_to_clips", {})
        mem.iteration = raw.get("iteration", 0)
        mem.total_duration = raw.get("total_duration", 0.0)
        return mem
