"""Parse natural language requests into structured HarvestRequest objects."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..models.harvest import HarvestRequest

if TYPE_CHECKING:
    from ..llm_client import LLMClient

logger = logging.getLogger(__name__)

PARSE_SYSTEM_PROMPT = """\
You are a request parser for a video harvesting tool. Parse the user's natural \
language request into structured data.

Extract:
- target_duration: number of seconds of footage wanted. Convert minutes to seconds \
(e.g. "5 minutes" -> 300). If no duration is specified, default to 60.
- content_criteria: list of visual/content characteristics to search and match against. \
Be specific - split compound descriptions into separate criteria.
- style_criteria: technical requirements like resolution, quality, aspect ratio (object, \
can be empty).
- constraints: any other constraints like platform preference, max single video length, \
etc. (object, can be empty).

Return ONLY valid JSON, no markdown fences:
{
    "target_duration": 150.0,
    "content_criteria": ["2D anime", "blonde girls"],
    "style_criteria": {},
    "constraints": {}
}"""


class RequestParser:
    """Parse natural language requests into HarvestRequest via LLM."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def parse(self, user_input: str) -> HarvestRequest:
        """Parse natural speech into a structured HarvestRequest.

        Examples of user input:
          "I want 150 seconds of 2D anime blonde girls"
          "get me 60s of drone footage over mountains, at least 720p"
          "find 5 minutes of funny cat compilation videos"
        """
        data = self.llm.generate_json(user_input, system_prompt=PARSE_SYSTEM_PROMPT)

        return HarvestRequest(
            description=user_input,
            target_duration=float(data.get("target_duration", 60)),
            content_criteria=data.get("content_criteria", []),
            style_criteria=data.get("style_criteria", {}),
            constraints=data.get("constraints", {}),
        )
