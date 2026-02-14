"""Keyframe extraction (OpenCV) and LLM vision analysis via grid mosaic."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, List

import cv2
import numpy as np

from ..models.harvest import ContentAnalysis

if TYPE_CHECKING:
    from ..llm_client import LLMClient

logger = logging.getLogger(__name__)

# How many frames to extract for the grid
GRID_FRAMES = 12
# Grid dimensions — 4 columns x 3 rows = 12 cells (larger cells = better recognition)
GRID_COLS = 4
# Each cell is resized to this width (height scales proportionally)
CELL_WIDTH = 480

ANALYSIS_SYSTEM_PROMPT = """\
You are a video content analyst. You will receive a GRID IMAGE containing keyframes \
extracted from a video, plus the VIDEO TITLE. The grid reads left-to-right, \
top-to-bottom. Each cell has a timestamp label.

Your job: decide if this video matches the user's search criteria.

IMPORTANT RULES:
- The VIDEO TITLE is strong evidence. If the title clearly describes the content \
the user wants, that alone is enough to match — even if frames are unclear.
- Be GENEROUS. If the video is roughly about the right topic, it matches.
- You do NOT need to visually identify specific people. Trust the video title — \
a person at a podium in a video titled "TED talk by John Smith" IS that person.
- When in doubt, say YES. False positives are OK, false negatives waste downloaded bandwidth.

Return ONLY valid JSON (no markdown fences):
{
    "matches": true,
    "confidence": 0.85,
    "reason": "Title confirms keynote presentation, frames show speaker at podium",
    "relevant_timestamps": ["0:05", "0:15", "0:30"],
    "description": "Conference presentation footage"
}

- matches: true if video is relevant (be generous!)
- confidence: 0.0-1.0
- reason: brief explanation
- relevant_timestamps: which timestamps are most relevant (pick all if whole video matches)
- description: what you see"""


class FrameAnalyzer:
    """Extract keyframes from video, build a grid mosaic, analyze with LLM vision."""

    def __init__(self, llm_client: LLMClient, temp_dir: str | None = None):
        self.llm = llm_client
        self.num_frames = GRID_FRAMES
        self._temp_dir = temp_dir or str(Path.cwd() / ".lelouch_temp" / "frames")

    def extract_keyframes(self, video_path: str, num_frames: int | None = None) -> List[str]:
        """Extract evenly-spaced keyframes and build a labeled grid mosaic.

        Returns a list with ONE path — the grid image.
        """
        n = num_frames or self.num_frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration = total_frames / fps

        if total_frames <= 0:
            cap.release()
            raise RuntimeError(f"Video has no frames: {video_path}")

        # Pick evenly spaced indices (skip first/last 5% to avoid black frames)
        start = int(total_frames * 0.05)
        end = int(total_frames * 0.95)
        if end <= start:
            start, end = 0, total_frames
        step = max(1, (end - start) // n)
        indices = [start + i * step for i in range(n) if start + i * step < end]

        frames = []
        timestamps = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frames.append(frame)
            t = idx / fps
            timestamps.append(f"{int(t)//60}:{int(t)%60:02d}")

        cap.release()

        if not frames:
            raise RuntimeError(f"Could not extract frames from {video_path}")

        # Build the grid mosaic
        grid_path = self._build_grid(frames, timestamps, video_path)
        logger.info("Built grid mosaic (%d frames) from %s", len(frames), video_path)
        return [grid_path]

    def _build_grid(
        self, frames: list, timestamps: list[str], video_path: str
    ) -> str:
        """Stitch frames into a labeled grid image."""
        cols = min(GRID_COLS, len(frames))
        rows = math.ceil(len(frames) / cols)

        # Resize all frames to uniform cell size
        cells = []
        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            cell_h = int(CELL_WIDTH * h / w)
            cell = cv2.resize(frame, (CELL_WIDTH, cell_h))

            # Add timestamp label
            ts = timestamps[i] if i < len(timestamps) else ""
            cv2.putText(
                cell, ts, (5, cell_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                cell, ts, (5, cell_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                cv2.LINE_AA,
            )
            cells.append(cell)

        # Normalize all cells to the same height (use median)
        heights = [c.shape[0] for c in cells]
        target_h = sorted(heights)[len(heights) // 2]
        for i, cell in enumerate(cells):
            if cell.shape[0] != target_h:
                cells[i] = cv2.resize(cell, (CELL_WIDTH, target_h))

        # Pad to fill the grid
        while len(cells) < rows * cols:
            cells.append(np.zeros((target_h, CELL_WIDTH, 3), dtype=np.uint8))

        # Assemble rows
        row_imgs = []
        for r in range(rows):
            row_cells = cells[r * cols : (r + 1) * cols]
            row_imgs.append(np.hstack(row_cells))

        grid = np.vstack(row_imgs)

        # Save
        stem = Path(video_path).stem
        grid_path = str(Path(self._temp_dir) / f"{stem}_grid.jpg")
        cv2.imwrite(grid_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return grid_path

    def analyze(
        self,
        frame_paths: List[str],
        criteria: List[str],
        video_duration: float,
        video_title: str = "",
    ) -> ContentAnalysis:
        """Send the grid mosaic to vision LLM and parse the analysis result."""
        if not frame_paths:
            return ContentAnalysis(
                matches=False, confidence=0.0, reason="No frames extracted"
            )

        prompt = (
            f"VIDEO TITLE: {video_title}\n\n"
            f"This grid image shows {self.num_frames} keyframes from the video "
            f"(total duration: {video_duration:.1f}s), reading left-to-right, top-to-bottom. "
            f"Each cell has a timestamp.\n\n"
            f"The user is looking for footage matching: {criteria}\n\n"
            f"Does this video match? Consider the title as strong evidence."
        )

        raw = self.llm.generate_with_images(
            prompt, frame_paths, system_prompt=ANALYSIS_SYSTEM_PROMPT
        )

        return self._parse_response(raw, video_duration)

    @staticmethod
    def _is_refusal(text: str) -> bool:
        """Detect if the LLM response is a content-policy refusal."""
        lower = text.lower()
        refusal_phrases = [
            "i cannot", "i can't", "i'm unable", "i am unable",
            "i'm sorry", "i apologize", "content policy",
            "safety guidelines", "safety policy", "not able to",
            "cannot assist", "can't assist", "cannot analyze",
            "can't analyze", "cannot process", "can't process",
            "inappropriate content", "explicit content",
            "violates", "not appropriate",
        ]
        return any(phrase in lower for phrase in refusal_phrases)

    def _parse_response(
        self, raw: str, video_duration: float
    ) -> ContentAnalysis:
        """Parse LLM JSON response into ContentAnalysis."""
        # Check for content-policy refusal FIRST (before JSON parsing)
        # The VLM may refuse as plain text OR wrap refusal in valid JSON
        if self._is_refusal(raw):
            logger.info("Vision model refused analysis (content policy), auto-accepting")
            return ContentAnalysis(
                matches=True,
                confidence=0.75,
                reason="Vision model refused analysis — auto-accepted",
                relevant_segments=[(0.0, video_duration)],
                description=raw[:200],
            )

        try:
            # Strip markdown fences if present
            match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
            text = match.group(1).strip() if match else raw.strip()
            data = json.loads(text)
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning("Failed to parse LLM analysis response: %s", e)
            return ContentAnalysis(
                matches=False,
                confidence=0.0,
                reason=f"Failed to parse LLM response: {e}",
                description=raw[:200],
            )

        # Also check if the reason field itself is a refusal
        reason = data.get("reason", "")
        if self._is_refusal(reason):
            logger.info("Vision model hid refusal in JSON reason field, auto-accepting")
            return ContentAnalysis(
                matches=True,
                confidence=0.75,
                reason="Vision model refused analysis — auto-accepted",
                relevant_segments=[(0.0, video_duration)],
                description=data.get("description", raw[:200]),
            )

        # Convert relevant timestamps to time segments
        ts_list = data.get("relevant_timestamps", [])
        segments = self._timestamps_to_segments(ts_list, video_duration)

        return ContentAnalysis(
            matches=bool(data.get("matches", False)),
            confidence=float(data.get("confidence", 0.0)),
            reason=reason,
            relevant_segments=segments,
            description=data.get("description", ""),
        )

    @staticmethod
    def _timestamps_to_segments(
        timestamps: list[str], duration: float
    ) -> list[tuple[float, float]]:
        """Convert timestamp strings like '1:30' to (start, end) time pairs.

        Groups nearby timestamps into contiguous segments with a buffer.
        """
        if not timestamps:
            return []

        # Parse timestamps to seconds
        times = []
        for ts in timestamps:
            try:
                parts = ts.strip().split(":")
                parts = [int(p) for p in parts]
                if len(parts) == 2:
                    times.append(parts[0] * 60 + parts[1])
                elif len(parts) == 3:
                    times.append(parts[0] * 3600 + parts[1] * 60 + parts[2])
            except (ValueError, IndexError):
                continue

        if not times:
            return [(0.0, duration)]

        times.sort()
        buffer = 15.0  # seconds around each timestamp

        segments = []
        seg_start = max(0, times[0] - buffer)
        seg_end = min(duration, times[0] + buffer)

        for t in times[1:]:
            if t - buffer <= seg_end:
                seg_end = min(duration, t + buffer)
            else:
                segments.append((seg_start, seg_end))
                seg_start = max(0, t - buffer)
                seg_end = min(duration, t + buffer)

        segments.append((seg_start, seg_end))
        return segments
