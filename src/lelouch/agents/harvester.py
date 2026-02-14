"""Core agentic harvester loop: search -> download -> analyze -> clip."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Callable, List, Optional

from ..config import (
    DOWNLOAD_TIMEOUT,
    MAX_CANDIDATES_PER_ITERATION,
    MIN_CLIP_DURATION,
    MIN_CONFIDENCE,
    OUTPUT_DIR,
    Config,
)
from ..llm_client import LLMClient
from ..models.harvest import (
    ClipResult,
    ContentAnalysis,
    HarvestRequest,
    HarvestResult,
    VideoCandidate,
)
from .frame_analyzer import FrameAnalyzer
from .harvest_memory import HarvestMemory
from .search import VideoSearcher
from .video_clipper import VideoClipper

logger = logging.getLogger(__name__)

EventCallback = Callable[..., None]

QUERY_GEN_SYSTEM_PROMPT = """\
You are an expert search query generator for finding individual, downloadable video clips on the internet.

## Rules for EVERY query

- ALWAYS include the word "video" in every query
- Write queries in the same language as the user's request
- Keep queries SHORT and natural
- Do NOT use exclusion operators (-live -full etc.) — they hurt recall
- Do NOT use intitle: or other advanced operators (only site: and "quoted phrases")

## Platform targeting

- 2-3 queries: no site: restriction (broadest results)
- 1-2 queries: site:youtube.com
- If the user explicitly asks for a specific platform, respect that

## Generate 5 diverse queries

- Query #1 MUST be the user's original request words (only fix grammar/typos), with "video" added
- Mix broad and narrow queries, different synonyms and phrasings
- Always differ from previously tried queries listed in the context

## Output format

Return valid JSON only (no markdown fences, no extra text):
{"analysis": "Brief strategy explanation", "queries": ["query1", "query2", "query3", "query4", "query5"]}"""


class VideoHarvester:
    """Orchestrates the full harvest pipeline with event-driven updates."""

    def __init__(
        self,
        config: Config,
        llm_client: LLMClient,
        event_callback: Optional[EventCallback] = None,
    ):
        self.config = config
        self.llm = llm_client
        self.searcher = VideoSearcher()
        self.analyzer = FrameAnalyzer(llm_client)
        self.clipper = VideoClipper()
        self.clipper.set_progress_hook(self._on_download_progress)
        self.memory = HarvestMemory()
        self.emit = event_callback or (lambda *_a, **_kw: None)
        self._temp_dir = ""  # set in harvest() once output_dir is known
        # Track consecutive failures per platform to skip broken ones
        self._platform_fails: dict[str, int] = {}
        self._platform_fail_limit = 2  # skip platform after N consecutive failures

    # ── Public API ───────────────────────────────────────────────────

    def harvest(self, request: HarvestRequest) -> HarvestResult:
        """Run the full autonomous harvest loop."""
        os.makedirs(request.output_dir, exist_ok=True)

        # Temp dirs inside output so user can find / clean them
        self._temp_dir = os.path.join(request.output_dir, ".lelouch_temp", "downloads")
        frames_dir = os.path.join(request.output_dir, ".lelouch_temp", "frames")
        os.makedirs(self._temp_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)
        self.analyzer._temp_dir = frames_dir

        self.emit("start", request)

        while self.memory.total_duration < request.target_duration:
            if self.memory.iteration >= request.max_iterations:
                self.emit(
                    "log",
                    f"Reached max iterations ({request.max_iterations}), stopping.",
                )
                break

            self.memory.iteration += 1
            self.emit("iteration_start", self.memory.iteration)

            # Step 1 — generate search queries
            queries = self._generate_queries(request)
            self.emit("queries_generated", queries)

            # Step 2 — search
            candidates: List[VideoCandidate] = []
            for q in queries:
                self.emit("searching", q)
                results = self.searcher.search(
                    q, max_results=MAX_CANDIDATES_PER_ITERATION
                )
                self.emit("search_done", q, len(results))
                candidates.extend(results)

            candidates = self._prefilter(candidates)
            self.memory.add_candidates(candidates)
            self.emit("candidates_found", len(candidates))

            if not candidates:
                self.emit("log", "No new candidates this iteration, continuing.")
                continue

            # Step 3 — evaluate each candidate
            for candidate in candidates:
                if self.memory.total_duration >= request.target_duration:
                    break

                self.emit("evaluating", candidate)
                self._evaluate_candidate(candidate, request)

            self.emit("iteration_end", self.memory)

        result = self._build_result(request)
        self.emit("complete", result)
        self._cleanup()
        return result

    # ── Private helpers ──────────────────────────────────────────────

    def _on_download_progress(
        self, downloaded_mb: float, pct: float, speed: float, eta: float
    ) -> None:
        """Forward yt-dlp download progress to the event system."""
        self.emit("download_progress", downloaded_mb, pct, speed, eta)

    def _evaluate_candidate(
        self, candidate: VideoCandidate, request: HarvestRequest
    ) -> None:
        """Download, analyze, and potentially clip a single candidate."""
        platform = candidate.platform or "unknown"

        # Skip platforms that keep failing
        if self._platform_fails.get(platform, 0) >= self._platform_fail_limit:
            reason = f"Skipping {platform} (failed {self._platform_fails[platform]}x in a row)"
            self.memory.add_rejected(candidate.url, reason)
            self.emit("clip_rejected", candidate, reason)
            return

        video_path = None
        try:
            # Pre-check: extract info to get duration (fast, no download)
            self.emit("checking_info", candidate)
            try:
                from .video_clipper import MAX_VIDEO_DURATION
                info = self.clipper.get_video_info(candidate.url)
                vid_duration = info.get("duration") or 0
                self.emit("info_ready", candidate, vid_duration)

                if vid_duration > MAX_VIDEO_DURATION:
                    reason = f"Too long ({vid_duration:.0f}s > {MAX_VIDEO_DURATION}s limit)"
                    self.memory.add_rejected(candidate.url, reason)
                    self.emit("clip_rejected", candidate, reason)
                    return

                # Store actual duration on candidate for later use
                candidate.duration = vid_duration
            except Exception as e:
                err_str = str(e).lower()
                # Hard failures — skip immediately, don't waste time downloading
                if any(s in err_str for s in (
                    "unsupported url", "not a valid url", "no video",
                    "is not available", "private video", "removed",
                    "blocked", "geo", "age",
                )):
                    reason = f"Unsupported: {str(e)[:80]}"
                    self.memory.add_rejected(candidate.url, reason)
                    self.emit("clip_rejected", candidate, reason)
                    return
                # Soft failures (network timeout etc.) — still try downloading
                self.emit("log", f"Info check failed, trying download: {str(e)[:60]}")

            # Download
            self.emit("downloading", candidate)
            video_path = os.path.join(
                self._temp_dir,
                f"dl_{self.memory.iteration}_{id(candidate)}",
            )
            video_path = self.clipper.download_video(
                candidate.url,
                video_path,
                timeout=DOWNLOAD_TIMEOUT,
            )
            self.emit("downloaded", candidate)

            # Keyframe extraction → grid mosaic
            self.emit("extracting_frames", candidate)
            frames = self.analyzer.extract_keyframes(video_path)
            from .frame_analyzer import GRID_FRAMES
            self.emit("keyframes_extracted", GRID_FRAMES)

            # Duration
            try:
                duration = self.clipper.get_duration(video_path)
            except Exception:
                duration = candidate.duration or 60.0

            # LLM vision analysis (single grid image)
            self.emit("analyzing", candidate, GRID_FRAMES)
            analysis = self.analyzer.analyze(
                frames, request.content_criteria, duration,
                video_title=candidate.title,
            )
            self.emit("analysis_complete", analysis)

            if (
                analysis.matches
                and analysis.confidence >= MIN_CONFIDENCE
            ):
                clips = self._create_clips(
                    video_path, candidate, analysis, request, duration
                )
                for clip in clips:
                    self.memory.add_accepted(clip, source_query=candidate.source_query)
                    self.emit("clip_accepted", clip)
                # Success — reset platform fail counter
                self._platform_fails[platform] = 0
            else:
                reason = analysis.reason or f"Low confidence ({analysis.confidence:.2f})"
                self.memory.add_rejected(candidate.url, reason)
                self.emit("clip_rejected", candidate, reason)

        except Exception as e:
            # Track consecutive platform failures
            self._platform_fails[platform] = self._platform_fails.get(platform, 0) + 1
            fails = self._platform_fails[platform]
            limit = self._platform_fail_limit
            logger.warning("Error evaluating %s: %s", candidate.url, e)
            self.emit("error", candidate, str(e))
            if fails >= limit:
                self.emit("log", f"Disabling {platform} after {fails} consecutive failures")
            self.memory.add_rejected(candidate.url, str(e))
        finally:
            # Clean up downloaded video and any partial files (keep clips)
            if video_path:
                base = Path(video_path)
                # Remove the file itself + any yt-dlp artifacts with same stem
                for p in [base] + list(base.parent.glob(f"{base.stem}.*")):
                    try:
                        if p.exists():
                            p.unlink()
                    except OSError:
                        pass

    def _create_clips(
        self,
        video_path: str,
        candidate: VideoCandidate,
        analysis: ContentAnalysis,
        request: HarvestRequest,
        video_duration: float,
    ) -> List[ClipResult]:
        """Clip matching segments from the video."""
        segments = analysis.relevant_segments
        if not segments:
            # Use the full video as one clip
            segments = [(0.0, video_duration)]

        clips: List[ClipResult] = []
        clip_idx = len(self.memory.accepted_clips)

        for start, end in segments:
            seg_duration = end - start
            if seg_duration < MIN_CLIP_DURATION:
                continue

            # Check if we'd exceed target
            remaining = request.target_duration - self.memory.total_duration
            if remaining <= 0:
                break
            if seg_duration > remaining:
                end = start + remaining

            clip_filename = f"clip_{clip_idx:04d}.mp4"
            clip_path = os.path.join(request.output_dir, clip_filename)

            try:
                self.emit("clipping", start, end, seg_duration)
                self.clipper.clip(video_path, clip_path, start, end)
                clip = self.clipper.make_clip_result(
                    source_url=candidate.url,
                    source_title=candidate.title,
                    clip_path=clip_path,
                    start=start,
                    end=end,
                    confidence=analysis.confidence,
                )
                clips.append(clip)
                clip_idx += 1
            except Exception as e:
                logger.warning("Failed to clip %s [%.1f-%.1f]: %s", video_path, start, end, e)

        return clips

    def _generate_queries(self, request: HarvestRequest) -> List[str]:
        """Ask LLM to generate smart search queries with site: operators."""
        self.emit("generating_queries")
        memory_ctx = self.memory.get_llm_context()
        prompt = (
            f"Generate 5 search queries for finding video footage.\n"
            f"Target content: {request.description}\n"
            f"Content criteria: {request.content_criteria}\n\n"
            f"Current state:\n{memory_ctx}\n\n"
            f"Generate diverse queries that differ from previously tried ones. "
            f"Include both broad and specific queries."
        )

        try:
            data = self.llm.generate_json(prompt, system_prompt=QUERY_GEN_SYSTEM_PROMPT)
            if isinstance(data, dict) and "queries" in data:
                queries = [str(q) for q in data["queries"][:5]]
            elif isinstance(data, list):
                queries = [str(q) for q in data[:5]]
            else:
                queries = [request.description]
        except Exception as e:
            logger.warning("Query generation failed, using fallback: %s", e)
            queries = [
                request.description,
                " ".join(request.content_criteria),
            ]

        self.memory.tried_queries.extend(queries)
        return queries

    def _prefilter(self, candidates: List[VideoCandidate]) -> List[VideoCandidate]:
        """Remove duplicates, already-seen URLs, and too-short videos."""
        seen_urls: set[str] = set()
        filtered: List[VideoCandidate] = []

        for c in candidates:
            if not c.url:
                continue
            if c.url in seen_urls:
                continue
            if self.memory.is_url_seen(c.url):
                continue
            # Skip very short videos (< MIN_CLIP_DURATION)
            if c.duration and c.duration < MIN_CLIP_DURATION:
                continue
            seen_urls.add(c.url)
            filtered.append(c)

        return filtered[:MAX_CANDIDATES_PER_ITERATION]

    def _build_result(self, request: HarvestRequest) -> HarvestResult:
        return HarvestResult(
            clips=list(self.memory.accepted_clips),
            total_duration=self.memory.total_duration,
            iterations_used=self.memory.iteration,
            candidates_evaluated=len(self.memory.found_candidates),
            candidates_rejected=len(self.memory.rejected),
            request=request,
        )

    def _cleanup(self) -> None:
        """Remove temporary download/frames directories."""
        # Clean the parent .lelouch_temp dir (holds both downloads/ and frames/)
        temp_parent = str(Path(self._temp_dir).parent) if self._temp_dir else ""
        if temp_parent and Path(temp_parent).name == ".lelouch_temp":
            try:
                shutil.rmtree(temp_parent, ignore_errors=True)
            except OSError:
                pass
