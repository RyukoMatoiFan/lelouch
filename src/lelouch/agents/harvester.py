"""Core agentic harvester loop using LLM tool calling."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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
from .tool_defs import HARVEST_TOOLS
from .video_clipper import VideoClipper

logger = logging.getLogger(__name__)

EventCallback = Callable[..., None]

MAX_AGENT_TURNS = 50

AGENT_SYSTEM_PROMPT = """\
You are an autonomous video harvester agent. Your job is to find and collect \
video footage matching the user's request by searching the web, processing \
promising videos, and clipping matching segments.

## Available tools

- **search_videos(query)**: Search the web for videos. Always include "video" in queries. \
Use diverse phrasing, synonyms, and platform targeting (e.g. site:youtube.com). \
Each call returns a batch of candidates.
- **process_video(url, title)**: Download, analyze with vision AI, and auto-clip a video. \
Returns whether it matched and clip details. This is expensive — only process \
videos whose titles look promising.
- **check_progress()**: See current stats (duration collected, clips, rejections).
- **finish_harvest(reason)**: Signal that you're done. Call this when target duration \
is reached, returns are diminishing, or you've exhausted useful strategies.

## Adaptive tools

- **refine_criteria(criteria, reasoning)**: Update what the vision model looks for. \
Use when rejections show a pattern (e.g. "no X found" repeated 3+ times). \
You can broaden ("blonde" → "blonde or light-haired"), narrow, or refocus.
- **adjust_confidence(min_confidence)**: Change the acceptance threshold (0.1–0.9). \
Lower if good-looking videos are borderline rejected. Raise if junk gets through.
- **get_rejection_analysis()**: See WHY videos are being rejected, grouped by pattern. \
Use this before refining criteria — understand the problem first.

## Pipeline control tools

- **set_download_options(timeout, retries)**: Adjust download timeout (10–600s) and retries (1–5). \
Increase retries if downloads keep failing due to flaky connections.
- **adjust_frame_sampling(num_frames)**: Change keyframe count (4–24, default 12). \
Use more frames for long videos, fewer for short ones to save analysis cost.
- **re_enable_platform(platform)**: Re-enable a platform disabled after repeated failures. \
Use when a platform may have been temporarily down.
- **set_video_quality(resolution)**: Set download quality ('360p', '480p', '720p', '1080p'). \
Higher quality improves vision analysis but slows downloads.
- **set_max_video_duration(max_seconds)**: Change the per-video duration limit (60–7200s). \
Auto-scales to target duration. Raise if long videos are being skipped unnecessarily.

## Strategy

1. Start with a search using the user's own words (+ "video"), then broaden.
2. Review search results and pick the most promising candidates by title.
3. Process one video at a time. Check progress after each to see if target is met.
4. If a search yields no good results, try different keywords, synonyms, or platforms.
5. Vary your queries — don't repeat the same search twice.
6. If a platform keeps failing, switch to others (YouTube is most reliable).
7. Stop when: target duration is reached, or you've tried many queries with diminishing returns.

## Adaptive strategy

- After 3+ rejections, call get_rejection_analysis to understand patterns.
- If content mismatches dominate, refine_criteria to better match available content.
- If confidence is borderline, adjust_confidence slightly (±0.05–0.10).
- Don't change criteria on the first rejection — wait for a pattern.
- If downloads keep failing, try set_download_options with more retries before giving up.
- Use adjust_frame_sampling(16–20) for longer videos where 12 frames may miss content.
- If vision analysis seems poor, try set_video_quality('720p') for better frames.

## Important

- You choose which videos to process and in what order — be selective based on titles.
- You decide when to search again vs. process more candidates from previous results.
- Call finish_harvest when done — don't just stop responding."""


# ── Fallback: old hardcoded loop (used when model lacks tool support) ────

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
        # Agent loop state
        self._request: Optional[HarvestRequest] = None
        self._stop = False
        # Cache search results for the agent to pick from
        self._pending_candidates: List[VideoCandidate] = []
        # Mutable runtime overrides (set per-harvest in harvest())
        self._effective_confidence: float = MIN_CONFIDENCE
        self._effective_criteria: List[str] = []
        self._effective_timeout: int = DOWNLOAD_TIMEOUT
        self._effective_retries: int = 1
        self._effective_max_duration: float = 0  # set per-harvest based on target

    # ── Public API ───────────────────────────────────────────────────

    def harvest(self, request: HarvestRequest) -> HarvestResult:
        """Run the autonomous harvest loop using LLM tool calling."""
        self._request = request
        self._stop = False
        self._effective_confidence = MIN_CONFIDENCE
        self._effective_criteria = list(request.content_criteria) if request.content_criteria else []
        self._effective_timeout = DOWNLOAD_TIMEOUT
        self._effective_retries = 1
        from .video_clipper import MAX_VIDEO_DURATION
        self._effective_max_duration = max(MAX_VIDEO_DURATION, request.target_duration)
        self.clipper.format_override = None
        os.makedirs(request.output_dir, exist_ok=True)

        # Temp dirs inside output so user can find / clean them
        self._temp_dir = os.path.join(request.output_dir, ".lelouch_temp", "downloads")
        frames_dir = os.path.join(request.output_dir, ".lelouch_temp", "frames")
        os.makedirs(self._temp_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)
        self.analyzer._temp_dir = frames_dir

        self.emit("start", request)

        if self.config.llm.agentic_mode:
            try:
                self._run_agent_loop(request)
            except Exception as e:
                # If tool calling fails entirely (model doesn't support it),
                # fall back to the old hardcoded loop
                err_msg = str(e).lower()
                if "tools" in err_msg or "tool" in err_msg or "function" in err_msg:
                    logger.warning(
                        "Tool calling not supported by model, falling back to legacy loop: %s", e
                    )
                    self.emit("log", "Model doesn't support tool calling, using legacy mode.")
                    self._run_legacy_loop(request)
                else:
                    raise
        else:
            self._run_legacy_loop(request)

        result = self._build_result(request)
        self.emit("complete", result)
        self._cleanup()
        return result

    # ── Agent loop (tool calling) ────────────────────────────────────

    def _run_agent_loop(self, request: HarvestRequest) -> None:
        """LLM-driven agent loop: the model decides which tools to call."""
        criteria_str = ", ".join(request.content_criteria) if request.content_criteria else request.description
        user_msg = (
            f"Find {request.target_duration:.0f} seconds of video footage.\n"
            f"Description: {request.description}\n"
            f"Content criteria: {criteria_str}\n\n"
            f"Start by searching, then process the most promising results."
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        for turn in range(MAX_AGENT_TURNS):
            if self._stop:
                break
            if self.memory.total_duration >= request.target_duration:
                self.emit("log", "Target duration reached.")
                break

            response = self.llm.generate_with_tools(messages, HARVEST_TOOLS)
            assistant_msg = response.choices[0].message

            # Append the assistant message to history
            messages.append(assistant_msg.model_dump(exclude_none=True))

            tool_calls = assistant_msg.tool_calls
            if not tool_calls:
                # No tool calls — model is done or just chatting
                if assistant_msg.content:
                    self.emit("log", f"Agent: {assistant_msg.content[:120]}")
                break

            for tc in tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    fn_args = {}

                self.emit("log", f"Tool: {fn_name}({json.dumps(fn_args)[:100]})")
                result_str = self._execute_tool(fn_name, fn_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

        else:
            self.emit("log", f"Reached max agent turns ({MAX_AGENT_TURNS}), stopping.")

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Dispatch a tool call to the appropriate backend."""
        try:
            if name == "search_videos":
                return self._tool_search_videos(args.get("query", ""))
            elif name == "process_video":
                return self._tool_process_video(
                    args.get("url", ""),
                    args.get("title", ""),
                )
            elif name == "check_progress":
                return self._tool_check_progress()
            elif name == "finish_harvest":
                return self._tool_finish_harvest(args.get("reason", ""))
            elif name == "refine_criteria":
                return self._tool_refine_criteria(
                    args.get("criteria", []),
                    args.get("reasoning", ""),
                )
            elif name == "adjust_confidence":
                return self._tool_adjust_confidence(args.get("min_confidence", MIN_CONFIDENCE))
            elif name == "get_rejection_analysis":
                return self._tool_get_rejection_analysis()
            elif name == "set_download_options":
                return self._tool_set_download_options(
                    args.get("timeout"),
                    args.get("retries"),
                )
            elif name == "adjust_frame_sampling":
                return self._tool_adjust_frame_sampling(args.get("num_frames", 12))
            elif name == "re_enable_platform":
                return self._tool_re_enable_platform(args.get("platform", ""))
            elif name == "set_video_quality":
                return self._tool_set_video_quality(args.get("resolution", "480p"))
            elif name == "set_max_video_duration":
                return self._tool_set_max_video_duration(args.get("max_seconds", 1800))
            else:
                return json.dumps({"error": f"Unknown tool: {name}"})
        except Exception as e:
            logger.warning("Tool %s failed: %s", name, e)
            return json.dumps({"error": str(e)[:300]})

    # ── Tool implementations ─────────────────────────────────────────

    def _tool_search_videos(self, query: str) -> str:
        """search_videos tool: DDG search + prefilter, return candidates."""
        self.emit("searching", query)
        results = self.searcher.search(query, max_results=MAX_CANDIDATES_PER_ITERATION)
        self.emit("search_done", query, len(results))

        results = self._prefilter(results)
        self.memory.add_candidates(results)
        self.memory.tried_queries.append(query)
        self.emit("candidates_found", len(results))

        # Store for the agent to reference
        self._pending_candidates.extend(results)

        candidates_out = []
        for c in results:
            entry: Dict[str, Any] = {
                "url": c.url,
                "title": c.title,
                "platform": c.platform,
            }
            if c.duration:
                entry["duration"] = c.duration
            candidates_out.append(entry)

        return json.dumps({
            "candidates_found": len(candidates_out),
            "candidates": candidates_out,
        })

    def _tool_process_video(self, url: str, title: str = "") -> str:
        """process_video tool: download → keyframes → vision → auto-clip."""
        if not url:
            return json.dumps({"error": "No URL provided"})

        # Build a VideoCandidate from the URL
        candidate = None
        for c in self._pending_candidates:
            if c.url == url:
                candidate = c
                break

        if not candidate:
            from .search import _detect_platform
            candidate = VideoCandidate(
                url=url,
                title=title or "Unknown",
                platform=_detect_platform(url),
            )

        self.emit("evaluating", candidate)
        self._evaluate_candidate(candidate, self._request)

        # Build result summary
        # Check if the URL ended up accepted or rejected
        if any(clip.source_url == url for clip in self.memory.accepted_clips):
            accepted = [c for c in self.memory.accepted_clips if c.source_url == url]
            total_clip_dur = sum(c.duration for c in accepted)
            return json.dumps({
                "matched": True,
                "clips_created": len(accepted),
                "clip_duration": round(total_clip_dur, 1),
                "confidence": round(accepted[0].confidence, 2) if accepted else 0,
                "total_collected": round(self.memory.total_duration, 1),
                "target": self._request.target_duration,
            })
        else:
            reason = self.memory.rejected.get(url, "Unknown reason")
            return json.dumps({
                "matched": False,
                "reason": reason[:200],
                "total_collected": round(self.memory.total_duration, 1),
                "target": self._request.target_duration,
            })

    def _tool_check_progress(self) -> str:
        """check_progress tool: return current harvest state."""
        ctx = self.memory.get_llm_context()
        return json.dumps({
            "total_collected": round(self.memory.total_duration, 1),
            "target": self._request.target_duration if self._request else 0,
            "clips_accepted": len(self.memory.accepted_clips),
            "candidates_evaluated": len(self.memory.found_candidates),
            "candidates_rejected": len(self.memory.rejected),
            "queries_tried": self.memory.tried_queries,
            "summary": ctx,
        })

    def _tool_finish_harvest(self, reason: str) -> str:
        """finish_harvest tool: signal completion."""
        self._stop = True
        self.emit("log", f"Agent finished: {reason}")
        return json.dumps({
            "status": "harvest_complete",
            "reason": reason,
            "total_collected": round(self.memory.total_duration, 1),
            "clips": len(self.memory.accepted_clips),
        })

    def _tool_refine_criteria(self, criteria: List[str], reasoning: str) -> str:
        """refine_criteria tool: update what the vision model looks for."""
        if not criteria:
            return json.dumps({"error": "criteria list cannot be empty"})
        old_criteria = list(self._effective_criteria)
        self._effective_criteria = criteria
        self.emit("log", f"Criteria refined: {old_criteria} -> {criteria} ({reasoning})")
        return json.dumps({
            "status": "criteria_updated",
            "old_criteria": old_criteria,
            "new_criteria": criteria,
            "reasoning": reasoning,
        })

    def _tool_adjust_confidence(self, min_confidence: float) -> str:
        """adjust_confidence tool: change the acceptance threshold."""
        clamped = max(0.1, min(0.9, min_confidence))
        old = self._effective_confidence
        self._effective_confidence = clamped
        self.emit("log", f"Confidence threshold: {old:.2f} -> {clamped:.2f}")
        return json.dumps({
            "status": "confidence_updated",
            "old_threshold": round(old, 2),
            "new_threshold": round(clamped, 2),
        })

    def _tool_get_rejection_analysis(self) -> str:
        """get_rejection_analysis tool: group rejections by pattern."""
        categories: Dict[str, List[str]] = {
            "content_mismatch": [],
            "low_confidence": [],
            "download_failure": [],
            "platform_error": [],
            "duration_issue": [],
            "other": [],
        }

        for url, reason in self.memory.rejected.items():
            lower = reason.lower()
            if any(kw in lower for kw in (
                "no match", "not found", "doesn't match", "does not match",
                "no relevant", "content mismatch", "no .* found",
            )):
                categories["content_mismatch"].append(reason)
            elif any(kw in lower for kw in ("confidence", "low confidence")):
                categories["low_confidence"].append(reason)
            elif any(kw in lower for kw in (
                "download", "timeout", "connection", "network", "http",
            )):
                categories["download_failure"].append(reason)
            elif any(kw in lower for kw in (
                "unsupported", "platform", "unavailable", "private",
                "removed", "blocked", "geo", "age", "skipping",
            )):
                categories["platform_error"].append(reason)
            elif any(kw in lower for kw in ("too long", "too short", "duration")):
                categories["duration_issue"].append(reason)
            else:
                categories["other"].append(reason)

        summary: Dict[str, Any] = {}
        for cat, reasons in categories.items():
            if reasons:
                # Deduplicate example reasons
                unique = list(dict.fromkeys(reasons))
                summary[cat] = {
                    "count": len(reasons),
                    "examples": unique[:3],
                }

        return json.dumps({
            "total_rejected": len(self.memory.rejected),
            "current_confidence_threshold": round(self._effective_confidence, 2),
            "current_criteria": self._effective_criteria,
            "categories": summary,
        })

    def _tool_set_download_options(
        self, timeout: Optional[int] = None, retries: Optional[int] = None,
    ) -> str:
        """set_download_options tool: configure download timeout and retries."""
        changes: Dict[str, Any] = {}
        if timeout is not None:
            old_timeout = self._effective_timeout
            self._effective_timeout = max(10, min(600, timeout))
            changes["timeout"] = {"old": old_timeout, "new": self._effective_timeout}
        if retries is not None:
            old_retries = self._effective_retries
            self._effective_retries = max(1, min(5, retries))
            changes["retries"] = {"old": old_retries, "new": self._effective_retries}
        if not changes:
            return json.dumps({"error": "No options provided (specify timeout and/or retries)"})
        self.emit("log", f"Download options updated: {changes}")
        return json.dumps({"status": "download_options_updated", **changes})

    def _tool_adjust_frame_sampling(self, num_frames: int) -> str:
        """adjust_frame_sampling tool: change keyframe count for vision analysis."""
        clamped = max(4, min(24, num_frames))
        old = self.analyzer.num_frames
        self.analyzer.num_frames = clamped
        self.emit("log", f"Frame sampling: {old} -> {clamped} keyframes")
        return json.dumps({
            "status": "frame_sampling_updated",
            "old_num_frames": old,
            "new_num_frames": clamped,
        })

    def _tool_re_enable_platform(self, platform: str) -> str:
        """re_enable_platform tool: reset failure counter for a platform."""
        if not platform:
            return json.dumps({"error": "No platform specified"})
        old_fails = self._platform_fails.get(platform, 0)
        self._platform_fails[platform] = 0
        self.emit("log", f"Re-enabled platform: {platform} (was at {old_fails} failures)")
        return json.dumps({
            "status": "platform_re_enabled",
            "platform": platform,
            "previous_failures": old_fails,
        })

    _QUALITY_FORMATS = {
        "360p": "18/best[height<=360]/best",
        "480p": "18/22/best[height<=480]/best",
        "720p": "22/best[height<=720]/best",
        "1080p": "best[height<=1080]/best",
    }

    def _tool_set_video_quality(self, resolution: str) -> str:
        """set_video_quality tool: change yt-dlp format selector."""
        resolution = resolution.lower().strip()
        fmt = self._QUALITY_FORMATS.get(resolution)
        if not fmt:
            return json.dumps({
                "error": f"Unknown resolution '{resolution}'. Use: 360p, 480p, 720p, or 1080p",
            })
        self.clipper.format_override = fmt
        self.emit("log", f"Video quality set to {resolution}")
        return json.dumps({
            "status": "video_quality_updated",
            "resolution": resolution,
        })

    def _tool_set_max_video_duration(self, max_seconds: int) -> str:
        """set_max_video_duration tool: change per-video duration limit."""
        clamped = max(60, min(7200, max_seconds))
        old = self._effective_max_duration
        self._effective_max_duration = clamped
        self.emit("log", f"Max video duration: {old:.0f}s -> {clamped}s")
        return json.dumps({
            "status": "max_duration_updated",
            "old_max_seconds": round(old),
            "new_max_seconds": clamped,
        })

    # ── Legacy loop (fallback when tool calling unavailable) ─────────

    def _run_legacy_loop(self, request: HarvestRequest) -> None:
        """Original hardcoded loop: generate queries → search → evaluate."""
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
                info = self.clipper.get_video_info(candidate.url)
                vid_duration = info.get("duration") or 0
                self.emit("info_ready", candidate, vid_duration)

                if vid_duration > self._effective_max_duration:
                    reason = f"Too long ({vid_duration:.0f}s > {self._effective_max_duration:.0f}s limit)"
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
                timeout=self._effective_timeout,
                retries=self._effective_retries,
            )
            self.emit("downloaded", candidate)

            # Keyframe extraction → grid mosaic
            self.emit("extracting_frames", candidate)
            frames = self.analyzer.extract_keyframes(video_path)
            self.emit("keyframes_extracted", self.analyzer.num_frames)

            # Duration
            try:
                duration = self.clipper.get_duration(video_path)
            except Exception:
                duration = candidate.duration or 60.0

            # LLM vision analysis (single grid image)
            # Use effective criteria (may have been refined by agent)
            active_criteria = self._effective_criteria or request.content_criteria
            self.emit("analyzing", candidate, self.analyzer.num_frames)
            analysis = self.analyzer.analyze(
                frames, active_criteria, duration,
                video_title=candidate.title,
            )
            self.emit("analysis_complete", analysis)

            if (
                analysis.matches
                and analysis.confidence >= self._effective_confidence
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
