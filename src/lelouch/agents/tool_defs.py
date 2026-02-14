"""Tool schemas for the harvest agent loop (OpenAI function calling format)."""

from __future__ import annotations

HARVEST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_videos",
            "description": (
                "Search the web for videos matching a query. "
                "Returns a list of video candidates (url, title, platform). "
                "Use diverse queries and vary phrasing between calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query. Always include the word 'video'. "
                            "Can use site:youtube.com for platform targeting."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "process_video",
            "description": (
                "Download a video, extract keyframes, run vision analysis, "
                "and auto-clip matching segments. Returns analysis result "
                "with clip info or rejection reason."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL of the video to process.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Title of the video (from search results).",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_progress",
            "description": (
                "Check current harvest progress: duration collected, "
                "clips accepted/rejected, queries tried, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_harvest",
            "description": (
                "Signal that the harvest is complete. Call this when the "
                "target duration is reached, returns are diminishing, "
                "or you've exhausted useful search strategies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why the harvest is being finished.",
                    },
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refine_criteria",
            "description": (
                "Update the content criteria used by vision analysis. "
                "Use when rejections show a criteria mismatch pattern "
                "(e.g. 'no X found' repeated 3+ times). You can broaden, "
                "narrow, or refocus what the vision model looks for."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "New list of content criteria for vision analysis. "
                            "Replaces the current criteria entirely."
                        ),
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why you are changing the criteria.",
                    },
                },
                "required": ["criteria", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_confidence",
            "description": (
                "Raise or lower the minimum confidence threshold for "
                "accepting clips. Lower if good videos are borderline "
                "rejected; raise if junk is getting through. Use sparingly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "min_confidence": {
                        "type": "number",
                        "description": (
                            "New minimum confidence threshold (0.1–0.9). "
                            "Default is 0.35."
                        ),
                    },
                },
                "required": ["min_confidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_rejection_analysis",
            "description": (
                "Get a detailed breakdown of why videos are being rejected, "
                "grouped by pattern. Use this before refining criteria or "
                "adjusting confidence — understand the problem first."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_download_options",
            "description": (
                "Configure download timeout and retry count. Increase retries "
                "for flaky connections, increase timeout for large/slow videos."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timeout": {
                        "type": "integer",
                        "description": (
                            "Download timeout in seconds (10–600). "
                            "Default is 120."
                        ),
                    },
                    "retries": {
                        "type": "integer",
                        "description": (
                            "Number of retry attempts per download (1–5). "
                            "Default is 1."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_frame_sampling",
            "description": (
                "Change how many keyframes are extracted for vision analysis. "
                "More frames = better coverage but slower/costlier analysis. "
                "Fewer frames = faster but may miss content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "num_frames": {
                        "type": "integer",
                        "description": (
                            "Number of keyframes to extract (4–24). "
                            "Default is 12."
                        ),
                    },
                },
                "required": ["num_frames"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "re_enable_platform",
            "description": (
                "Re-enable a platform that was auto-disabled after repeated "
                "failures. Use when you want to retry a platform that may "
                "have been temporarily down."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "platform": {
                        "type": "string",
                        "description": (
                            "Platform name to re-enable (e.g. 'youtube', "
                            "'dailymotion', 'vimeo')."
                        ),
                    },
                },
                "required": ["platform"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_video_quality",
            "description": (
                "Set preferred video quality for downloads. Higher quality "
                "gives better vision analysis but slower downloads."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resolution": {
                        "type": "string",
                        "description": (
                            "Max resolution: '360p', '480p', '720p', or '1080p'. "
                            "Default is '480p'. Higher = better analysis, slower download."
                        ),
                    },
                },
                "required": ["resolution"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_max_video_duration",
            "description": (
                "Set the maximum duration (in seconds) for individual videos "
                "to download. Videos longer than this are skipped. "
                "Auto-scales to target duration, but you can raise or lower it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_seconds": {
                        "type": "integer",
                        "description": (
                            "Maximum video duration in seconds (60–7200). "
                            "Default auto-scales to target duration."
                        ),
                    },
                },
                "required": ["max_seconds"],
            },
        },
    },
]
