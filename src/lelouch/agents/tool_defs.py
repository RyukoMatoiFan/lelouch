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
]
