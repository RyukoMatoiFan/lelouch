"""DuckDuckGo text search filtered for video platform URLs."""

from __future__ import annotations

import logging
import re
import time
from typing import List
from urllib.parse import urlparse

from ddgs import DDGS

from ..config import SAFESEARCH, SEARCH_REGION
from ..models.harvest import VideoCandidate

logger = logging.getLogger(__name__)

# Domains that host downloadable video content (yt-dlp supported)
_VIDEO_DOMAINS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be",
    "vimeo.com", "www.vimeo.com",
    # dailymotion excluded — consistently gives connection errors
    "rutube.ru", "www.rutube.ru",
    "vkvideo.ru", "www.vkvideo.ru", "vk.com",
    "bilibili.com", "www.bilibili.com",
    "tiktok.com", "www.tiktok.com",
    "twitch.tv", "www.twitch.tv",
    "streamable.com",
    "rumble.com", "www.rumble.com",
    "odysee.com", "www.odysee.com",
    "nicovideo.jp", "www.nicovideo.jp",
    "peertube.tv",
}

# Paths that indicate non-video pages (channels, playlists, categories)
_REJECT_PATH_PATTERNS = [
    re.compile(r"^/@"),                     # channel pages (@username)
    re.compile(r"/playlist\b"),             # youtube playlists
    re.compile(r"/plst/"),                  # rutube playlists
    re.compile(r"/channel/"),              # youtube channels
    re.compile(r"/c/"),                    # youtube custom channels
    re.compile(r"/user/"),                 # youtube user pages
    re.compile(r"/results\b"),             # search results pages
    re.compile(r"/hashtag/"),              # hashtag pages
    re.compile(r"/trending"),              # trending pages
    re.compile(r"/feed"),                  # feed pages
    re.compile(r"/music$"),                # vkvideo /music
    re.compile(r"/discover/"),             # tiktok discover/category pages
    re.compile(r"/tag/"),                  # tiktok/other tag pages
    re.compile(r"/search"),               # search pages
    re.compile(r"/embed"),                # embedded player pages
    re.compile(r"/shorts$"),              # youtube shorts listing
    re.compile(r"/live$"),                # live stream listings
]


def _is_video_url(url: str) -> bool:
    """Check if a URL points to a video page on a known platform."""
    try:
        parsed = urlparse(url)
        domain = parsed.hostname or ""
        # Strip www. / m. for matching
        bare = domain
        for prefix in ("www.", "m."):
            if bare.startswith(prefix):
                bare = bare[len(prefix):]
        # Check domain
        if domain not in _VIDEO_DOMAINS and bare not in _VIDEO_DOMAINS:
            return False
        path = parsed.path
        # Reject bare domain / homepage
        if len(path) <= 1:
            return False
        # youtu.be is always a video link
        if bare == "youtu.be":
            return True
        # Reject known non-video paths
        for pattern in _REJECT_PATH_PATTERNS:
            if pattern.search(path):
                # Exception: /@user/video/123 is a valid TikTok video
                if pattern.pattern == r"^/@" and "/video/" in path:
                    continue
                return False
        # VK: only accept paths containing /video (not community pages)
        if bare == "vk.com" and "/video" not in path:
            return False
        return True
    except Exception:
        return False


def _detect_platform(url: str) -> str:
    """Extract platform name from URL."""
    try:
        host = urlparse(url).hostname or ""
        host = host.lstrip("www.").lstrip("m.")
        parts = host.split(".")
        return parts[0] if parts else "unknown"
    except Exception:
        return "unknown"


def _parse_duration(duration_str: str | None) -> float | None:
    """Parse a duration string like '3:45' or '1:02:30' into seconds."""
    if not duration_str:
        return None
    try:
        parts = duration_str.strip().split(":")
        parts = [int(p) for p in parts]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 1:
            return float(parts[0])
    except (ValueError, IndexError):
        pass
    return None


class VideoSearcher:
    """Search for videos using DuckDuckGo text search."""

    def __init__(self):
        self._last_request_time = 0.0
        self._min_delay = 2.0  # seconds between DDG requests
        self._max_retries = 3

    def search(self, query: str, max_results: int = 10) -> List[VideoCandidate]:
        """Text-search DDG, filter results for video platform URLs."""
        # Request more results than needed since we filter non-video URLs
        fetch_count = max_results * 4

        for attempt in range(self._max_retries):
            self._rate_limit()
            try:
                results = list(DDGS().text(
                    query,
                    safesearch=SAFESEARCH,
                    region=SEARCH_REGION,
                    max_results=fetch_count,
                ))
                break
            except Exception as e:
                err_str = str(e).lower()
                if "ratelimit" in err_str or "429" in err_str:
                    wait = self._min_delay * (2 ** attempt)
                    logger.info("Rate limited on %r, retrying in %.0fs (%d/%d)",
                                query, wait, attempt + 1, self._max_retries)
                    time.sleep(wait)
                    continue
                logger.warning("Search failed for %r: %s", query, e)
                return []
        else:
            logger.warning("All retries exhausted for %r", query)
            return []

        # Filter for video URLs and convert
        candidates = []
        for r in results:
            url = r.get("href", "")
            if _is_video_url(url):
                candidates.append(VideoCandidate(
                    url=url,
                    title=r.get("title", "Untitled"),
                    description=r.get("body"),
                    platform=_detect_platform(url),
                    source_query=query,
                ))
                if len(candidates) >= max_results:
                    break

        logger.info("Search %r: %d results -> %d video candidates",
                     query, len(results), len(candidates))
        return candidates

    def _rate_limit(self) -> None:
        """Enforce minimum delay between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_delay:
            time.sleep(self._min_delay - elapsed)
        self._last_request_time = time.monotonic()
