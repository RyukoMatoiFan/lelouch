"""Video downloading (yt-dlp) and clipping (ffmpeg)."""

from __future__ import annotations

import json
import logging
import subprocess
import warnings
from pathlib import Path

import yt_dlp

# Suppress ResourceWarning from yt-dlp's unclosed SSL sockets
warnings.filterwarnings("ignore", category=ResourceWarning, module="yt_dlp")
warnings.filterwarnings("ignore", category=ResourceWarning, module="http.cookiejar")

from ..models.harvest import ClipResult

logger = logging.getLogger(__name__)

# Max video duration we'll bother downloading (seconds)
MAX_VIDEO_DURATION = 600  # 10 minutes

# Prefer pre-merged single-stream formats (no ffmpeg merge needed).
# 18 = YouTube 360p MP4, 22 = YouTube 720p MP4.
# Falls back to best single stream ≤480p, then anything.
_FORMAT = "18/22/best[height<=480]/best"


class _YDLLogger:
    """Redirect all yt-dlp output to Python logging instead of stdout/stderr."""

    def debug(self, msg):
        if msg.startswith("[download]"):
            return  # suppress noisy progress lines
        logger.debug("yt-dlp: %s", msg)

    def info(self, msg):
        logger.debug("yt-dlp: %s", msg)

    def warning(self, msg):
        logger.debug("yt-dlp: %s", msg)

    def error(self, msg):
        logger.debug("yt-dlp: %s", msg)

_ydl_logger = _YDLLogger()


class VideoClipper:
    """Download videos with yt-dlp and clip segments with ffmpeg."""

    def __init__(self):
        self._progress_hook = None
        self.format_override: str | None = None

    def set_progress_hook(self, hook):
        """Set a callback for download progress: hook(downloaded_mb, pct, speed, eta)."""
        self._progress_hook = hook

    def get_video_info(self, url: str) -> dict:
        """Extract video info (duration, title, formats) without downloading.

        Returns the info dict or raises RuntimeError.
        """
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "no_check_certificates": True,
            "socket_timeout": 15,
            "logger": _ydl_logger,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                raise RuntimeError("No info extracted")
            return info

    def download_video(
        self, url: str, output_path: str, timeout: int = 120, retries: int = 1,
    ) -> str:
        """Download a video using yt-dlp.

        Returns the path to the downloaded file.
        Raises RuntimeError if the download fails or produces an empty file.
        """
        fmt = self.format_override or _FORMAT
        ydl_opts = {
            "format": fmt,
            "outtmpl": output_path,
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": min(timeout, 20),
            "retries": retries,
            "fragment_retries": retries,
            "extractor_retries": retries,
            "file_access_retries": retries,
            "no_check_certificates": True,
            "logger": _ydl_logger,
        }

        if self._progress_hook:
            ydl_opts["progress_hooks"] = [self._ydl_progress_adapter]

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        actual_path = self._find_downloaded_file(output_path)
        p = Path(actual_path)
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeError("Downloaded file is empty or missing")

        return actual_path

    def _ydl_progress_adapter(self, d: dict) -> None:
        """Translate yt-dlp progress dict to our hook callback."""
        if not self._progress_hook:
            return
        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            pct = (downloaded / total * 100) if total else -1
            speed = d.get("speed") or 0
            eta = d.get("eta") or 0
            downloaded_mb = downloaded / (1024 * 1024)
            self._progress_hook(downloaded_mb, pct, speed, eta)
        elif status == "finished":
            total = d.get("total_bytes") or d.get("downloaded_bytes") or 0
            downloaded_mb = total / (1024 * 1024)
            self._progress_hook(downloaded_mb, 100, 0, 0)

    @staticmethod
    def _find_downloaded_file(output_path: str) -> str:
        """Locate the file yt-dlp actually wrote (it may change the extension)."""
        base = Path(output_path)
        if base.exists():
            return str(base)
        for ext in (".mp4", ".webm", ".mkv", ".flv", ".mp4.part"):
            candidate = base.with_suffix(ext)
            if candidate.exists():
                return str(candidate)
        matches = list(base.parent.glob(f"{base.stem}.*"))
        if matches:
            return str(matches[0])
        return output_path

    def clip(
        self,
        input_path: str,
        output_path: str,
        start: float,
        end: float,
    ) -> str:
        """Trim a video segment using ffmpeg (stream copy, fast)."""
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", f"{start:.2f}",
            "-to", f"{end:.2f}",
            "-i", input_path,
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("ffmpeg clip failed: %s", result.stderr[-500:])
            raise RuntimeError(f"ffmpeg clip failed: {result.stderr[-200:]}")
        return output_path

    def get_duration(self, path: str) -> float:
        """Get video duration in seconds via ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr[-200:]}")
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])

    def make_clip_result(
        self,
        source_url: str,
        source_title: str,
        clip_path: str,
        start: float,
        end: float,
        confidence: float,
    ) -> ClipResult:
        """Create a ClipResult, measuring actual duration from the file."""
        try:
            duration = self.get_duration(clip_path)
        except Exception:
            duration = end - start
        return ClipResult(
            source_url=source_url,
            source_title=source_title,
            clip_path=clip_path,
            duration=duration,
            start_time=start,
            end_time=end,
            confidence=confidence,
        )
