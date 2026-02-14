"""Tests for the VideoHarvester prefilter and result building."""

from unittest.mock import patch

from lelouch.agents.harvester import VideoHarvester
from lelouch.config import Config
from lelouch.models.harvest import HarvestRequest, VideoCandidate


def _make_candidate(url: str, duration: float = 30.0) -> VideoCandidate:
    return VideoCandidate(url=url, title=f"Video {url}", duration=duration)


class FakeLLM:
    """Minimal stub for LLMClient."""

    def generate(self, prompt, system_prompt=None):
        return '["test query"]'

    def generate_json(self, prompt, system_prompt=None):
        return ["test query"]

    def generate_with_images(self, prompt, image_paths, system_prompt=None):
        return '{"matches": false, "confidence": 0.0, "reason": "stub"}'


class TestPrefilter:
    def setup_method(self):
        self.config = Config()
        self.harvester = VideoHarvester(self.config, FakeLLM())

    def test_removes_duplicates(self):
        candidates = [
            _make_candidate("http://a.com"),
            _make_candidate("http://a.com"),
            _make_candidate("http://b.com"),
        ]
        filtered = self.harvester._prefilter(candidates)
        urls = [c.url for c in filtered]
        assert urls == ["http://a.com", "http://b.com"]

    def test_removes_seen_urls(self):
        self.harvester.memory.add_rejected("http://a.com", "bad")
        candidates = [
            _make_candidate("http://a.com"),
            _make_candidate("http://b.com"),
        ]
        filtered = self.harvester._prefilter(candidates)
        assert len(filtered) == 1
        assert filtered[0].url == "http://b.com"

    @patch("lelouch.agents.harvester.MIN_CLIP_DURATION", 5.0)
    def test_removes_short_videos(self):
        self.harvester = VideoHarvester(self.config, FakeLLM())
        candidates = [
            _make_candidate("http://a.com", duration=2.0),
            _make_candidate("http://b.com", duration=10.0),
        ]
        filtered = self.harvester._prefilter(candidates)
        assert len(filtered) == 1
        assert filtered[0].url == "http://b.com"

    def test_skips_empty_urls(self):
        candidates = [
            _make_candidate(""),
            _make_candidate("http://b.com"),
        ]
        filtered = self.harvester._prefilter(candidates)
        assert len(filtered) == 1

    @patch("lelouch.agents.harvester.MAX_CANDIDATES_PER_ITERATION", 2)
    def test_respects_max_candidates(self):
        self.harvester = VideoHarvester(self.config, FakeLLM())
        candidates = [_make_candidate(f"http://{i}.com") for i in range(10)]
        filtered = self.harvester._prefilter(candidates)
        assert len(filtered) == 2


class TestBuildResult:
    def test_build_result_empty(self):
        config = Config()
        harvester = VideoHarvester(config, FakeLLM())
        request = HarvestRequest(
            description="test", target_duration=60.0, content_criteria=["test"]
        )
        result = harvester._build_result(request)
        assert result.total_duration == 0.0
        assert result.clips == []
        assert result.request is request
