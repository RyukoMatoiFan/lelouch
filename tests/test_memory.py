"""Tests for HarvestMemory state tracking."""

import json
import tempfile
from pathlib import Path

from lelouch.agents.harvest_memory import HarvestMemory
from lelouch.models.harvest import ClipResult, VideoCandidate


def _make_clip(url: str = "http://example.com/v1", duration: float = 10.0) -> ClipResult:
    return ClipResult(
        source_url=url,
        source_title="Test Video",
        clip_path="/tmp/clip.mp4",
        duration=duration,
        start_time=0.0,
        end_time=duration,
        confidence=0.9,
    )


def _make_candidate(url: str = "http://example.com/v1") -> VideoCandidate:
    return VideoCandidate(url=url, title="Test", source_query="test query")


class TestHarvestMemory:
    def test_add_accepted_updates_duration(self):
        mem = HarvestMemory()
        clip = _make_clip(duration=12.5)
        mem.add_accepted(clip, source_query="anime blonde")
        assert mem.total_duration == 12.5
        assert len(mem.accepted_clips) == 1

    def test_add_rejected(self):
        mem = HarvestMemory()
        mem.add_rejected("http://example.com/bad", "no match")
        assert "http://example.com/bad" in mem.rejected
        assert mem.rejected["http://example.com/bad"] == "no match"

    def test_is_url_seen_candidates(self):
        mem = HarvestMemory()
        c = _make_candidate("http://example.com/v1")
        mem.add_candidates([c])
        assert mem.is_url_seen("http://example.com/v1")
        assert not mem.is_url_seen("http://example.com/other")

    def test_is_url_seen_rejected(self):
        mem = HarvestMemory()
        mem.add_rejected("http://example.com/bad", "no match")
        assert mem.is_url_seen("http://example.com/bad")

    def test_successful_queries(self):
        mem = HarvestMemory()
        mem.add_accepted(_make_clip(), source_query="anime blonde")
        mem.add_accepted(_make_clip(), source_query="anime blonde")
        mem.add_accepted(_make_clip(), source_query="2d girl")
        assert set(mem.successful_queries()) == {"anime blonde", "2d girl"}

    def test_common_rejection_reasons(self):
        mem = HarvestMemory()
        mem.add_rejected("u1", "no match")
        mem.add_rejected("u2", "no match")
        mem.add_rejected("u3", "download failed")
        reasons = mem.common_rejection_reasons(2)
        assert reasons[0] == "no match"

    def test_get_llm_context_contains_key_info(self):
        mem = HarvestMemory()
        mem.iteration = 3
        mem.total_duration = 45.0
        mem.tried_queries = ["query1", "query2"]
        ctx = mem.get_llm_context()
        assert "45.0s" in ctx
        assert "query1" in ctx

    def test_save_and_load_roundtrip(self):
        mem = HarvestMemory()
        mem.tried_queries = ["q1", "q2"]
        mem.add_accepted(_make_clip(duration=15.0), source_query="q1")
        mem.add_rejected("http://bad.com", "no match")
        mem.iteration = 5
        mem.total_duration = 15.0

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        mem.save(path)
        loaded = HarvestMemory.load(path)

        assert loaded.tried_queries == ["q1", "q2"]
        assert len(loaded.accepted_clips) == 1
        assert loaded.accepted_clips[0].duration == 15.0
        assert loaded.rejected == {"http://bad.com": "no match"}
        assert loaded.iteration == 5
        assert loaded.total_duration == 15.0

        Path(path).unlink()
