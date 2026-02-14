"""Tests for video search utilities."""

from lelouch.agents.search import _parse_duration


class TestParseDuration:
    def test_mm_ss(self):
        assert _parse_duration("3:45") == 225.0

    def test_hh_mm_ss(self):
        assert _parse_duration("1:02:30") == 3750.0

    def test_seconds_only(self):
        assert _parse_duration("90") == 90.0

    def test_none(self):
        assert _parse_duration(None) is None

    def test_empty(self):
        assert _parse_duration("") is None

    def test_invalid(self):
        assert _parse_duration("abc") is None

    def test_with_whitespace(self):
        assert _parse_duration("  2:30  ") == 150.0
