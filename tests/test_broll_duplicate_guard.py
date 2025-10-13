from types import SimpleNamespace

from video_processor import _candidate_used_before, _remember_selected_candidate


def test_candidate_used_before_detects_string_url():
    candidate = SimpleNamespace(url="https://cdn.example.com/a.mp4")
    used = {"https://cdn.example.com/a.mp4"}

    assert _candidate_used_before(candidate, used) is True


def test_candidate_used_before_ignores_non_string_urls():
    candidate = SimpleNamespace(url=None)
    used = {"https://cdn.example.com/a.mp4"}

    assert _candidate_used_before(candidate, used) is False


def test_remember_selected_candidate_registers_new_url():
    candidate = SimpleNamespace(url="https://cdn.example.com/b.mp4")
    used = set()

    _remember_selected_candidate(used, candidate)

    assert "https://cdn.example.com/b.mp4" in used


def test_remember_selected_candidate_skips_empty_values():
    candidate = SimpleNamespace(url="")
    used = {"https://cdn.example.com/a.mp4"}

    _remember_selected_candidate(used, candidate)

    assert used == {"https://cdn.example.com/a.mp4"}
