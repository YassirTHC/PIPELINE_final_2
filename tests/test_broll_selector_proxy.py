from __future__ import annotations

import pytest


class _DummySelector:
    def __init__(self):
        self.select_kwargs = None

    def normalize_keywords(self, keywords):
        # ensure original keywords forwarded
        assert keywords == ["Sunset", "City"]
        return {"sunset", "city"}

    def expand_keywords(self, keywords):
        # find_broll_matches doit nous passer une liste exploitable
        assert isinstance(keywords, list)
        assert set(keywords) == {"sunset", "city"}
        return ["sunset", "city", "evening skyline"]

    def select_brolls(self, **kwargs):
        self.select_kwargs = kwargs
        return {
            "selected": [
                {
                    "score": 0.87,
                    "asset": {
                        "file_path": "clips/sunset.mp4",
                        "duration": 4.2,
                        "tags": ["sunset", "city"],
                        "source": "pexels",
                    },
                }
            ]
        }


def test_find_broll_matches_forwards_keywords_and_formats(monkeypatch):
    selector = _DummySelector()

    # empêcher l'initialisation lourde et renvoyer notre stub
    module = pytest.importorskip("broll_selector")
    monkeypatch.setattr(module, "get_broll_selector", lambda *_, **__: selector)

    matches = module.find_broll_matches(["Sunset", "City"], max_count=2, min_duration=1.0, max_duration=8.0)

    assert selector.select_kwargs == {"keywords": ["sunset", "city", "evening skyline"], "desired_count": 2}
    assert matches == [
        {
            "file_path": "clips/sunset.mp4",
            "duration": 4.2,
            "score": 0.87,
            "tags": ["sunset", "city"],
            "source": "pexels",
        }
    ]
