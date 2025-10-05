import json
from typing import Any, Dict, Iterable

import pipeline_core.llm_service as llm_module
from pipeline_core.llm_service import LLMMetadataGeneratorService


class _DummyStreamResponse:
    def __init__(self, lines: Iterable[str]):
        self._lines = list(lines)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode: bool = True):
        for line in self._lines:
            yield line


class _DummyUrlopenResponse:
    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def _make_stream_events(events: Iterable[Dict[str, Any]]):
    for event in events:
        yield "data: " + json.dumps(event)


def test_stream_empty_triggers_non_stream(monkeypatch, caplog):
    def fake_post(*args, **kwargs):
        events = _make_stream_events([{}, {"done": True}])
        return _DummyStreamResponse(events)

    urlopen_calls = []

    def fake_urlopen(*args, **kwargs):
        urlopen_calls.append((args, kwargs))
        return _DummyUrlopenResponse({"response": "OK"})

    monkeypatch.setattr(llm_module.requests, "post", fake_post)
    monkeypatch.setattr(llm_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_module.time, "sleep", lambda *_: None)

    caplog.clear()
    with caplog.at_level("INFO", logger="pipeline_core.llm_service"):
        text, reason, chunk_count, raw_len, attempts = llm_module._ollama_generate_text(
            "prompt", model="demo", options={"num_predict": 16, "temperature": 0.2, "top_p": 0.9, "repeat_penalty": 1.1}, timeout=5.0
        )

    assert text == "OK"
    assert reason == ""
    assert chunk_count == 0
    assert raw_len == len("OK")
    assert attempts == 1
    assert urlopen_calls, "expected non-streaming helper to be invoked"
    assert "non-streaming fallback) ok" in caplog.text


def test_min_chars_guard_propagates_when_fallback_empty(monkeypatch, caplog):
    def fake_post(*args, **kwargs):
        events = _make_stream_events([{}, {"done": True}])
        return _DummyStreamResponse(events)

    monkeypatch.setenv("PIPELINE_LLM_MIN_CHARS", "9999")

    def fake_urlopen(*args, **kwargs):
        return _DummyUrlopenResponse({"response": ""})

    monkeypatch.setattr(llm_module.requests, "post", fake_post)
    monkeypatch.setattr(llm_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_module.time, "sleep", lambda *_: None)

    caplog.clear()
    with caplog.at_level("INFO", logger="pipeline_core.llm_service"):
        text, reason, chunk_count, raw_len, attempts = llm_module._ollama_generate_text(
            "prompt", model="demo", options={"num_predict": 16}, timeout=5.0
        )

    assert text == ""
    assert reason == "empty_payload"
    assert chunk_count == 0
    assert raw_len == 0
    assert attempts == 3
    assert "non-streaming fallback) ok" not in caplog.text


def test_dynamic_context_uses_non_stream_without_tfidf(monkeypatch, caplog):
    def fake_post(*args, **kwargs):
        events = _make_stream_events([{}, {"done": True}])
        return _DummyStreamResponse(events)

    def fake_urlopen(*args, **kwargs):
        payload = {
            "detected_domains": [],
            "language": "en",
            "keywords": ["alpha"],
            "synonyms": {},
            "search_queries": ["beta"],
            "segment_briefs": [],
        }
        return _DummyUrlopenResponse({"response": json.dumps(payload)})

    monkeypatch.setattr(llm_module.requests, "post", fake_post)
    monkeypatch.setattr(llm_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_module.time, "sleep", lambda *_: None)

    service = LLMMetadataGeneratorService(reuse_shared=False)

    caplog.clear()
    with caplog.at_level("INFO", logger="pipeline_core.llm_service"):
        result = service.generate_dynamic_context("transcript text")

    assert result["keywords"] == ["alpha"]
    assert result["search_queries"] == ["beta"]
    assert "non-streaming fallback) ok" in caplog.text
    assert "dynamic context fell back to TF-IDF" not in caplog.text
