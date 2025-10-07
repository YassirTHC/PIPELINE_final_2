import json

import pipeline_core.llm_service as llm_module
from pipeline_core.llm_service import LLMMetadataGeneratorService


def test_llm_stream_fallback_once(monkeypatch):
    monkeypatch.setenv("PIPELINE_FAST_TESTS", "1")
    monkeypatch.delenv("PIPELINE_LLM_FORCE_NON_STREAM", raising=False)
    llm_module._reset_stream_state_for_tests()

    error_line = "data: " + json.dumps({"error": "boom"})
    ok_line = "data: " + json.dumps({"response": "stream text"})
    done_line = "data: " + json.dumps({"done": True})

    class DummyResponse:
        def __init__(self, lines):
            self._lines = lines

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self, decode_unicode=True):
            for line in self._lines:
                yield line

    calls = {"stream": 0, "sync": 0}

    def fake_post(*_args, **_kwargs):
        calls["stream"] += 1
        if calls["stream"] == 1:
            return DummyResponse([error_line])
        return DummyResponse([ok_line, done_line])

    def fake_sync(endpoint, model, prompt, options):
        calls["sync"] += 1
        return f"blocking-{calls['sync']}"

    monkeypatch.setattr(llm_module.requests, "post", fake_post)
    monkeypatch.setattr(llm_module, "_ollama_generate_sync", fake_sync)

    service = LLMMetadataGeneratorService(reuse_shared=False)

    text1, reason1, chunk1 = service._complete_text("hello", max_tokens=32, purpose="dynamic")
    assert text1.startswith("blocking-1")
    assert reason1 in {"", "stream_err", "timeout"} or reason1.startswith("error:")
    assert calls["stream"] == 1
    assert calls["sync"] == 1

    text2, reason2, chunk2 = service._complete_text("second", max_tokens=32, purpose="dynamic")
    assert text2.startswith("blocking-2")
    assert reason2 in {"", "stream_err", "timeout"} or reason2.startswith("error:")
    assert calls["sync"] == 2
    assert calls["stream"] == 1

    path_mode, path_reason = llm_module._current_llm_path()
    assert path_mode == "segment_blocking"
    assert path_reason in {"stream_err", "timeout"}
