import os

import pipeline_core.llm_service as llm_module
from pipeline_core.llm_service import LLMMetadataGeneratorService


def test_llm_segment_blocking(monkeypatch):
    monkeypatch.setenv("PIPELINE_FAST_TESTS", "1")
    monkeypatch.setenv("PIPELINE_LLM_FORCE_NON_STREAM", "1")
    llm_module._reset_stream_state_for_tests()

    calls = {"sync": 0}

    def fake_sync(endpoint, model, prompt, options):
        calls["sync"] += 1
        return "non_stream_payload"

    monkeypatch.setattr(llm_module, "_ollama_generate_sync", fake_sync)

    service = LLMMetadataGeneratorService(reuse_shared=False)
    text, reason, chunk_count = service._complete_text("hello", max_tokens=16, purpose="dynamic")

    assert text == "non_stream_payload"
    assert reason == ""
    assert chunk_count == 0
    assert calls["sync"] == 1

    path_mode, path_reason = llm_module._current_llm_path()
    assert path_mode == "segment_blocking"
    assert path_reason == "flag_disable"
