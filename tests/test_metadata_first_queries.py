import pipeline_core.llm_service as llm_module
from pipeline_core.llm_service import LLMMetadataGeneratorService


def test_metadata_first_queries(monkeypatch):
    monkeypatch.setenv("PIPELINE_FAST_TESTS", "1")
    monkeypatch.setenv("PIPELINE_DISABLE_DYNAMIC_SEGMENT_LLM", "1")
    llm_module._reset_stream_state_for_tests()

    def fail_segment_json(*_args, **_kwargs):
        raise AssertionError("segment LLM should not be called in metadata-first mode")

    monkeypatch.setattr(LLMMetadataGeneratorService, "_segment_llm_json", fail_segment_json)

    service = LLMMetadataGeneratorService(reuse_shared=False)
    service.last_metadata = {
        "queries": ["Focus Session", "focus sessions", "deep work office"],
        "broll_keywords": ["people working", "worker", "workers"],
    }
    service._metadata_cache_queries = ["Focus Session", "focus sessions", "deep work office"]
    service._metadata_cache_keywords = ["people working", "worker", "workers"]

    result = service.generate_hints_for_segment("This focus session boosts deep work", 0.0, 5.0)

    assert result["source"] == "metadata_first"
    assert result["queries"]
    assert len(result["queries"]) <= 3
    lowered = {" ".join(q.split()).lower() for q in result["queries"]}
    assert len(lowered) == len(result["queries"])
