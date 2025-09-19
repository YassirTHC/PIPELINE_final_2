from types import SimpleNamespace

from pipeline_core.llm_service import LLMMetadataGeneratorService


def test_llm_single_init(monkeypatch):
    calls = {"init": 0}

    def fake_integration(cfg):
        calls["init"] += 1
        return object()

    monkeypatch.setattr(
        "pipeline_core.llm_service.create_pipeline_integration",
        fake_integration,
        raising=True,
    )

    cfg = SimpleNamespace(llm=SimpleNamespace())
    a = LLMMetadataGeneratorService.get_shared(cfg)
    b = LLMMetadataGeneratorService.get_shared(cfg)

    assert a is not None and b is not None
    assert calls["init"] == 1
    assert a._get_integration() is b._get_integration()
