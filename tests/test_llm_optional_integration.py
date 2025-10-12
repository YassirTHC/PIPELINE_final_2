import importlib.machinery

import pipeline_core.llm_service as llm_module


class _BrokenLoader:
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        exc = ModuleNotFoundError("sklearn")
        exc.name = "sklearn"
        raise exc


def test_missing_optional_dependency_falls_back(monkeypatch, caplog):
    original_spec = llm_module.importlib.util.spec_from_file_location

    def fake_spec(name, path, submodule_search_locations=None):
        if name == "utils.pipeline_integration":
            loader = _BrokenLoader()
            spec = importlib.machinery.ModuleSpec(name, loader)
            spec.loader = loader
            spec.origin = str(path)
            return spec
        return original_spec(name, path, submodule_search_locations=submodule_search_locations)

    monkeypatch.setattr(llm_module.importlib.util, "spec_from_file_location", fake_spec)

    caplog.clear()
    with caplog.at_level("WARNING", logger="pipeline_core.llm_service"):
        factory = llm_module._load_integration_factory(initial_fast_tests=False)

    assert callable(factory)
    assert factory.__name__ == "_fallback_stub"
    assert "Optional pipeline integration disabled (missing dependency: sklearn)" in caplog.text
    assert "Falling back to stub pipeline integration (module load failed)" in caplog.text
