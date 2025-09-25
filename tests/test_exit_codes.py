
import run_pipeline
from pipeline_core.runtime import PipelineResult


def _make_result(final_ok: bool, errors: list[str] | None = None) -> PipelineResult:
    result = PipelineResult()
    result.final_export_ok = final_ok
    if errors:
        result.errors.extend(errors)
    result.finish()
    return result


def test_exit_code_success(monkeypatch):
    expected = _make_result(True, [])

    def fake_main(argv, *, return_result=False):
        assert return_result is True
        return expected

    monkeypatch.setattr(run_pipeline.video_processor, "main", fake_main)

    code = run_pipeline.main(["--video", "clip.mp4"])
    assert code == 0


def test_exit_code_with_errors(monkeypatch):
    expected = _make_result(True, ["fallback"])

    def fake_main(argv, *, return_result=False):
        assert return_result is True
        return expected

    monkeypatch.setattr(run_pipeline.video_processor, "main", fake_main)

    code = run_pipeline.main(["--video", "clip.mp4"])
    assert code == 2


def test_exit_code_export_failure(monkeypatch):
    expected = _make_result(False, [])

    def fake_main(argv, *, return_result=False):
        assert return_result is True
        return expected

    monkeypatch.setattr(run_pipeline.video_processor, "main", fake_main)

    code = run_pipeline.main(["--video", "clip.mp4"])
    assert code == 1

