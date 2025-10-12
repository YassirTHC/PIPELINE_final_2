import types

import tools.llm_env_check as llm_env_check


def test_build_command_appends_extra_args():
    command = llm_env_check.build_pytest_command(["-k", "smoke"])
    assert command[:4] == [
        "pytest",
        "tests/test_llm_optional_integration.py",
        "tests/test_llm_service_fallback.py",
        "tests/test_run_pipeline_env.py",
    ]
    assert command[-2:] == ["-k", "smoke"]


def test_list_flag_prints_command(capsys):
    exit_code = llm_env_check.main(["--list"])
    assert exit_code == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("pytest ")
    for target in (
        "tests/test_llm_optional_integration.py",
        "tests/test_llm_service_fallback.py",
        "tests/test_run_pipeline_env.py",
    ):
        assert target in out


def test_main_runs_pytest_with_forwarded_args(monkeypatch):
    recorded: dict[str, object] = {}

    def fake_run(command, check):
        recorded["command"] = command
        recorded["check"] = check
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(llm_env_check.shutil, "which", lambda name: "/usr/bin/pytest")
    monkeypatch.setattr(llm_env_check.subprocess, "run", fake_run)

    exit_code = llm_env_check.main(["--", "-k", "fallback"])

    assert exit_code == 0
    assert recorded["command"][-2:] == ["-k", "fallback"]
    assert recorded["command"][0] == "pytest"
    assert recorded["check"] is False


def test_main_reports_missing_pytest(monkeypatch, capsys):
    monkeypatch.setattr(llm_env_check.shutil, "which", lambda name: None)

    exit_code = llm_env_check.main([])

    assert exit_code == 3
    captured = capsys.readouterr()
    assert "pytest executable not found" in captured.out
