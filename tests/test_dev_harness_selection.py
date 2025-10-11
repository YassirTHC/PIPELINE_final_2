import os, subprocess, sys
def test_llm_paths_and_heuristic():
    env = os.environ.copy()
    env["VP_DEV_TESTS"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run([sys.executable, "video_processor.py"], capture_output=True, text=True, env=env)
    out = (p.stdout or "") + (p.stderr or "")
    assert p.returncode == 0
    assert out.count("Source métadonnées retenue: llm") >= 2
    assert "Auto-generated Clip Title" not in out
    assert "Titre fallback:" not in out
