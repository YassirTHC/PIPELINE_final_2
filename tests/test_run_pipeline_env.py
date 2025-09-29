import importlib
import json
import os
import sys
import types


import run_pipeline


def test_sanitize_env_trims_and_unquotes(monkeypatch):
    monkeypatch.setenv('PEXELS_API_KEY', '  "abc123"  ')
    monkeypatch.setenv('PIXABAY_API_KEY', "  'xyz789'  ")
    monkeypatch.setenv('BROLL_FETCH_PROVIDER', '  pexels  ')

    run_pipeline._sanitize_env_values(['PEXELS_API_KEY', 'PIXABAY_API_KEY', 'BROLL_FETCH_PROVIDER'])

    assert os.environ['PEXELS_API_KEY'] == 'abc123'
    assert os.environ['PIXABAY_API_KEY'] == 'xyz789'
    assert os.environ['BROLL_FETCH_PROVIDER'] == 'pexels'


def test_sanitize_env_removes_blank_values(monkeypatch):
    monkeypatch.setenv('PIXABAY_API_KEY', '   ')

    run_pipeline._sanitize_env_values(['PIXABAY_API_KEY'])

    assert 'PIXABAY_API_KEY' not in os.environ


def test_diag_missing_keys_writes_report(monkeypatch, tmp_path):
    monkeypatch.delenv('PEXELS_API_KEY', raising=False)
    monkeypatch.delenv('PIXABAY_API_KEY', raising=False)

    exit_code = run_pipeline._run_broll_diagnostic(tmp_path)

    assert exit_code == 2
    report = tmp_path / 'diagnostic_broll.json'
    assert report.exists()

    data = json.loads(report.read_text(encoding='utf-8'))
    assert data['providers'][0]['error'] == 'missing_api_key'
    assert data['providers'][1]['error'] == 'missing_api_key'


def test_import_does_not_load_dotenv():
    original_run_pipeline = sys.modules.pop('run_pipeline', None)
    original_dotenv = sys.modules.pop('dotenv', None)

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_load_dotenv(*args, **kwargs):
        calls.append((args, kwargs))
        return True

    fake_dotenv = types.SimpleNamespace(load_dotenv=fake_load_dotenv)
    sys.modules['dotenv'] = fake_dotenv

    imported = importlib.import_module('run_pipeline')

    assert imported.load_dotenv is fake_load_dotenv
    assert calls == []

    sys.modules.pop('run_pipeline', None)
    if original_dotenv is not None:
        sys.modules['dotenv'] = original_dotenv
    else:
        sys.modules.pop('dotenv', None)

    if original_run_pipeline is not None:
        sys.modules['run_pipeline'] = original_run_pipeline
    else:
        importlib.import_module('run_pipeline')
