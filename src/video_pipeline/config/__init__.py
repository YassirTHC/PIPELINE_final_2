import os
from pathlib import Path

from .settings import (
    Settings,
    load_settings as _raw_load_settings,
    log_effective_settings,
    reset_startup_log_for_tests,
)

try:
    from pipeline_core.configuration import (
        print_config,
        diag_broll_provider_limits,
        resolved_providers,
        to_bool,
    )
except Exception:
    pass

def apply_llm_overrides(settings, *_, **__):
    return settings

_GLOBAL_SETTINGS = None

def set_settings(s):
    global _GLOBAL_SETTINGS
    _GLOBAL_SETTINGS = s

def get_settings(overrides=None):
    if _GLOBAL_SETTINGS is not None and (overrides is None or overrides == {}):
        return _GLOBAL_SETTINGS
    return load_settings(overrides)

def _resolve_font_path(font_name: str | None) -> str | None:
    for env_name in ("PIPELINE_SUBTITLE_FONT_PATH", "PIPELINE_SUB_FONT_PATH"):
        p = os.getenv(env_name)
        if not p:
            continue
        try:
            pp = Path(p)
            if not pp.is_absolute():
                root = Path(__file__).resolve().parents[2]
                pp = (root / pp).resolve()
            if pp.exists():
                return str(pp)
        except Exception:
            pass
    try:
        root = Path(__file__).resolve().parents[2]
        for rel in ("assets/fonts/Montserrat-ExtraBold.ttf",
                    "assets/fonts/Montserrat-Bold.ttf"):
            cand = (root / rel)
            if cand.exists():
                return str(cand)
    except Exception:
        pass
    try:
        win = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        for name in ("Montserrat-ExtraBold.ttf","Montserrat-Bold.ttf","impact.ttf"):
            cand = win / name
            if cand.exists():
                return str(cand)
    except Exception:
        pass
    return None

def load_settings(overrides=None):
    s = _raw_load_settings(overrides)
    try:
        sub = getattr(s, "subtitles", None)
        if sub is not None and (getattr(sub, "font_path", None) in (None, "", False)):
            path = _resolve_font_path(getattr(sub, "font", None))
            if path:
                sub.font_path = path
    except Exception:
        pass
    return s

__all__ = [
    "Settings", "load_settings", "log_effective_settings", "reset_startup_log_for_tests",
    "print_config", "diag_broll_provider_limits", "resolved_providers", "to_bool",
    "apply_llm_overrides", "get_settings", "set_settings",
]
