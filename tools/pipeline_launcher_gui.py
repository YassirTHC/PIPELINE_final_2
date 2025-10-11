#!/usr/bin/env python3
"""Simple GUI launcher for run_pipeline.py."""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

try:
    from video_pipeline.config import load_settings
    _SETTINGS = load_settings()
except Exception:
    _SETTINGS = None

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"

DEFAULT_CONFIG = {
    "PIPELINE_LLM_PROVIDER": "ollama",
    "PIPELINE_LLM_ENDPOINT": "http://localhost:11434",
    "PIPELINE_LLM_MODEL_TEXT": "qwen3:8b",
    "PIPELINE_LLM_MODEL_JSON": "qwen3:8b",
    "PIPELINE_LLM_FORCE_NON_STREAM": "true",
    "PIPELINE_LLM_JSON_MODE": "true",
    "PIPELINE_LLM_TIMEOUT_S": "60",
    "PIPELINE_LLM_FALLBACK_TIMEOUT_S": "30",
    "BROLL_FETCH_PROVIDER": "pixabay,pexels",
    "FETCH_MAX": "8",
    "PIXABAY_API_KEY": "51724939-ee09a81ccfce0f5623df46a69",
    "PEXELS_API_KEY": "pwhBa9K7fa9IQJCmfCy0NfHFWy8QyqoCkGnWLK3NC2SbDTtUeuhxpDoD",
    "BROLL_FETCH_ALLOW_IMAGES": "1",
    "BROLL_FETCH_ALLOW_VIDEOS": "1",
}

if _SETTINGS is not None:
    try:
        DEFAULT_CONFIG.update(
            {
                "PIPELINE_LLM_PROVIDER": getattr(_SETTINGS.llm, "provider", "ollama"),
                "PIPELINE_LLM_ENDPOINT": getattr(_SETTINGS.llm, "endpoint", "http://localhost:11434"),
                "PIPELINE_LLM_MODEL_TEXT": _SETTINGS.llm.model_text or _SETTINGS.llm.model,
                "PIPELINE_LLM_MODEL_JSON": _SETTINGS.llm.model_json or _SETTINGS.llm.model,
                "PIPELINE_LLM_FORCE_NON_STREAM": str(_SETTINGS.llm.force_non_stream).lower(),
                "PIPELINE_LLM_JSON_MODE": str(_SETTINGS.llm.json_mode).lower(),
                "PIPELINE_LLM_TIMEOUT_S": str(int(_SETTINGS.llm.timeout_stream_s)),
                "PIPELINE_LLM_FALLBACK_TIMEOUT_S": str(int(_SETTINGS.llm.timeout_fallback_s)),
                "BROLL_FETCH_PROVIDER": ",".join(_SETTINGS.fetch.providers) or "pixabay,pexels",
                "FETCH_MAX": str(int(_SETTINGS.fetch.max_per_keyword)),
                "PIXABAY_API_KEY": (_SETTINGS.fetch.api_keys or {}).get("PIXABAY_API_KEY", DEFAULT_CONFIG["PIXABAY_API_KEY"]),
                "PEXELS_API_KEY": (_SETTINGS.fetch.api_keys or {}).get("PEXELS_API_KEY", DEFAULT_CONFIG["PEXELS_API_KEY"]),
                "BROLL_FETCH_ALLOW_IMAGES": str(int(getattr(_SETTINGS.fetch, "allow_images", True))).lower(),
                "BROLL_FETCH_ALLOW_VIDEOS": str(int(getattr(_SETTINGS.fetch, "allow_videos", True))).lower(),
            }
        )
    except Exception:
        pass

class PipelineLauncher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Video Pipeline Launcher")
        self.geometry("900x700")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.process: Optional[subprocess.Popen[str]] = None
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self._create_widgets()
        self._poll_log_queue()

    def _create_widgets(self) -> None:
        options_frame = ttk.LabelFrame(self, text="Pipeline Options")
        options_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        for i in range(4):
            options_frame.columnconfigure(i, weight=1)

        ttk.Label(options_frame, text="Video file").grid(row=0, column=0, sticky="w")
        self.video_var = tk.StringVar()
        video_entry = ttk.Entry(options_frame, textvariable=self.video_var)
        video_entry.grid(row=1, column=0, columnspan=3, sticky="ew", padx=(0, 5))
        ttk.Button(options_frame, text="Browse…", command=self._choose_video).grid(row=1, column=3, sticky="ew")

        ttk.Label(options_frame, text="Output directory").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.output_var = tk.StringVar(value=str(DEFAULT_OUTPUT_DIR))
        out_entry = ttk.Entry(options_frame, textvariable=self.output_var)
        out_entry.grid(row=3, column=0, columnspan=3, sticky="ew", padx=(0, 5))
        ttk.Button(options_frame, text="Browse…", command=self._choose_output).grid(row=3, column=3, sticky="ew")

        llm_frame = ttk.LabelFrame(options_frame, text="LLM settings")
        llm_frame.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(15, 0))
        for i in range(4):
            llm_frame.columnconfigure(i, weight=1)

        self.env_vars: Dict[str, tk.StringVar] = {}
        fields = [
            ("Provider", "PIPELINE_LLM_PROVIDER"),
            ("Endpoint", "PIPELINE_LLM_ENDPOINT"),
            ("Model (text)", "PIPELINE_LLM_MODEL_TEXT"),
            ("Model (JSON)", "PIPELINE_LLM_MODEL_JSON"),
            ("Timeout stream (s)", "PIPELINE_LLM_TIMEOUT_S"),
            ("Timeout fallback (s)", "PIPELINE_LLM_FALLBACK_TIMEOUT_S"),
            ("Max tokens", "PIPELINE_LLM_NUM_PREDICT"),
        ]
        for idx, (label_text, env_key) in enumerate(fields):
            ttk.Label(llm_frame, text=label_text).grid(row=idx, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=DEFAULT_CONFIG.get(env_key, ""))
            entry = ttk.Entry(llm_frame, textvariable=var)
            entry.grid(row=idx, column=1, columnspan=3, sticky="ew", pady=2)
            self.env_vars[env_key] = var

        self.force_non_stream_var = tk.BooleanVar(value=DEFAULT_CONFIG.get("PIPELINE_LLM_FORCE_NON_STREAM", "true").lower() == "true")
        self.json_mode_var = tk.BooleanVar(value=DEFAULT_CONFIG.get("PIPELINE_LLM_JSON_MODE", "true").lower() == "true")
        ttk.Checkbutton(llm_frame, text="Force non stream", variable=self.force_non_stream_var).grid(row=len(fields), column=0, sticky="w", pady=2)
        ttk.Checkbutton(llm_frame, text="JSON mode", variable=self.json_mode_var).grid(row=len(fields), column=1, sticky="w", pady=2)

        fetch_frame = ttk.LabelFrame(options_frame, text="Fetch settings")
        fetch_frame.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(15, 0))
        for i in range(4):
            fetch_frame.columnconfigure(i, weight=1)

        self.env_vars["BROLL_FETCH_PROVIDER"] = tk.StringVar(value=DEFAULT_CONFIG.get("BROLL_FETCH_PROVIDER", ""))
        self.env_vars["FETCH_MAX"] = tk.StringVar(value=DEFAULT_CONFIG.get("FETCH_MAX", ""))
        self.env_vars["PIXABAY_API_KEY"] = tk.StringVar(value=DEFAULT_CONFIG.get("PIXABAY_API_KEY", ""))
        self.env_vars["PEXELS_API_KEY"] = tk.StringVar(value=DEFAULT_CONFIG.get("PEXELS_API_KEY", ""))
        self.env_vars["BROLL_FETCH_ALLOW_IMAGES"] = tk.StringVar(value=DEFAULT_CONFIG.get("BROLL_FETCH_ALLOW_IMAGES", "1"))
        self.env_vars["BROLL_FETCH_ALLOW_VIDEOS"] = tk.StringVar(value=DEFAULT_CONFIG.get("BROLL_FETCH_ALLOW_VIDEOS", "1"))

        ttk.Label(fetch_frame, text="Providers (comma list)").grid(row=0, column=0, sticky="w")
        ttk.Entry(fetch_frame, textvariable=self.env_vars["BROLL_FETCH_PROVIDER"]).grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 5))
        ttk.Label(fetch_frame, text="Max per keyword").grid(row=0, column=2, sticky="w")
        ttk.Entry(fetch_frame, textvariable=self.env_vars["FETCH_MAX"]).grid(row=1, column=2, sticky="ew")

        ttk.Label(fetch_frame, text="Pixabay API key").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(fetch_frame, textvariable=self.env_vars["PIXABAY_API_KEY"]).grid(row=3, column=0, columnspan=2, sticky="ew", padx=(0, 5))
        ttk.Label(fetch_frame, text="Pexels API key").grid(row=2, column=2, sticky="w", pady=(10, 0))
        ttk.Entry(fetch_frame, textvariable=self.env_vars["PEXELS_API_KEY"]).grid(row=3, column=2, sticky="ew")

        ttk.Label(fetch_frame, text="Allow images").grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(fetch_frame, textvariable=self.env_vars["BROLL_FETCH_ALLOW_IMAGES"]).grid(row=5, column=0, sticky="ew", padx=(0, 5))
        ttk.Label(fetch_frame, text="Allow videos").grid(row=4, column=1, sticky="w", pady=(10, 0))
        ttk.Entry(fetch_frame, textvariable=self.env_vars["BROLL_FETCH_ALLOW_VIDEOS"]).grid(row=5, column=1, sticky="ew")

        self.verbose_var = tk.BooleanVar(value=False)
        self.no_emoji_var = tk.BooleanVar(value=False)
        self.diag_broll_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(options_frame, text="Verbose", variable=self.verbose_var).grid(row=6, column=0, sticky="w", pady=(15, 0))
        ttk.Checkbutton(options_frame, text="No emoji", variable=self.no_emoji_var).grid(row=6, column=1, sticky="w", pady=(15, 0))
        ttk.Checkbutton(options_frame, text="Diagnostic B-roll", variable=self.diag_broll_var).grid(row=6, column=2, sticky="w", pady=(15, 0))

        buttons_frame = ttk.Frame(options_frame)
        buttons_frame.grid(row=7, column=0, columnspan=4, sticky="ew", pady=(20, 0))
        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)
        ttk.Button(buttons_frame, text="Run pipeline", command=self._run_pipeline).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(buttons_frame, text="Stop", command=self._stop_pipeline).grid(row=0, column=1, sticky="ew")

        log_frame = ttk.LabelFrame(self, text="Logs")
        log_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, wrap="word", state="disabled", background="#1e1e1e", foreground="#f0f0f0")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text['yscrollcommand'] = scrollbar.set

    def _append_log(self, line: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _choose_video(self) -> None:
        path = filedialog.askopenfilename(title="Select video", filetypes=[("Video files", "*.mp4;*.mov;*.m4v;*.mkv"), ("All files", "*.*")])
        if path:
            self.video_var.set(path)

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_var.set(path)

    def _run_pipeline(self) -> None:
        if self.process is not None:
            messagebox.showwarning("Pipeline running", "A pipeline process is already running.")
            return

        video_path = self.video_var.get().strip()
        if not video_path:
            messagebox.showerror("Missing video", "Please choose a video file.")
            return
        video_file = Path(video_path)
        if not video_file.exists():
            messagebox.showerror("Invalid video", "The selected video file does not exist.")
            return

        output_dir = Path(self.output_var.get().strip())
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [sys.executable, str(REPO_ROOT / "run_pipeline.py"), "--video", str(video_file)]
        if self.verbose_var.get():
            cmd.append("--verbose")
        if self.no_emoji_var.get():
            cmd.append("--no-emoji")
        if self.diag_broll_var.get():
            cmd.append("--diag-broll")

        env = os.environ.copy()
        for key, var in self.env_vars.items():
            value = var.get().strip()
            if value:
                env[key] = value
        env["PIPELINE_LLM_FORCE_NON_STREAM"] = "true" if self.force_non_stream_var.get() else "false"
        env["PIPELINE_LLM_JSON_MODE"] = "true" if self.json_mode_var.get() else "false"
        env["PIPELINE_OUTPUT_DIR"] = str(output_dir)

        self._append_log("\n>>> Running: {}\n".format(' '.join(cmd)))

        def target() -> None:
            try:
                self.process = subprocess.Popen(
                    cmd,
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )
                assert self.process.stdout is not None
                for line in self.process.stdout:
                    self.log_queue.put(line)
                self.process.wait()
                self.log_queue.put("\n>>> Process exited with code {}\n".format(self.process.returncode))
            except Exception as exc:
                self.log_queue.put("\n[ERROR] {}\n".format(exc))
            finally:
                self.process = None

        threading.Thread(target=target, daemon=True).start()

    def _stop_pipeline(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        self.log_queue.put("\n>>> Termination signal sent.\n")

    def _poll_log_queue(self) -> None:
        try:
            while True:
                line = self.log_queue.get_nowait()
                self._append_log(line)
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)


def main() -> None:
    app = PipelineLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
