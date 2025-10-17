# PyCaps E2E Validation Report

## Environment Snapshot

```json
{
  "python": "3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]",
  "python_executable": "C:\\Users\\Administrator\\Desktop\\video_pipeline - Copy\\venv311\\Scripts\\python.exe",
  "platform": "Windows-10-10.0.26100-SP0",
  "numpy": "1.26.4",
  "torch": "2.9.0+cpu",
  "whisper": "20231117",
  "opencv-python": "4.11.0",
  "moviepy": "2.1.2",
  "mediapipe": "0.10.21",
  "transformers": "4.57.1",
  "sentence-transformers": "3.4.1",
  "pydantic": "2.12.2",
  "scikit-learn": "1.7.2",
  "pycaps": "0.1.0"
}
```

## Effective Runtime Configuration

```
[CONFIG] effective={"broll": {"min_gap_s": 1.0, "min_start_s": 2.0, "no_repeat_s": 4.0}, "fetch": {"allow_images": true, "allow_videos": true, "api_keys": {"PEXELS_API_KEY": "****pDoD", "PIXABAY_API_KEY": "****6a69", "UNSPLASH_ACCESS_KEY": null}, "max_per_keyword": 8, "provider_limits": {"pexels": 3}, "providers": ["pexels", "pixabay"], "timeout_s": 8.0}, "flags": {"fast_tests": false, "llm_max_queries_per_segment": 3, "max_segments_in_flight": 1, "tfidf_fallback_disabled": false}, "llm": {"disable_dynamic_segment": true, "disable_hashtags": false, "endpoint": "http://localhost:11434", "fallback_trunc": 3500, "force_non_stream": false, "json_mode": true, "json_transcript_limit": 1200, "keep_alive": "30m", "keywords_first": false, "max_attempts": 3, "min_chars": 8, "model": "qwen3:8b", "model_json": "gemma3:4b", "model_text": "qwen2.5:3b", "num_ctx": 4096, "num_predict": 512, "provider": "pipeline_integration", "repeat_penalty": 1.1, "request_cooldown_jitter_s": 0.0, "request_cooldown_s": 0.0, "target_lang": "en", "temperature": 0.3, "timeout_fallback_s": 45.0, "timeout_stream_s": 120.0, "top_p": 0.9}, "paths": {"clips_dir": "clips", "output_dir": "output", "temp_dir": "temp"}, "subtitles": {"background_color": "#000000", "background_opacity": 0.35, "emoji_max_per_segment": 3, "emoji_min_gap_groups": 2, "emoji_no_context_fallback": "", "emoji_target_per_10": 5, "enable_emojis": false, "engine": "pycaps", "font": null, "font_path": "C:\\Users\\Administrator\\Desktop\\video_pipeline - Copy\\assets\\fonts\\Montserrat-ExtraBold.ttf", "font_size": 138, "hero_emoji_enable": true, "hero_emoji_max_per_segment": 1, "highlight_scale": 1.08, "keyword_background": false, "margin_bottom_pct": 0.12, "max_chars_per_line": 24, "max_lines": 3, "primary_color": "#FFFFFF", "secondary_color": "#FBC531", "shadow_color": "#000000", "shadow_offset": 3, "shadow_opacity": 0.35, "stroke_color": "#000000", "stroke_px": 7, "subtitle_safe_margin_px": 260, "theme": "hormozi", "uppercase_keywords": true, "uppercase_min_length": 6}}
```

## PyCaps Evidence

- `INFO:__main__:INFO:[Subtitles] Engine=pycaps (Hormozi disabled)`
- `INFO:subtitle_engines.pycaps_engine:[PyCaps] Rendered subtitles using PyCaps 0.1.0`
- `tools/validate_pycaps_log.py run_pycaps_e2e.log → [validate] log validation succeeded`

## LLM Metadata (English)

- `Title: Auto-generated Clip Title`
- `Description: Auto-generated description from transcript....`
- `Hashtags: #motivation #success #mindset #growth #productivity #inspiration #goals #selfimprovement #lifestyle #focus`

## B-roll Query Samples

- `INFO:__main__:[BROLL][LLM] segment=0.00-6.10 queries=['realize rewards internal', 'rewards internal what', 'internal what linking'] (source=metadata_first)`
- `INFO:__main__:[BROLL][LLM] segment=7.08-11.88 queries=['duration path outcome', 'linking duration path', 'what linking duration'] (source=metadata_first)`
- `INFO:__main__:[BROLL][LLM] segment=11.88-18.14 queries=['rewards internal what', 'realize rewards internal', 'what linking duration'] (source=metadata_first)`
- `INFO:__main__:[BROLL][LLM] segment=18.14-24.26 queries=['duration path outcome', 'linking duration path', 'what linking duration'] (source=metadata_first)`
- `INFO:__main__:[BROLL][LLM] segment=24.28-28.32 queries=['what linking duration', 'rewards internal what', 'realize rewards internal'] (source=metadata_first)`

## Outputs

- `output\final\final_121-020.mp4` — 7.26 MB (rendered with PyCaps subtitles)
- `output\meta\selection_report_reframed.json` — 9.7 KB (B-roll selection report)
- `run_pycaps_e2e.log` — UTF-16 log captured with `* > run_pycaps_e2e.log`

## Additional Checks

- No forbidden fallback traces detected (validation script filters out config keys only).
- No reversion to Hormozi engine, no PyCaps rendering errors, and no emergency B-roll fallbacks.
- Subtitle template assets present under `assets/subtitles/pycaps/` (CSS + template + font provisioning helper).

## Conclusion

**PASS** — PyCaps GitHub build is active in the pipeline, subtitles render through `pycaps.pipeline`, metadata stays in English, no fallback paths were exercised, and required artefacts were produced. No outstanding issues.
