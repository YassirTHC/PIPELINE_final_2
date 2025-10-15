# Local PyCaps Renderer

The pipeline now embeds a lightweight kinetic caption renderer inspired by the
open-source `pycaps` project.  It mirrors the Hormozi-style kinetic captions
without depending on the upstream PyPI package and keeps the public entry
points (`render_with_pycaps`, `to_pycaps_input`, `ensure_template_assets`) used
across the codebase.

## Rendering capabilities

* Word-level timing with automatic fallback when Whisper tokens are missing.
* Animated highlights that scale the active word (`highlight_scale`) while
  blending between the primary and secondary colours.
* Drop shadow, stroke and optional rounded background bar for readability.
* Automatic wrapping with configurable line limits and safe bottom margin.
* Optional emoji support controlled via the `VP_SUBTITLES_EMOJIS` toggle.
* Hardware-friendly encoding defaults (`libx264`, `aac`, `-movflags
  +faststart`, configurable thread count).

## Theme presets

Three presets are bundled and can be selected through the `theme` field on the
subtitle settings or via `VP_SUBTITLES_THEME`:

| Name     | Primary / Secondary | Defaults                                  |
|----------|---------------------|--------------------------------------------|
| `clean`  | White / Steel blue  | Lighter shadow, tighter background opacity |
| `bold`   | White / Coral red   | Larger type with heavier stroke            |
| `hormozi`| White / Golden yellow | Matches the legacy Hormozi integration   |

Each preset can be customised with the overrides listed below.

## Configuration knobs

The typed settings (`SubtitleSettings`) expose the kinetic caption parameters
that can be set in configuration files or via environment variables:

| Field / Environment variable                            | Description                                     |
|---------------------------------------------------------|-------------------------------------------------|
| `theme` / `VP_SUBTITLES_THEME`                          | Preset name (`clean`, `bold`, `hormozi`).       |
| `font`, `font_path`, `font_size`                        | Typography.                                     |
| `primary_color`, `secondary_color`, `stroke_color`      | Colour palette (hex values).                    |
| `shadow_color`, `shadow_opacity`, `shadow_offset`       | Drop shadow styling.                            |
| `background_color`, `background_opacity` (`VP_SUBTITLES_BG_*`) | Backplate colour and alpha.             |
| `margin_bottom_pct`                                     | Safe area margin (percentage of video height).  |
| `max_lines`, `max_chars_per_line`                       | Wrapping limits (default 3 lines, 24 chars).    |
| `uppercase_keywords`, `uppercase_min_length`            | Heuristics for automatic emphasis.              |
| `highlight_scale`                                       | Scale factor applied to the active word.        |
| `enable_emojis` / `VP_SUBTITLES_EMOJIS`                 | Allow emoji injection in captions.              |
| `subtitle_safe_margin_px`                               | Legacy compatibility for template generation.   |

When the engine is set to `pycaps` (`VP_SUBTITLES_ENGINE=pycaps`) emojis are
disabled by default, unless explicitly re-enabled via `VP_SUBTITLES_EMOJIS=1`.

## Emoji toggle

Emojis can be kept or stripped without touching the renderer code:

```bash
export VP_SUBTITLES_ENGINE=pycaps
export VP_SUBTITLES_EMOJIS=0  # disable entirely
# export VP_SUBTITLES_EMOJIS=1  # enable if desired
```

## Limitations

* The renderer relies on MoviePy; ensure version 2+ is available and importable
  from `moviepy` (not `moviepy.editor`).
* Font discovery still follows the previous behaviour (bundled Montserrat,
  Windows font directory, then system fallbacks).
* Highlight animation uses a static scale for simplicity; easing curves can be
  added later if required.
* Rendering very long segments may incur additional processing time because the
  renderer generates per-word overlays.

## Performance tips

* Keep `max_chars_per_line` under ~28 to avoid excessively wide overlays.
* Use the `threads` option (environment variable `VP_SUBTITLES_THREADS`) to
  tune encoding throughput on multi-core systems.
* Reuse the same template directory across runs so that the bundled fonts are
  copied only once by `ensure_template_assets`.
* Run the focused unit tests (`pytest tests/test_subtitles_engine.py`) after any
  change to the renderer to catch regressions quickly.
