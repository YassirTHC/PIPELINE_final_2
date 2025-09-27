# B-roll Pipeline Audit (2024-XX-XX)

## Scope
This note captures the state of the B-roll insertion pipeline in `video_processor.py`
and the surrounding `pipeline_core` helpers as of this analysis. The goal is to
highlight legacy vestiges that can compromise determinism or observability and
to provide actionable next steps.

## Current flow (happy path)
1. `VideoProcessor.insert_brolls_if_enabled` builds dynamic context with
   `LLMMetadataGeneratorService.generate_dynamic_context` to derive global
   keywords and per-segment briefs.
2. `_maybe_use_pipeline_core` invokes
   `VideoProcessor._insert_brolls_pipeline_core`, which orchestrates provider
   fetches through `pipeline_core.fetchers.FetcherOrchestrator` and logs
   per-segment metrics (`broll_segment_queries`, `broll_segment_decision`) plus a
   `broll_summary` event.
3. When `_insert_brolls_pipeline_core` returns a count, the caller prints the
   completion banner but **immediately returns the original `input_path`**.
   As of today the core loop never produces a rendered timeline – the
   selected assets are not downloaded or composited.
4. Because of step 3, the execution continues into the legacy
   `AI-B-roll/src.pipeline` branch which performs its own keyword extraction,
   fetch, FAISS indexing, and rendering. This legacy path still imports
   large modules (`SyncContextAnalyzer`, `BrollSelector`, etc.) and contains
   numerous heuristics that can reintroduce placeholders or French tokens.

## Legacy vestiges & risks
- **Duplicate fetch logic** – The core orchestrator decides on assets but the
  legacy path re-fetches candidates using a different scoring stack. This makes
  the JSONL telemetry diverge from the rendered output, and legacy fetchers can
  still reach non-Pexels providers when configuration toggles change.
- **Placeholder queries** – `generate_broll_prompts_ai` and related helpers in
  the legacy branch still emit terms such as "doctor talking" or
  "person discussing". Although `_dedupe_queries` filters some stopwords, the
  legacy path bypasses this filter through manual prompt construction.
- **Language drift** – The normalisation introduced in
  `_insert_brolls_pipeline_core`/`enforce_fetch_language` is not propagated to
  the legacy fetch branch. When the fallback executes, French or accent-stripped
  tokens (e.g. `rcompense`) leak back into provider queries.
- **Mediapipe dependencies** – The import guards avoid crashes when Mediapipe is
  absent, but several branches still assume `mp` is available (e.g. legacy
  analysis) and may throw if toggled back on.
- **Telemetry mismatch** – Console banners and JSONL events reflect only the
  core selection counts. When the legacy pipeline overrides the decisions, the
  logs describe assets that never touch the final render.

## Recommended actions
1. **Render within `pipeline_core`** – Materialise `selected_assets` by
   downloading the chosen clips, build a timeline respecting gap/duration rules
   (already enforced via `forced_keep_segments`), and return the path of the
   rendered video. Once the render succeeds, short-circuit the legacy branch by
   returning early with the new path.
2. **Feature-flag or delete the legacy stack** – At minimum introduce an
   environment flag (default OFF) so `insert_brolls_if_enabled` skips the old
   `AI-B-roll/src.pipeline` imports when `pipeline_core` completes successfully.
3. **Shared normalisation layer** – Reuse `enforce_fetch_language` and the
   anti-placeholder logic for every query builder, including
   `generate_broll_prompts_ai` and the intelligent selector prompts.
4. **Tighten provider guardrails** – Ensure every legacy fetch call reuses the
   orchestrator or a thin wrapper that enforces the Pexels/Pixabay-only rule and
   logs latency/timeout metrics consistently.
5. **Regression tests** – Add an integration-style test that patches
   `FetcherOrchestrator` to return deterministic candidates and asserts that the
   rendered timeline differs from the input. Expand
   `tests/test_segment_queries.py` to cover the legacy prompt helpers until they
   are deleted.

## Test confirmation
`pytest -q` (Python 3.11.8) – 30 tests pass, 0 failures, 2 deprecation warnings
from protobuf's upb containers.

