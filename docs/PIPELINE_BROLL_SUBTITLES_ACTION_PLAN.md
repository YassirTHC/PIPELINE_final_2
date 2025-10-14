# Pipeline B-Roll & Subtitles — Root Cause Analysis and Implementation Plan

## 1. Root Cause Analysis

### 1.1 PyCaps integration fails despite installed package
- **Observed behaviour**: Runtime reports `PyCaps is not installed` even when `pycaps 0.2.0` is present and the engine is configured to use PyCaps.
- **Code path**: `subtitle_engines/pycaps_engine.py` performs `from pycaps.pipeline import JsonConfigLoader`. The published 0.2.0 wheel exposes `pycaps.JsonConfigLoader` instead. The `ImportError` is caught and re-raised with a misleading message, so the pipeline exits before attempting alternative symbols.
- **Dependencies**: Relies on PyCaps packaging layout, `SubtitleEngineFactory` to select the PyCaps engine, and the orchestrator’s error handling which surfaces the generic "not installed" message.

### 1.2 Metadata generator emits `None` title/description
- **Observed behaviour**: Runs log `Titre: None` / `Description: None` while hashtags are produced.
- **Code path**: Metadata prompt defined in `pipeline_core/llm_service.py` (metadata task) uses a single JSON schema request containing optional fields. The parser in `pipeline_core/metadata_writer.py` accepts null values and persists them to `meta.txt` without validation.
- **Dependencies**: LLM JSON parsing utilities, metadata persistence layer, downstream `meta.txt` consumer expecting non-empty strings.

### 1.3 B-roll query generation too conceptual and repetitive
- **Observed behaviour**: Queries such as `internal motivation visuals` repeated across segments, leading to low recall.
- **Code path**: Query prompt in `pipeline_core/llm_service.py` emphasises abstract themes. `_combine_broll_queries` in `broll_selector.py` merges per-segment hints but truncates to the first few entries, amplifying repetition. Sanitization only trims whitespace; there is no lexical filter.
- **Dependencies**: LLM prompt design, query sanitizer (currently minimal), fetcher modules relying on concrete search terms.

### 1.4 Frequent LLM fallbacks and forced-keep behaviour
- **Observed behaviour**: Logs show fallbacks to TF-IDF heuristics and forced-keep clips with low confidence scores.
- **Code path**: `pipeline_core/llm_service.generate_segment_hints` retries limited times before delegating to `fallback_heuristic.py`. When LLM output is empty, `broll_selector.py` calls `_fallback_keep_first_candidate` which forces low-score assets to maintain continuity.
- **Dependencies**: LLM invocation layer, fallback heuristic module, scoring functions in `enhanced_scoring.py`.

### 1.5 Temporal rules reject large share of candidates
- **Observed behaviour**: `min_start`, `min_gap`, and `no_repeat` thresholds drop 40–55% of fetched candidates.
- **Code path**: `broll_selector.apply_timing_rules` enforces global thresholds regardless of candidate pool size. There is no adaptive logic based on recall rate.
- **Dependencies**: Timing configuration from `config/broll_rules.json`, heuristics inside `broll_selector`, and logging in `broll_verification_system.py`.

### 1.6 Fetch configuration ignores environment overrides
- **Observed behaviour**: Effective config logs still show defaults (`timeout_s=8`, `max_per_keyword=8`, `provider_limits.pexels=3`) even when environment variables request higher values.
- **Code path**: `config/__init__.py` loads static JSON first and only applies environment overrides at process start. The batch runner spawns sub-processes without propagating the env overrides, and the override parser does not cover provider-specific keys (e.g. `PEXELS`).
- **Dependencies**: Configuration loader, environment parsing utility (`utils/env.py`), fetcher initialisation.

### 1.7 Lack of structured query sanitization and emergency fallbacks
- **Observed behaviour**: Segments end up with <3 usable queries when LLM output is malformed; no deterministic fallback covers all positions.
- **Code path**: `broll_selector.clean_queries` only deduplicates and trims strings. Emergency profile logic is absent; fallbacks rely on limited heuristics in `fallback_heuristic.py`.
- **Dependencies**: Query sanitizer, fallback mapping module to be introduced, integration with LLM output validation.

### 1.8 Metadata prompt lacks guardrails
- **Observed behaviour**: The pipeline accepts missing title/description without remediation.
- **Code path**: Combined metadata prompt in `llm_service` does not enforce mandatory fields; validation layer in `metadata_writer` lacks fallback assignment.
- **Dependencies**: JSON schema definition, validation utilities, meta writer module.

### 1.9 Limited observability metrics
- **Observed behaviour**: Operators cannot quickly gauge fallback rates, rejection ratios, or sanitized query counts.
- **Code path**: Logging is scattered; no aggregated counters in `pipeline_core` or `broll_selector`.
- **Dependencies**: Logging framework, metrics aggregation utilities.

## 2. Implementation Plan

For each task, we outline scope, acceptance criteria, and tests.

### A. Stabilise PyCaps engine
- **Tasks**
  1. Introduce resilient loader: attempt `pycaps.pipeline.JsonConfigLoader`, fallback to `pycaps.JsonConfigLoader`, and inspect module attributes dynamically.
  2. Update error messages to reflect actual import failure cause.
  3. Log interpreter path, PyCaps version, and module file.
- **Acceptance criteria**
  - Engine loads successfully when either symbol is available.
  - When PyCaps lacks the loader, error message names the missing attribute and suggests the GitHub source install.
  - Logs include executable path and PyCaps metadata.
- **Tests**
  - Unit tests monkeypatching `sys.modules` to simulate various module layouts.
  - Integration test running the subtitle engine with a stub PyCaps implementation confirming fallback logic.

### B. Rewrite LLM prompt for B-roll queries
- **Tasks**
  1. Create new prompt template emphasising concrete, filmable actions (2–4 words) and banning abstract vocabulary.
  2. Apply recommended inference parameters (`temperature=0.2`, `top_p=0.85`, `repeat_penalty=1.2`, `num_predict=256`).
- **Acceptance criteria**
  - Prompt unit tests validate presence of guidance text and banned term list.
  - Mocked LLM responses produce filmable queries; sanitizer removes banned phrases.
- **Tests**
  - Snapshot test of prompt template.
  - Unit test verifying parameter injection in the LLM client.

### C. Implement query sanitizer module
- **Tasks**
  1. Add `pipeline_core/query_sanitizer.py` with blacklist enforcement, length bounds, and fallback filler queries.
  2. Integrate sanitizer before fetch stage.
- **Acceptance criteria**
  - Queries containing banned terms or outside length bounds are removed.
  - When <3 queries remain, fallback entries are inserted.
- **Tests**
  - Unit tests covering banned term removal, trimming, deduplication, and fallback injection.

### D. Batch segment-specific LLM generation
- **Tasks**
  1. Split query generation into three stages: global summary, global queries, batched segment queries (size=3).
  2. Enforce strict JSON parsing and partial failure recovery per batch.
- **Acceptance criteria**
  - Pipeline proceeds when some batches fail, using outputs from successful batches.
  - Segment hints cover ≥90% segments during nominal mock runs.
- **Tests**
  - Unit tests on JSON validators handling partial success.
  - Integration test with mocked LLM raising errors on specific batches to ensure graceful degradation.

### E. Introduce positional emergency fallback
- **Tasks**
  1. Define static mapping of segment index → list of four fallback queries.
  2. Trigger fallback when sanitized query count <3.
- **Acceptance criteria**
  - Every segment yields ≥3 usable queries after fallback.
  - Logging indicates when positional fallback activated.
- **Tests**
  - Unit tests verifying mapping coverage for 12 segments.
  - Integration test simulating empty LLM output to assert fallback usage.

### F. Adaptive temporal rule profiles
- **Tasks**
  1. Add profile configuration (default vs low-recall) to timing engine.
  2. Implement heuristic: switch to low-recall when candidate count < threshold.
- **Acceptance criteria**
  - In low-recall scenarios, more candidates survive timing filters without violating constraints.
  - Logs show profile selection per segment.
- **Tests**
  - Unit tests for heuristic selection logic.
  - Integration test with synthetic candidate lists verifying rule adjustments.

### G. Harden metadata generation
- **Tasks**
  1. Introduce dedicated prompt for title/description with explicit requirements and fallback text.
  2. Enforce non-null validation in `metadata_writer` and log fallback usage.
- **Acceptance criteria**
  - `meta.txt` always contains non-empty title and description.
  - Missing fields trigger deterministic fallback content and a warning log.
- **Tests**
  - Unit tests covering JSON parsing, fallback activation, and file output.
  - Integration test verifying pipeline behaviour when LLM returns `null` values.

### H. Honour fetch configuration overrides
- **Tasks**
  1. Extend env override parser to handle numeric limits and provider-specific keys.
  2. Ensure overrides propagate through spawned processes; log effective configuration.
- **Acceptance criteria**
  - Environment variables supersede config defaults in logs and runtime behaviour.
  - Fetchers respect updated limits during mock fetch runs.
- **Tests**
  - Unit tests patching environment to assert applied overrides.
  - Integration test with fake fetcher verifying limit enforcement.

### I. Expand observability metrics
- **Tasks**
  1. Add counters for fallback usage, sanitized query counts, timing rejection rates, and average B-roll scores.
  2. Surface metrics in structured log output for each run.
- **Acceptance criteria**
  - Logs include aggregated statistics at run completion.
  - Metrics reset correctly between runs.
- **Tests**
  - Unit tests for metric accumulator logic.

### J. Update documentation
- **Tasks**
  1. Document PyCaps setup, environment overrides, adaptive timing profiles, and fallback strategy in README/CONTRIBUTING.
  2. Provide troubleshooting steps for common issues (PyCaps missing symbol, empty metadata, low recall).
- **Acceptance criteria**
  - Documentation reflects new configuration knobs and testing commands.
  - Internal team can follow guide to enable PyCaps and interpret logs.
- **Tests**
  - Documentation review (no automated tests required).

## 3. Definition of Done
- ≥90% of segments have at least one B-roll candidate post-fetch.
- ≥9 of 12 segments receive inserted B-rolls on medium-difficulty clips.
- <10% of segments rely on TF-IDF/emergency fallbacks in standard runs.
- `meta.txt` never contains `None` for title or description.
- Logs confirm environment overrides applied to fetch configuration.
- PyCaps engine either renders successfully or emits precise import errors.
- Run-level logs emit metrics for fallback rate, sanitized queries, rejection ratios, and average scores.

