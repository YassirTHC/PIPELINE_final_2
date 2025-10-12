# Resolving B-roll merge conflicts

When merging `main` into the feature branch that introduced the LLM-only
B-roll query logic, keep the changes labelled `codex/investigate-missing-video-title-and-description-0aq4sw`.

* In `tests/test_segment_queries.py`, choose **Accept Current Change** so the
  new regression tests that exercise `_merge_segment_query_sources` remain in
  place. These tests verify the LLM-only flag, selector filtering, and the
  shared-token threshold. 【F:tests/test_segment_queries.py†L229-L309】
* In `video_processor.py`, keep the branch version that normalises LLM
  prefill terms, applies the `_filter_terms_for_segment` helper to selector
  keywords, and honours the LLM-only fast path. These behaviours live in the
  block starting at `_merge_segment_query_sources`. 【F:video_processor.py†L1195-L1287】

The `main` branch does not carry competing implementations in those regions,
so accepting the feature branch copy preserves the intended filtering and
fallback logic without reintroducing generic queries like "desk journaling at".
