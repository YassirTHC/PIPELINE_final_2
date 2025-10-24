## Provider Mix Router – Quick Reference

This pipeline now routes Pexels, Coverr and Pixabay with a weighted strategy to
maximise visual diversity while honouring provider quotas.

### Default weight targets

| Provider | Default weight | Notes                         |
|----------|----------------|------------------------------|
| Pexels   | 0.45           | Rich catalogue, fast latency |
| Coverr   | 0.30           | Adds lifestyle & branded B‑roll, attribution metadata exposed |
| Pixabay  | 0.25           | Broad catalogue, used as diversity fallback |

Weights, caps and timeouts can be tuned at runtime:

```
FETCH_PROVIDER_WEIGHT__PEXELS=0.40
FETCH_PROVIDER_WEIGHT__COVERR=0.35
FETCH_PROVIDER_WEIGHT__PIXABAY=0.25

FETCH_PROVIDER_LIMITS__PEXELS=6
FETCH_PROVIDER_LIMITS__COVERR=4
FETCH_PROVIDER_LIMITS__PIXABAY=6

FETCH_PROVIDER_TIMEOUT_S__COVERR=12.0
FETCH_PROVIDER_DAILY_QUOTA__COVERR=120
FETCH_PROVIDER_SOFT_CAP__PEXELS=8       # soft per-run cap

BROLL_MAX_REUSE_PER_URL=2               # cross-provider reuse cap
FETCH_DRY_RUN=1                         # log selections, skip downloads
```

### Coverr integration

Provide your Coverr API key via `COVERR_API_KEY`. Each candidate now exposes:

```
provider, url, preview_url, width, height, duration_s,
tags, author, license, attribution_required/text/url
```

`attribution_required` is also surfaced in `selection_report_reframed.json` and
the run summary (`attribution_required_count`).

### Telemetry & reporting

`broll_summary` now includes:

- `router_provider_mix` / `provider_mix_ratio`
- `router_provider_errors`
- `router_seen_assets`, `router_max_url_reuse`
- `unique_query_count`
- `visual_expansions_total/kept`
- `attribution_required_count`

During fetching, each segment prints and logs the provider ordering:

```
?? PROVIDER_ORDER seg=3 order=['coverr', 'pexels', 'pixabay'] kept=5
```

Use `FETCH_DRY_RUN=1` for CI/dry-runs: providers are queried, mix is logged, no
media files are downloaded (placeholder files are created instead).
