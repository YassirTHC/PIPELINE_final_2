#!/usr/bin/env python3
"""Aggregate pipeline bench results and recommend the strongest Ollama configuration."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

FILL_WEIGHT = 0.40
DYNAMIC_WEIGHT = 0.25
DROPS_WEIGHT = 0.20
PROVIDER_WEIGHT = 0.10
NOISE_WEIGHT = 0.05
EXPECTED_SEGMENTS = 12
BASE_ENDPOINT = 'http://localhost:11434'


@dataclass
class ModelScore:
    model: str
    safe_name: str
    score: float
    components: Dict[str, float]
    metrics: Dict[str, Any]
    passes: Dict[str, bool]
    bench_path: str


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(value, max_value))


def compute_fill_score(selected: Optional[int], target: Optional[int], fill_rate: Optional[float]) -> Tuple[float, float]:
    if fill_rate is None and selected is not None and target:
        fill_rate = selected / target if target > 0 else 0.0
    fill_ratio = clamp(fill_rate or 0.0)
    return fill_ratio, fill_ratio * FILL_WEIGHT * 100


def compute_dynamic_score(empty_payload: int, segment_count: int) -> Tuple[float, float]:
    segment_score = clamp(segment_count / EXPECTED_SEGMENTS)
    if segment_count >= 8:
        segment_score = clamp(segment_score + 0.1)
    if empty_payload > 0:
        segment_score *= 0.1
    return segment_score, segment_score * DYNAMIC_WEIGHT * 100


def compute_drop_score(duration: float, orientation: float) -> Tuple[float, float]:
    duration_score = clamp(1 - clamp(duration / 60.0))
    orientation_score = clamp(1 - clamp(orientation / 120.0))
    drop_score = (duration_score + orientation_score) / 2
    drop_score = clamp(drop_score)
    return drop_score, drop_score * DROPS_WEIGHT * 100


def compute_provider_score(diag: Dict[str, Any]) -> Tuple[float, float]:
    if not diag or diag.get('error'):
        return 0.0, 0.0
    providers = diag.get('providers') or []
    total_providers = max(len(providers), 1)
    success_ratio = clamp((diag.get('success_count') or 0) / total_providers)
    candidate_total = diag.get('candidates_total') or 0
    candidate_score = clamp(candidate_total / EXPECTED_SEGMENTS)
    latency = diag.get('latency_avg_sec')
    latency_score = 1.0 if latency is None else clamp(1 - clamp(latency / 15.0))
    provider_score = clamp(0.5 * success_ratio + 0.3 * candidate_score + 0.2 * latency_score)
    return provider_score, provider_score * PROVIDER_WEIGHT * 100


def compute_noise_score(provider_noise: int, generic_noise: int) -> Tuple[float, float]:
    total_noise = (provider_noise or 0) + (generic_noise or 0)
    if total_noise == 0:
        base = 1.0
    elif total_noise <= 2:
        base = 0.4
    else:
        base = 0.2
    noise_score = clamp(base)
    return noise_score, noise_score * NOISE_WEIGHT * 100


def badge(passed: bool, warn: bool = False) -> str:
    if passed:
        return '✅'
    if warn:
        return '⚠️'
    return '❌'


def meets_thresholds(data: Dict[str, Any]) -> Dict[str, bool]:
    settings = data.get('settings') or {}
    metrics = data.get('metrics') or {}
    fill = metrics.get('fill') or {}
    llm = metrics.get('llm') or {}
    noise = metrics.get('noise') or {}
    filters = metrics.get('filters') or {}
    diag = (data.get('diagnostics') or {}).get('diag_broll') or {}

    selected = fill.get('selected') or 0
    target = fill.get('target') or EXPECTED_SEGMENTS
    allow_images = int(settings.get('allow_images') or 0)
    fill_requirement = 12 if allow_images else 10

    thresholds = {
        'fill': selected >= fill_requirement,
        'empty_payload': (llm.get('empty_payload_count') or 0) == 0,
        'segments': (llm.get('llm_segment_count') or 0) >= 8,
        'noise': (noise.get('provider_noise_count') or 0) == 0 and (noise.get('generic_terms_count') or 0) == 0,
        'duration': (filters.get('duration_drops') or 0) <= 60,
        'orientation': (filters.get('orientation_drops') or 0) <= 120,
        'diag_broll': (diag.get('success_count') or 0) > 0 and (diag.get('candidates_total') or 0) > 0,
    }
    return thresholds


def load_probe_map(probe_path: Path) -> Dict[str, Dict[str, Any]]:
    data = load_json(probe_path)
    if not data:
        return {}
    models = {}
    for entry in data.get('models', []):
        if 'model' in entry:
            models[entry['model']] = entry
    return models


def score_models(bench_paths: List[Path], probe_map: Dict[str, Dict[str, Any]]) -> List[ModelScore]:
    scored: List[ModelScore] = []
    for path in bench_paths:
        data = load_json(path)
        if not data:
            continue
        metrics = data.get('metrics') or {}
        fill = metrics.get('fill') or {}
        llm = metrics.get('llm') or {}
        noise = metrics.get('noise') or {}
        filters = metrics.get('filters') or {}
        diag = (data.get('diagnostics') or {}).get('diag_broll') or {}

        fill_ratio, fill_points = compute_fill_score(
            fill.get('selected'),
            fill.get('target'),
            fill.get('fill_rate'),
        )
        dynamic_ratio, dynamic_points = compute_dynamic_score(
            llm.get('empty_payload_count') or 0,
            llm.get('llm_segment_count') or 0,
        )
        drop_ratio, drop_points = compute_drop_score(
            float(filters.get('duration_drops') or 0.0),
            float(filters.get('orientation_drops') or 0.0),
        )
        provider_ratio, provider_points = compute_provider_score(diag)
        noise_ratio, noise_points = compute_noise_score(
            noise.get('provider_noise_count') or 0,
            noise.get('generic_terms_count') or 0,
        )

        total_score = round(fill_points + dynamic_points + drop_points + provider_points + noise_points, 2)
        thresholds = meets_thresholds(data)

        scored.append(
            ModelScore(
                model=data.get('model', path.stem),
                safe_name=data.get('safe_name', path.stem),
                score=total_score,
                components={
                    'fill': round(fill_ratio, 3),
                    'dynamic': round(dynamic_ratio, 3),
                    'drops': round(drop_ratio, 3),
                    'providers': round(provider_ratio, 3),
                    'noise': round(noise_ratio, 3),
                },
                metrics=data,
                passes=thresholds,
                bench_path=str(path),
            )
        )
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored


def determine_recommendation(scores: List[ModelScore], probe_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    best_pass = next((item for item in scores if all(item.passes.values())), None)
    if best_pass:
        env_block = [
            f'$env:PIPELINE_LLM_ENDPOINT="{BASE_ENDPOINT}"',
            f'$env:PIPELINE_LLM_MODEL="{best_pass.model}"',
            '$env:PIPELINE_LLM_JSON_MODE="1"',
        ]
        return {
            'type': 'single',
            'model': best_pass.model,
            'env': env_block,
        }

    json_model = None
    text_model = None

    for model in scores:
        probe = probe_map.get(model.model) or {}
        json_info = probe.get('json_strict') or {}
        if json_info.get('valid'):
            json_model = model.model
            break

    def text_quality(entry: Dict[str, Any]) -> Tuple[int, float, float]:
        if not entry:
            return (1, float('inf'), float('inf'))
        text_block = entry.get('text') or {}
        long_resp = text_block.get('long') or {}
        short_resp = text_block.get('short') or {}
        empty_count = int(long_resp.get('empty', False)) + int(short_resp.get('empty', False))
        errors = (len(long_resp.get('errors') or []) + len(short_resp.get('errors') or []))
        latency = float(long_resp.get('latency_sec') or math.inf)
        return (empty_count + errors, latency, float(short_resp.get('latency_sec') or math.inf))

    sorted_probe = sorted(
        probe_map.items(),
        key=lambda item: text_quality(item[1]),
    )
    for model_name, entry in sorted_probe:
        quality = text_quality(entry)
        if quality[0] == 0:
            text_model = model_name
            break
    if not text_model and sorted_probe:
        text_model = sorted_probe[0][0]

    env_block = [
        f'$env:PIPELINE_LLM_ENDPOINT="{BASE_ENDPOINT}"',
    ]
    if text_model:
        env_block.append(f'$env:PIPELINE_LLM_MODEL_TEXT="{text_model}"')
    if json_model:
        env_block.append(f'$env:PIPELINE_LLM_MODEL_JSON="{json_model}"')
    env_block.append('$env:PIPELINE_LLM_JSON_MODE="1"')

    return {
        'type': 'combo',
        'text_model': text_model,
        'json_model': json_model,
        'env': env_block,
    }


def build_markdown(scores: List[ModelScore], recommendation: Dict[str, Any]) -> str:
    lines = [
        '# Benchmark Summary',
        '',
        f'Generated: {datetime.now(timezone.utc).isoformat()}',
        '',
        '| Model | Score | Fill | Empty Payload | Segments | Noise | Duration | Orientation | Providers |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for item in scores:
        metrics = item.metrics.get('metrics') or {}
        fill = metrics.get('fill') or {}
        llm = metrics.get('llm') or {}
        noise = metrics.get('noise') or {}
        filters = metrics.get('filters') or {}
        diag = (item.metrics.get('diagnostics') or {}).get('diag_broll') or {}
        passes = item.passes
        lines.append(
            '| {model} | {score:.2f} | {fill_badge} {fill_val} | {empty_badge} {empty_cnt} | {seg_badge} {segments} | '
            '{noise_badge} {noise_val} | {dur_badge} {dur_val} | {ori_badge} {ori_val} | {prov_badge} {prov_val} |'.format(
                model=item.model,
                score=item.score,
                fill_badge=badge(passes.get('fill', False)),
                fill_val=f"{fill.get('selected') or 0}/{fill.get('target') or EXPECTED_SEGMENTS}",
                empty_badge=badge(passes.get('empty_payload', False)),
                empty_cnt=llm.get('empty_payload_count') or 0,
                seg_badge=badge(passes.get('segments', False)),
                segments=llm.get('llm_segment_count') or 0,
                noise_badge=badge(passes.get('noise', False)),
                noise_val=(noise.get('provider_noise_count') or 0) + (noise.get('generic_terms_count') or 0),
                dur_badge=badge(passes.get('duration', False)),
                dur_val=filters.get('duration_drops') or 0,
                ori_badge=badge(passes.get('orientation', False)),
                ori_val=filters.get('orientation_drops') or 0,
                prov_badge=badge(passes.get('diag_broll', False)),
                prov_val=diag.get('success_count') or 0,
            )
        )
    lines.append('')

    if recommendation.get('type') == 'single':
        lines.append('## Recommendation')
        lines.append(f"Best model: **{recommendation.get('model')}** (passes all thresholds)")
    else:
        lines.append('## Recommendation')
        json_model = recommendation.get('json_model') or 'n/a'
        text_model = recommendation.get('text_model') or 'n/a'
        lines.append(f"JSON strict model: **{json_model}**, text model: **{text_model}**")
        lines.append('Both models should be wired in the pipeline with sensible fallbacks to `$env:PIPELINE_LLM_MODEL` if specialized vars are absent.')
    lines.append('')
    lines.append('### Environment Block')
    lines.append('```powershell')
    for line in recommendation.get('env', []):
        lines.append(line)
    lines.append('```')
    lines.append('')
    lines.append('Note: update `pipeline_core/llm_service.py` (or the module that reads `PIPELINE_LLM_MODEL`) to optionally consume `PIPELINE_LLM_MODEL_TEXT` and `PIPELINE_LLM_MODEL_JSON` when present, falling back to the legacy variable otherwise.')
    return '\n'.join(lines)


def main() -> int:
    out_dir = Path('tools/out')
    bench_files = sorted(out_dir.glob('bench_*.json'))
    if not bench_files:
        print('No bench_*.json files found in tools/out', file=sys.stderr)
        return 1

    probe_path = out_dir / 'llm_probe_results.json'
    probe_map = load_probe_map(probe_path)

    scores = score_models(bench_files, probe_map)
    if not scores:
        print('Unable to parse bench files', file=sys.stderr)
        return 1

    recommendation = determine_recommendation(scores, probe_map)

    summary_payload = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'models': [
            {
                'model': item.model,
                'safe_name': item.safe_name,
                'score': item.score,
                'components': item.components,
                'passes': item.passes,
                'bench_path': item.bench_path,
            }
            for item in scores
        ],
        'recommendation': recommendation,
    }

    summary_json_path = out_dir / 'summary.json'
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding='utf-8')

    summary_md_path = out_dir / 'summary.md'
    summary_md_path.write_text(build_markdown(scores, recommendation), encoding='utf-8')

    return 0


if __name__ == '__main__':
    sys.exit(main())
