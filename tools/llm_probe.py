#!/usr/bin/env python3
"""Utility to probe Ollama models for JSON strict and text generation behaviour."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests


REQUEST_CONNECT_TIMEOUT_S = 10
REQUEST_READ_TIMEOUT_S = 90
RAW_SNIPPET_LIMIT = 500
JSON_PROMPT = 'Return strictly this JSON: {"title":"t","keywords":["a","b","c"]}'
SHORT_PROMPT = 'Respond with the word OK only.'
LONG_PROMPT = (
    "Imagine you are coaching a creator on how to frame vertical footage for social media. "
    "Explain, in about six sentences, how to position the camera, maintain subject headroom, "
    "leverage leading lines, incorporate subtle movement, balance negative space, and align captions "
    "with the main action. Mention considerations for lighting, background cleanup, keeping the subject "
    "inside safe areas for 9:16 and 4:5 aspect ratios, and planning transitions that feel smooth on a phone. "
    "Include a reminder about checking horizon lines, minimizing jitter through stabilization, previewing shots "
    "in editing apps, and planning complementary B-roll coverage. Encourage quick framing tests before recording "
    "and note how text overlays should respect top and bottom safe zones to stay readable."
)


@dataclass
class ProbeResult:
    """Container for a single generation attempt."""

    text: str
    elapsed: Optional[float]
    raw_chunks: List[str]
    errors: List[str]
    status_code: Optional[int]


def iter_ollama_chunks(response: requests.Response) -> Iterable[str]:
    """Yield decoded lines from a streaming Ollama response."""

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        yield line


def call_ollama(endpoint: str, payload: Dict[str, Any], timeout_s: float) -> ProbeResult:
    """Call the Ollama generate API and aggregate the streamed content."""

    url = endpoint.rstrip('/') + '/api/generate'
    start = time.perf_counter()
    text_parts: List[str] = []
    raw_chunks: List[str] = []
    errors: List[str] = []
    status_code: Optional[int] = None

    try:
        with requests.post(
            url,
            json=payload,
            stream=True,
            timeout=(REQUEST_CONNECT_TIMEOUT_S, timeout_s),
        ) as response:
            status_code = response.status_code
            if response.status_code != 200:
                body_preview = response.text[:RAW_SNIPPET_LIMIT]
                return ProbeResult(
                    text='',
                    elapsed=None,
                    raw_chunks=[body_preview],
                    errors=[f'http_{response.status_code}'],
                    status_code=status_code,
                )
            for chunk in iter_ollama_chunks(response):
                if len(raw_chunks) < 64:
                    raw_chunks.append(chunk)
                try:
                    parsed = json.loads(chunk)
                except json.JSONDecodeError as exc:
                    errors.append(f'chunk_decode_error: {exc}')
                    continue
                if isinstance(parsed, dict):
                    piece = parsed.get('response', '')
                    if piece:
                        text_parts.append(piece)
        elapsed = time.perf_counter() - start
        return ProbeResult(
            text=''.join(text_parts),
            elapsed=elapsed,
            raw_chunks=raw_chunks,
            errors=errors,
            status_code=status_code,
        )
    except requests.RequestException as exc:
        errors.append(f'request_exception: {exc}')
        return ProbeResult(
            text='',
            elapsed=None,
            raw_chunks=raw_chunks,
            errors=errors,
            status_code=status_code,
        )


def ensure_ascii_snippet(text: str) -> str:
    """Return a truncated raw snippet for logging."""

    if len(text) <= RAW_SNIPPET_LIMIT:
        return text
    return text[:RAW_SNIPPET_LIMIT] + '...'


def _coerce_chunk_text(chunk: Any) -> str:
    if isinstance(chunk, (bytes, bytearray)):
        return chunk.decode('utf-8', 'ignore')
    if chunk is None:
        return ''
    return str(chunk)


def has_non_ascii(text: str) -> bool:
    """Check whether the response contains non-ASCII characters."""

    return any(ord(char) > 127 for char in text)


def probe_model(endpoint: str, model: str) -> Dict[str, Any]:
    """Probe a model for JSON strict and plain text behaviours."""

    base_payload = {
        'model': model,
        'options': {
            'temperature': 0.2,
            'top_p': 0.85,
            'num_predict': 192,
        },
    }

    json_payload = dict(base_payload)
    json_payload.update({'format': 'json', 'prompt': JSON_PROMPT})
    json_result = call_ollama(endpoint, json_payload, REQUEST_READ_TIMEOUT_S)

    json_validation: Dict[str, Any] = {
        'valid': False,
        'error': None,
        'parsed': None,
        'response_text': json_result.text,
        'raw_preview': ensure_ascii_snippet(
            '\n'.join(_coerce_chunk_text(c) for c in json_result.raw_chunks)
        ),
        'errors': json_result.errors,
        'status_code': json_result.status_code,
        'elapsed_sec': json_result.elapsed,
    }
    if json_result.errors and not json_validation['error']:
        json_validation['error'] = json_result.errors[0]
    if json_result.text:
        try:
            parsed = json.loads(json_result.text)
            json_validation['parsed'] = parsed
            if isinstance(parsed, dict) and 'title' in parsed and 'keywords' in parsed:
                json_validation['valid'] = True
        except json.JSONDecodeError as exc:
            json_validation['error'] = f'json_decode_error: {exc}'
    else:
        if not json_validation['error']:
            json_validation['error'] = 'empty_json_response'

    text_prompts = {
        'short': SHORT_PROMPT,
        'long': LONG_PROMPT,
    }
    text_results: Dict[str, Any] = {}
    for key, prompt in text_prompts.items():
        payload = dict(base_payload)
        payload.update({'prompt': prompt})
        result = call_ollama(endpoint, payload, REQUEST_READ_TIMEOUT_S)
        response_text = result.text.strip()
        text_results[key] = {
            'latency_sec': result.elapsed,
            'length': len(result.text),
            'empty': response_text == '',
            'non_ascii': has_non_ascii(result.text),
            'raw_preview': ensure_ascii_snippet(
                '\n'.join(_coerce_chunk_text(c) for c in result.raw_chunks)
            ),
            'errors': result.errors,
            'status_code': result.status_code,
            'response': response_text,
        }

    return {
        'model': model,
        'endpoint': endpoint,
        'json_strict': json_validation,
        'text': text_results,
    }


def format_text_summary(meta: Dict[str, Any]) -> str:
    """Format a text probe summary for the Markdown table."""

    if not meta:
        return 'n/a'
    latency = meta.get('latency_sec')
    latency_str = f"{latency:.2f}s" if isinstance(latency, (int, float)) else 'n/a'
    length = meta.get('length', 0)
    empty = 'empty' if meta.get('empty') else 'ok'
    non_ascii = '⚠️' if meta.get('non_ascii') else '✅'
    errors = meta.get('errors') or []
    if errors:
        return f"{latency_str}, len={length}, {empty}, {non_ascii}, errors"
    return f"{latency_str}, len={length}, {empty}, {non_ascii}"


def build_markdown(results: List[Dict[str, Any]]) -> str:
    """Create a human-readable Markdown summary."""

    lines = [
        '# LLM Probe Results',
        '',
        f'Generated: {datetime.now(timezone.utc).isoformat()}',
        '',
        '| Model | JSON Strict | Short Prompt | Long Prompt |',
        '| --- | --- | --- | --- |',
    ]
    for entry in results:
        model = entry['model']
        json_ok = '✅' if entry['json_strict'].get('valid') else '❌'
        short_desc = format_text_summary(entry['text'].get('short', {}))
        long_desc = format_text_summary(entry['text'].get('long', {}))
        lines.append(f'| {model} | {json_ok} | {short_desc} | {long_desc} |')
    lines.append('')
    lines.append('Notes: latency is in seconds, length counts raw characters, and non-ASCII detection is heuristic.')
    return '\n'.join(lines)


def parse_models(value: str) -> List[str]:
    """Split the CSV models argument into a list."""

    models = [item.strip() for item in value.split(',') if item.strip()]
    if not models:
        raise argparse.ArgumentTypeError('provide at least one model name')
    return models


def main(argv: Optional[List[str]] = None) -> int:
    """Argument parsing and file emission entry point."""

    parser = argparse.ArgumentParser(description='Probe Ollama models for JSON and text responses.')
    parser.add_argument('--models', required=True, type=parse_models, help='Comma-separated list of models to test')
    parser.add_argument('--endpoint', default='http://localhost:11434', help='Ollama endpoint base URL')
    parser.add_argument('--output-dir', default='tools/out', help='Directory to store probe results')
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for model in args.models:
        results.append(probe_model(args.endpoint, model))

    payload = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'endpoint': args.endpoint,
        'models': results,
    }

    json_path = output_dir / 'llm_probe_results.json'
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

    md_path = output_dir / 'llm_probe_results.md'
    md_path.write_text(build_markdown(results), encoding='utf-8')

    return 0


if __name__ == '__main__':
    sys.exit(main())
