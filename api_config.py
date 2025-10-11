#!/usr/bin/env python3
"""API key bootstrap for the B-roll system."""

import os

DEFAULT_API_KEYS = {
    'PEXELS_API_KEY': 'pwhBa9K7fa9IQJCmfCy0NfHFWy8QyqoCkGnWLK3NC2SbDTtUeuhxpDoD',
    'PIXABAY_API_KEY': '51724939-ee09a81ccfce0f5623df46a69',
}

DEFAULT_FETCH_ENV = {
    'BROLL_FETCH_ENABLE': '1',
    'BROLL_FETCH_PROVIDER': 'pixabay,pexels',
    'BROLL_FETCH_ALLOW_VIDEOS': '1',
    'BROLL_FETCH_ALLOW_IMAGES': '1',
    'BROLL_FETCH_MAX_PER_KEYWORD': '8',
    'BROLL_PEXELS_MAX_PER_KEYWORD': '3',
    'FETCH_MAX': '8',
    'ENABLE_PIPELINE_CORE_FETCHER': 'true',
    'BROLL_DELETE_AFTER_USE': '1',
    'BROLL_PURGE_AFTER_RUN': '1',
}


def _mask(value: str | None) -> str:
    if not value:
        return 'missing'
    trimmed = value.strip()
    if len(trimmed) <= 4:
        return '****'
    return f"{trimmed[:8]}******"


def apply_defaults() -> None:
    """Populate environment variables with sensible defaults."""

    for key, value in DEFAULT_API_KEYS.items():
        os.environ.setdefault(key, value)

    for key, value in DEFAULT_FETCH_ENV.items():
        os.environ.setdefault(key, value)

    provider_value = os.environ.get('BROLL_FETCH_PROVIDER', DEFAULT_FETCH_ENV['BROLL_FETCH_PROVIDER'])
    os.environ.setdefault('AI_BROLL_FETCH_PROVIDER', provider_value)


def print_summary() -> None:
    print('API key configuration applied')
    for env_key in ('PEXELS_API_KEY', 'PIXABAY_API_KEY'):
        print(f' {env_key}: {_mask(os.environ.get(env_key))}')
    print(f" BROLL_FETCH_PROVIDER: {os.environ.get('BROLL_FETCH_PROVIDER', 'unset')}")
    print(f" AI_BROLL_FETCH_PROVIDER: {os.environ.get('AI_BROLL_FETCH_PROVIDER', 'unset')}")
    print(f" FETCH_MAX: {os.environ.get('FETCH_MAX', 'unset')}")
    print(f" BROLL_FETCH_MAX_PER_KEYWORD: {os.environ.get('BROLL_FETCH_MAX_PER_KEYWORD', 'unset')}")
    print(f" BROLL_PEXELS_MAX_PER_KEYWORD: {os.environ.get('BROLL_PEXELS_MAX_PER_KEYWORD', 'unset')}")


if __name__ == '__main__':
    apply_defaults()
    print_summary()
