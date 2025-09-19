from types import SimpleNamespace

from video_processor import SEEN_URLS


def test_url_dedupe_basic():
    SEEN_URLS.clear()
    pool = [
        SimpleNamespace(url="https://example.com/clip.mp4"),
        SimpleNamespace(url="https://example.com/clip.mp4"),
    ]

    unique = []
    for candidate in pool:
        if candidate.url in SEEN_URLS:
            continue
        SEEN_URLS.add(candidate.url)
        unique.append(candidate)

    assert len(unique) == 1
