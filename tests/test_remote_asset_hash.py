from pipeline_core.fetchers import RemoteAssetCandidate


def test_remote_asset_candidate_hash():
    a = RemoteAssetCandidate(
        provider="pexels",
        url="https://example.com/video/123",
        thumb_url=None,
        width=1920,
        height=1080,
        duration=12.5,
        title="Clip A",
        identifier="123",
        tags=[],
    )
    b = RemoteAssetCandidate(
        provider="pexels",
        url="https://example.com/video/123",
        thumb_url=None,
        width=1920,
        height=1080,
        duration=12.5,
        title="Clip A duplicate",
        identifier="123",
        tags=[],
    )
    c = RemoteAssetCandidate(
        provider="pexels",
        url="https://example.com/video/456",
        thumb_url=None,
        width=1280,
        height=720,
        duration=8.0,
        title="Clip B",
        identifier="456",
        tags=[],
    )

    s = {a, b, c}
    assert len(s) == 2

    d = {a: 1, b: 2, c: 3}
    assert d[a] == 2
