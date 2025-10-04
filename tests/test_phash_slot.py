from pipeline_core.fetchers import RemoteAssetCandidate
import pytest

def test_phash_slot_and_repr():
    c = RemoteAssetCandidate(
        provider="p",
        url="u",
        thumb_url=None,
        width=1,
        height=2,
        duration=0.5,
        title="",
        identifier="",
        tags=[],
    )
    assert hasattr(c, "_phash") and c._phash is None
    c._phash = "abc"
    assert c._phash == "abc"
    assert "_phash" not in repr(c)
    with pytest.raises(AttributeError):
        c.new_field = 1
