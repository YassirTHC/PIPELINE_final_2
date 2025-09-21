def test_segment_terms_from_briefs_picks_and_limits():
    from video_processor import _segment_terms_from_briefs
    dyn = {"segment_briefs": [
        {"segment_index": 0, "keywords": ["deep_work", "focus"], "queries": ["laptop focus", "office desk"]},
        {"segment_index": 1, "keywords": ["sleep_hygiene", "people"], "queries": ["dark bedroom", "nice background"]},
    ]}
    out0 = _segment_terms_from_briefs(dyn, 0, 3)
    assert out0 and len(out0) <= 3
    # underscores normalized to spaces
    assert any("deep work" == t for t in out0)

    # segment 1 filters anti-terms like 'people' and 'nice background'
    out1 = _segment_terms_from_briefs(dyn, 1, 4)
    assert all(x not in {"people", "nice background"} for x in out1)

