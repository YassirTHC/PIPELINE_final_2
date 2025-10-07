from hormozi_subtitles import HormoziSubtitles


def _build_segment(words, start=0.0, step=0.4):
    payload = []
    for index, word in enumerate(words):
        payload.append(
            {
                "word": word,
                "text": word,
                "start": start + index * step,
                "end": start + (index + 1) * step,
            }
        )
    return {
        "text": " ".join(words),
        "start": start,
        "end": start + len(words) * step,
        "words": payload,
    }


def _emoji_indices(groups):
    placements = []
    for idx, group in enumerate(groups):
        emojis = group.get("emojis") or []
        if not emojis:
            continue
        entry = emojis[0]
        emoji = entry if isinstance(entry, str) else entry.get("char")
        placements.append((idx, emoji))
    return placements


def test_density_and_gap_are_respected_across_segments():
    proc = HormoziSubtitles()
    proc.config.update({
        'emoji_target_per_10': 4,
        'emoji_min_gap_groups': 2,
        'emoji_max_per_segment': 3,
    })

    seg1_words = [
        'profit', 'energy', 'offer', 'sales', 'growth',
        'mobile', 'sports', 'success', 'money', 'alert',
    ]
    seg2_words = [
        'cash', 'energy', 'sales', 'success', 'mobile',
        'sports', 'profit', 'alert', 'growth', 'money',
    ]

    groups1 = proc.parse_transcription_to_word_groups([_build_segment(seg1_words)], group_size=2)
    placements1 = _emoji_indices(groups1)
    min_gap = proc.config['emoji_min_gap_groups']
    for (idx_a, _), (idx_b, _) in zip(placements1, placements1[1:]):
        assert idx_b - idx_a >= min_gap

    expected1 = int(round(len(groups1) * proc.config['emoji_target_per_10'] / 10.0))
    expected1 = min(expected1, proc.config['emoji_max_per_segment'])
    assert len(placements1) == expected1

    groups2 = proc.parse_transcription_to_word_groups([_build_segment(seg2_words, start=5.0)], group_size=2)
    placements2 = _emoji_indices(groups2)
    for (idx_a, _), (idx_b, _) in zip(placements2, placements2[1:]):
        assert idx_b - idx_a >= min_gap

    expected2 = int(round(len(groups2) * proc.config['emoji_target_per_10'] / 10.0))
    expected2 = min(expected2, proc.config['emoji_max_per_segment'])
    assert len(placements2) == expected2

    total_groups = len(groups1) + len(groups2)
    cumulative_target = int(round(total_groups * proc.config['emoji_target_per_10'] / 10.0))
    actual_total = len(placements1) + len(placements2)
    assert actual_total == cumulative_target

    # No emoji should repeat immediately even across segments
    recent = []
    window = proc.config.get('emoji_history_window', 4)
    for _, emoji in placements1 + placements2:
        assert emoji not in recent[-window:]
        recent.append(emoji)
