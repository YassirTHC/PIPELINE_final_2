from hormozi_subtitles import HormoziSubtitles


def _build_segment(words, start=0.0, step=0.3):
    payload = []
    for idx, word in enumerate(words):
        payload.append({
            "word": word,
            "text": word,
            "start": start + idx * step,
            "end": start + (idx + 1) * step,
        })
    return {
        "text": " ".join(words),
        "start": start,
        "end": start + len(words) * step,
        "words": payload,
    }


def test_emoji_spacing_and_density():
    proc = HormoziSubtitles()
    words = [
        "profit", "energy", "offer", "sales", "growth", "profit", "energy", "money", "offer", "profit",
        "energy", "sales", "profit", "money", "energy", "offer", "profit", "energy", "sales", "profit",
    ]
    segment = _build_segment(words)
    groups = proc.parse_transcription_to_word_groups([segment], group_size=2)
    emoji_indices = [idx for idx, group in enumerate(groups) if group.get("emojis")]

    min_gap = proc.config["emoji_min_gap_groups"]
    for a, b in zip(emoji_indices, emoji_indices[1:]):
        assert b - a >= min_gap

    target = proc.config["emoji_target_per_10"]
    expected = round(len(groups) * target / 10)
    expected = min(expected, proc.config["emoji_max_per_segment"])
    assert abs(len(emoji_indices) - expected) <= 1

    recent = []
    window = proc.config.get("emoji_history_window", 4)
    for group in groups:
        if not group.get("emojis"):
            continue
        emoji = group["emojis"][0]
        assert emoji not in recent[-window:]
        recent.append(emoji)


def test_emoji_fallback_is_empty_when_no_category():
    proc = HormoziSubtitles()
    bland_words = ["and", "the", "maybe", "because"]
    segment = _build_segment(bland_words, step=0.5)
    groups = proc.parse_transcription_to_word_groups([segment], group_size=2)
    assert all(not group.get("emojis") for group in groups)
