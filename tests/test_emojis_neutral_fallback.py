from hormozi_subtitles import HormoziSubtitles


def _build_segment(words, start=0.0, step=0.5):
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


def test_no_emoji_emitted_for_neutral_text():
    proc = HormoziSubtitles()
    proc.config['emoji_no_context_fallback'] = ""
    neutral_words = [
        'and', 'the', 'maybe', 'because', 'however', 'between', 'although', 'besides'
    ]
    segment = _build_segment(neutral_words)
    groups = proc.parse_transcription_to_word_groups([segment], group_size=2)
    assert all(not group.get('emojis') for group in groups)


def test_previous_usage_does_not_force_fallback():
    proc = HormoziSubtitles()
    proc.config['emoji_no_context_fallback'] = ""

    rich_segment = _build_segment(['profit', 'sales', 'growth', 'energy', 'success'])
    proc.parse_transcription_to_word_groups([rich_segment], group_size=2)

    neutral_segment = _build_segment(['anyway', 'perhaps', 'maybe', 'still'], start=5.0)
    groups = proc.parse_transcription_to_word_groups([neutral_segment], group_size=2)
    assert all(not group.get('emojis') for group in groups)
