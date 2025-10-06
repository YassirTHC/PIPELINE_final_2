from collections import deque

from hormozi_subtitles import HormoziSubtitles


def _token_for(word: str, keyword: bool = True):
    return {
        "text": word.upper(),
        "normalized": word.upper(),
        "is_keyword": keyword,
        "color": "#FFD54F",
    }


def test_category_emojis_cover_primary_domains():
    proc = HormoziSubtitles()
    categories = ['finance', 'sales', 'content', 'mobile', 'sports']
    for category in categories:
        emojis = proc.category_emojis.get(category)
        assert emojis, f"missing emojis for {category}"


def test_emoji_anti_repeat_avoids_duplicates():
    proc = HormoziSubtitles()
    tokens = [_token_for('money')]
    first = proc._choose_emoji_for_tokens(tokens, 'finance wins')
    second = proc._choose_emoji_for_tokens(tokens, 'finance wins')
    assert first != second


def test_emoji_returns_empty_when_no_category():
    proc = HormoziSubtitles()
    proc._recent_emojis = deque(maxlen=3)
    tokens = [_token_for('hello', keyword=False)]
    result = proc._choose_emoji_for_tokens(tokens, 'hello world')
    assert result == ""
