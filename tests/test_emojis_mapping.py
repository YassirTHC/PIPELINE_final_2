from collections import deque

from hormozi_subtitles import HormoziSubtitles


def _token_for(proc: HormoziSubtitles, word: str, keyword: bool = True):
    normalized = word.upper()
    category = proc.keyword_to_category.get(normalized)
    if not category:
        category = proc.emoji_alias.get(normalized)
    color = proc._category_color(category) if category else "#FFFFFF"
    return {
        "text": normalized,
        "normalized": normalized,
        "is_keyword": keyword,
        "color": color,
        "category": category,
    }


def test_category_emojis_cover_primary_domains():
    proc = HormoziSubtitles()
    categories = ['finance', 'sales', 'content', 'mobile', 'sports']
    for category in categories:
        emojis = proc.category_emojis.get(category)
        assert emojis, f"missing emojis for {category}"


def test_emoji_anti_repeat_avoids_duplicates():
    proc = HormoziSubtitles()
    tokens = [_token_for(proc, 'money')]
    first = proc._choose_emoji_for_tokens(tokens, 'finance wins')
    proc._recent_emojis.append(first)
    second = proc._choose_emoji_for_tokens(tokens, 'finance wins')
    assert first != second


def test_emoji_returns_empty_when_no_category():
    proc = HormoziSubtitles()
    proc._recent_emojis = deque(maxlen=3)
    tokens = [_token_for(proc, 'hello', keyword=False)]
    result = proc._choose_emoji_for_tokens(tokens, 'hello world')
    assert result == ""


def test_money_sales_growth_energy_have_emojis():
    proc = HormoziSubtitles()
    words = ['money', 'sales', 'growth', 'energy']
    for word in words:
        tokens = [_token_for(proc, word)]
        emoji = proc._choose_emoji_for_tokens(tokens, word)
        assert emoji, f"expected emoji for {word}"
