from hormozi_subtitles import HormoziSubtitles


def test_primary_categories_have_multiple_emojis():
    proc = HormoziSubtitles()
    critical = [
        'finance',
        'sales',
        'content',
        'growth',
        'energy',
        'focus',
        'time',
        'success',
        'alert',
        'mobile',
        'sports',
        'health',
        'education',
    ]
    for category in critical:
        emojis = proc.category_emojis.get(category)
        assert emojis, f"missing emoji list for {category}"
        non_empty = [emoji for emoji in emojis if emoji]
        assert len(non_empty) >= 2, f"expected >=2 emojis for {category}, got {non_empty}"


def test_aliases_cover_common_fr_en_keywords():
    proc = HormoziSubtitles()
    expectations = {
        'ARGENT': 'finance',
        'VENTE': 'sales',
        'OFFRE': 'sales',
        'TELEPHONE': 'mobile',
        'VIRAL': 'content',
        'FOOT': 'sports',
        'ENERGIE': 'energy',
        'MINUTE': 'time',
        'RECOMPENSE': 'success',
        'ALERTE': 'alert',
    }
    for keyword, category in expectations.items():
        normalized = proc._normalize(keyword)
        alias_category = proc.emoji_alias.get(normalized)
        if not alias_category:
            alias_category = proc.keyword_to_category.get(normalized)
        assert alias_category == category, f"{keyword} should map to {category}, got {alias_category}"
