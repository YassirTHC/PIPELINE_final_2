"""Selection helpers (MMR, diversity) for the B-roll pipeline."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_WORD_RE = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)


def cand_key(c):
    """
    Immutable key for dict/set operations that need to deduplicate candidates while preserving order.
    """
    provider = getattr(c, "provider", None)
    pid = getattr(c, "asset_id", None) or getattr(c, "id", None) or getattr(c, "url", None)
    return (provider, pid)


def _candidate_tokens(candidate: Any) -> Tuple[str, ...]:
    parts: List[str] = []
    if candidate is None:
        return tuple()
    title = getattr(candidate, "title", None)
    if isinstance(title, str):
        parts.append(title)
    description = getattr(candidate, "description", None)
    if isinstance(description, str):
        parts.append(description)
    tags = getattr(candidate, "tags", None)
    if isinstance(tags, Iterable):
        for tag in tags:
            if isinstance(tag, str):
                parts.append(tag)
    raw = " ".join(parts).lower()
    return tuple(sorted(_WORD_RE.findall(raw)))


def _jaccard_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    if not a or not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    if not set_a or not set_b:
        return 0.0
    intersection = set_a.intersection(set_b)
    if not intersection:
        return 0.0
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return float(len(intersection)) / float(len(union))


def _asset_identifier(candidate: Any) -> Optional[str]:
    identifier = getattr(candidate, "identifier", None)
    if isinstance(identifier, str) and identifier.strip():
        return identifier.strip()
    url = getattr(candidate, "url", None)
    if isinstance(url, str) and url.strip():
        return url.strip()
    return None


def mmr_rerank(
    records: Sequence[Dict[str, Any]],
    *,
    alpha: float = 0.7,
    max_candidates: Optional[int] = None,
    recent_assets: Optional[Dict[str, int]] = None,
    repeat_penalty: float = 0.25,
    repeat_window: int = 2,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Perform a Maximal Marginal Relevance re-ranking to encourage diversity."""

    if not records:
        return [], {"repeat_penalties": 0, "dropped": 0}

    remaining = list(records)
    selected: List[Dict[str, Any]] = []
    token_cache: Dict[Tuple[Optional[str], Optional[str]], Tuple[str, ...]] = {}
    penalty_hits = 0

    def candidate_tokens(candidate: Any) -> Tuple[str, ...]:
        key = cand_key(candidate)
        if key in token_cache:
            return token_cache[key]
        tokens = _candidate_tokens(candidate)
        token_cache[key] = tokens
        return tokens

    while remaining:
        best_record: Optional[Dict[str, Any]] = None
        best_score = -math.inf
        best_penalised = False

        for record in remaining:
            candidate = record.get("candidate")
            base_score = float(record.get("score") or getattr(candidate, "provider_score", 0.0) or 0.0)
            if not selected:
                novelty_boost = 1.0
            else:
                candidate_tokens_current = candidate_tokens(candidate)
                max_sim = 0.0
                for sel in selected:
                    sel_tokens = candidate_tokens(sel.get("candidate"))
                    max_sim = max(max_sim, _jaccard_similarity(candidate_tokens_current, sel_tokens))
                novelty_boost = 1.0 - max_sim
            score = (alpha * base_score) + ((1.0 - alpha) * novelty_boost)

            penalised = False
            if recent_assets and repeat_window > 0:
                asset_id = _asset_identifier(candidate)
                if asset_id:
                    distance = recent_assets.get(asset_id)
                    if distance is not None and 0 < distance <= repeat_window:
                        score -= repeat_penalty * (1.0 / (1.0 + float(distance)))
                        penalised = True

            if score > best_score:
                best_score = score
                best_record = record
                best_penalised = penalised

        if best_record is None:
            break

        if best_penalised:
            penalty_hits += 1
        selected.append(best_record)
        remaining.remove(best_record)

        if max_candidates is not None and len(selected) >= max_candidates:
            break

    stats = {
        "repeat_penalties": penalty_hits,
        "dropped": max(0, len(records) - len(selected)),
    }
    return selected, stats


__all__ = ["cand_key", "mmr_rerank"]
