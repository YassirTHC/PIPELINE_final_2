import json

import pytest

from pipeline_core.llm_service import _expand_term_field, _concretize_queries


def test_expand_term_field_handles_scalar_and_nested_payloads():
    payload = {
        "comma": "alpha, beta",
        "json_array": json.dumps(["gamma", ["delta", {"extra": "epsilon"}]]),
        "dict": {"one": "zeta", "two": ["eta", "theta"]},
        "iterable": {"iota", "kappa"},
        "bytes": "lambda".encode("utf-8"),
        "number": 42,
        "bool": True,
    }

    terms = _expand_term_field(payload)

    assert "alpha" in terms and "beta" in terms
    assert "gamma" in terms and "delta" in terms and "epsilon" in terms
    assert "zeta" in terms and "eta" in terms and "theta" in terms
    assert "iota" in terms or "kappa" in terms
    assert "lambda" in terms
    assert "42" in terms
    assert "True" not in terms  # bools should be ignored entirely


@pytest.mark.parametrize(
    "raw,expected_options",
    [
        ("process framework", {"goal planner journal", "motivation quote wall"}),
        (["abstract concept"], {"abstract concept"}),
        (["team planning roadmap"], {"team planning roadmap"}),
        (["motivation process", "visual storytelling"], {"motivation process", "storytelling"}),
        ([{"primary": "growth mindset visual"}], {"growth mindset"}),
    ],
)
def test_concretize_queries_accepts_various_structures(raw, expected_options):
    concrete = _concretize_queries(raw)
    assert any(term in concrete for term in expected_options)
    assert all(isinstance(term, str) and term.strip() for term in concrete)
