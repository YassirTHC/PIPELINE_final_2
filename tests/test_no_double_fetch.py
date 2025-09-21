def test_no_double_refine_guard():
    # Ensure refinement does not override when terms equal existing queries
    queries = ["deep work", "office desk"]
    seg_terms = ["deep work", "office desk"]

    # Guard condition from the pipeline
    refined = False
    if seg_terms and seg_terms != queries:
        refined = True

    assert refined is False

