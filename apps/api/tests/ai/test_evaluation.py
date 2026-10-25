import pytest

from app.ai.evaluation import (
    abstained,
    args_match,
    key_facts_present,
    mean_scores,
    retrieval_scores,
    score_tool_case,
)

GOLD = {("sid_x", 7), ("sid_x", 9)}


def test_retrieval_scores_by_rank() -> None:
    assert retrieval_scores([("sid_x", 7)], GOLD) == {
        "hit@1": 1,
        "hit@3": 1,
        "hit@5": 1,
        "mrr@10": 1,
    }
    third = [("a", 1), ("b", 2), ("sid_x", 9)]
    assert retrieval_scores(third, GOLD) == {"hit@1": 0, "hit@3": 1, "hit@5": 1, "mrr@10": 1 / 3}
    miss = [("a", i) for i in range(10)] + [("sid_x", 7)]  # rank 11: outside MRR@10
    assert retrieval_scores(miss, GOLD) == {"hit@1": 0, "hit@3": 0, "hit@5": 0, "mrr@10": 0}
    assert mean_scores([{"x": 1.0}, {"x": 0.0}]) == {"x": 0.5}


def test_key_facts_numbers_and_text() -> None:
    phrase = ["2.00 % if the investment is redeemed on or before 365 days"]
    assert key_facts_present("A 2% exit load applies within 365 days.", phrase)
    assert not key_facts_present("A 1% exit load applies within 365 days.", phrase)
    assert key_facts_present("Benchmark: the NIFTY 500 TRI.", ["Nifty 500 TRI"])
    assert not key_facts_present("Benchmark: Nifty 50.", ["Nifty Midcap 150 Index"])
    assert key_facts_present("Minimum Rs 1,000.", ["Rs. 1,000/- and in multiples of Re. 1/-"])


def test_abstention() -> None:
    assert abstained("Not found in the documents.")
    assert not abstained("The exit load is 1%.")


def test_args_match_paths_and_tolerance() -> None:
    args = {"factor_shocks": {"MKT_RF": -0.2}, "direct_shocks": [{"target": "Financial Services"}]}
    assert args_match(args, {"factor_shocks.MKT_RF": -0.2})
    assert args_match(args, {"direct_shocks.0.target": "financial services"})
    assert not args_match(args, {"factor_shocks.MKT_RF": -20})
    assert not args_match(args, {"direct_shocks.1.target": "x"})
    assert args_match({"monthly_sip": 20000.0}, {"monthly_sip": 20000})


@pytest.mark.parametrize(
    ("calls", "tool", "args", "expected"),
    [
        (
            [("get_risk_summary", {"confidence": 0.99})],
            "get_risk_summary",
            {"confidence": 0.99},
            (1, 1),
        ),
        ([("get_risk_summary", {})], "get_risk_summary", {"confidence": 0.99}, (1, 0)),
        (
            [
                ("list_stress_scenarios", {}),
                ("run_stress_scenario", {"scenario_id": "credit_event"}),
            ],
            "run_stress_scenario",
            {"scenario_id": "credit_event"},
            (1, 1),
        ),
        ([("get_factor_exposures", {})], "get_risk_summary", {}, (0, 0)),
        ([], None, {}, (1, 1)),
        ([("get_risk_summary", {})], None, {}, (0, 0)),
    ],
)
def test_tool_case_scoring(calls, tool, args, expected) -> None:
    s = score_tool_case(calls, tool, args)
    assert (s["tool_correct"], s["args_correct"]) == expected


def test_key_facts_tolerate_unicode_hyphens_and_spaces() -> None:
    # Real model output (2026-10-25): non-breaking hyphen U+2011 and narrow no-break space U+202F.
    answer = "growth from an actively managed portfolio primarily of equity‑related securities"
    assert key_facts_present(
        answer, ["actively managed portfolio primarily of Equity Related Securities"]
    )
    assert key_facts_present("Benchmark: Nifty 500 TRI.", ["Nifty 500 TRI"])


def test_lenient_key_facts_accept_paraphrases_but_not_wrong_facts() -> None:
    from app.ai.evaluation import key_facts_lenient

    assert key_facts_lenient("It invests 65 % to 100 % in mid caps.", ["mid-cap stocks (65-100 %)"])
    assert not key_facts_present(
        "It invests 65 % to 100 % in mid caps.", ["mid-cap stocks (65-100 %)"]
    )
    assert key_facts_lenient(
        "The scheme does not undertake swing pricing.",
        ["The Scheme does not undertake swing pricing"],
    )
    assert (
        key_facts_lenient("Exit load: nil from Day 7.", ["Day 7 onwards NIL"]) is False
    )  # "onwards"
    assert not key_facts_lenient(
        "Minimum SIP is Rs 500.", ["Minimum Investment Amount : Rs. 5000/-"]
    )
    assert not key_facts_lenient("Managed by HDFC AMC.", ["Chirag Setalvad"])


def test_ambiguous_case_accepts_either_valid_tool() -> None:
    alt = [
        {"tool": "run_custom_stress", "args": {"direct_shocks.0.target": "Information Technology"}}
    ]
    expected = ("run_stress_scenario", {"scenario_id": "it_sector_shock"})
    custom = [("run_custom_stress", {"direct_shocks": [{"target": "Information Technology"}]})]
    builtin = [("run_stress_scenario", {"scenario_id": "it_sector_shock"})]
    wrong_target = [("run_custom_stress", {"direct_shocks": [{"target": "IT"}]})]
    assert score_tool_case(custom, *expected, alt) == {"tool_correct": 1.0, "args_correct": 1.0}
    assert score_tool_case(builtin, *expected, alt) == {"tool_correct": 1.0, "args_correct": 1.0}
    assert score_tool_case(wrong_target, *expected, alt) == {
        "tool_correct": 1.0,
        "args_correct": 0.0,
    }
    assert score_tool_case(custom, *expected) == {"tool_correct": 0.0, "args_correct": 0.0}
