"""Natural-language scenario parser: validation, one retry, clear failure."""

import json

import pytest

from app.ai.llm.fake import FakeLLMClient, reply
from app.ai.scenario_parser import ScenarioParseError, parse_scenario
from app.data.universe import load_universe

UNIVERSE = load_universe()


def obj(**over: object) -> str:
    base = {
        "id": "custom",
        "type": "hypothetical",
        "name": "IT and market fall",
        "description": "IT stocks fall 15% and the market falls 5%.",
        "rationale": "As described by the user.",
        "factor_shocks": {"MKT_RF": -0.05},
        "direct_shocks": [
            {"target": {"kind": "sector", "sector": "Information Technology"}, "shock": -0.15}
        ],
    }
    return json.dumps(base | over)


def test_valid_first_time() -> None:
    fake = FakeLLMClient([reply(obj())])
    res = parse_scenario(fake, "IT stocks -15%, market -5%", UNIVERSE)
    assert res.attempts == 1
    assert res.spec.factor_shocks == {"MKT_RF": -0.05}
    assert fake.requests[0]["json_mode"] is True
    system = fake.requests[0]["messages"][0].content
    assert "Information Technology" in system and "debt_gilt" in system  # universe vocabulary


def test_percent_mixup_is_retried_with_the_error() -> None:
    fake = FakeLLMClient([reply(obj(factor_shocks={"MKT_RF": -5})), reply(obj())])
    res = parse_scenario(fake, "market -5%", UNIVERSE)
    assert res.attempts == 2
    retry_msg = fake.requests[1]["messages"][-1].content
    assert retry_msg.startswith("That object is invalid: factor_shocks.MKT_RF")


def test_unknown_sector_is_retried() -> None:
    bad = obj(direct_shocks=[{"target": {"kind": "sector", "sector": "IT"}, "shock": -0.15}])
    fake = FakeLLMClient([reply(bad), reply(obj())])
    assert parse_scenario(fake, "IT -15%", UNIVERSE).attempts == 2
    assert "not in the universe" in fake.requests[1]["messages"][-1].content


def test_two_failures_raise() -> None:
    fake = FakeLLMClient([reply("not json"), reply(obj(factor_shocks={"GOLD": 0.1}))])
    with pytest.raises(ScenarioParseError, match="factor_shocks.GOLD"):
        parse_scenario(fake, "gold up", UNIVERSE)
    assert len(fake.requests) == 2  # exactly one retry


def test_extra_fields_are_rejected() -> None:
    fake = FakeLLMClient([reply(obj(probability=0.3)), reply(obj(probability=0.3))])
    with pytest.raises(ScenarioParseError, match="Extra inputs"):
        parse_scenario(fake, "x", UNIVERSE)
