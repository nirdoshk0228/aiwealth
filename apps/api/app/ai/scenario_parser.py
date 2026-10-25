"""Natural language -> HypotheticalScenarioSpec, validated, with one retry.

    1. Ask the model (JSON mode) for a scenario object, giving it the allowed factor names,
       fund categories and sectors of the universe, and the unit rule (decimals).
    2. Validate with HypotheticalScenarioSpec (Pydantic: closed enums, shocks in (-1, 1], no
       extra fields) and check_against_universe (sectors/categories/assets exist).
    3. On failure, send the validation errors back once and ask for a corrected object.
    4. Still invalid -> ScenarioParseError (the caller shows the reason; nothing is half-applied).

The model only translates words into shocks; the P&L is then computed by the engine.
"""

import json
from dataclasses import dataclass

from pydantic import ValidationError

from app.ai.llm.base import LLMClient, Message
from app.ai.trace import Trace
from app.data.scenarios import HypotheticalScenarioSpec, check_against_universe
from app.data.universe import Universe

MAX_ATTEMPTS = 2  # first try + one retry


class ScenarioParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedScenario:
    spec: HypotheticalScenarioSpec
    attempts: int


def _instructions(universe: Universe) -> str:
    categories = sorted({f.category for f in universe.funds})
    sectors = sorted({s.industry for s in universe.stocks})
    return f"""Convert the user's description of a market scenario into ONE JSON object:
{{"id": "custom", "type": "hypothetical", "name": "<short name>",
  "description": "<one sentence restating the scenario>",
  "rationale": "<why these shocks represent the description>",
  "factor_shocks": {{"MKT_RF": <decimal>, ...}},
  "direct_shocks": [
    {{"target": {{"kind": "sector", "sector": "<sector>"}}, "shock": <decimal>}},
    {{"target": {{"kind": "category", "category": "<category>"}}, "shock": <decimal>}}]}}
Rules:
- Shocks are simple returns as DECIMALS: "falls 20%" -> -0.2, "rises 10%" -> 0.1. Range (-1, 1].
- factor_shocks keys only: MKT_RF (broad equity market), SMB (small minus big), HML (value
  minus growth), WML (momentum). A general "market fall" is MKT_RF.
- Sectors (exact spelling): {", ".join(sectors)}.
- Fund categories (exact spelling): {", ".join(categories)}.
- Use only shocks the user stated; do not add others. At least one shock. No other keys.
- Output the JSON object only."""


def parse_scenario(
    client: LLMClient, text: str, universe: Universe, *, trace: Trace | None = None
) -> ParsedScenario:
    trace = trace or Trace()
    messages = [Message("system", _instructions(universe)), Message("user", text)]
    error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        res = client.chat(messages, json_mode=True)
        trace.add_usage(res.usage.prompt_tokens, res.usage.completion_tokens)
        raw = res.message.content or ""
        trace.record("scenario_parse", attempt=attempt, raw=raw)
        try:
            spec = HypotheticalScenarioSpec.model_validate(json.loads(raw))
            check_against_universe(spec, universe)
            return ParsedScenario(spec=spec, attempts=attempt)
        except json.JSONDecodeError:
            error = "the reply was not valid JSON"
        except ValidationError as exc:
            error = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:6]
            )
        except ValueError as exc:  # check_against_universe
            error = str(exc)
        trace.record("scenario_parse_invalid", attempt=attempt, error=error)
        messages += [
            Message("assistant", raw),
            Message(
                "user", f"That object is invalid: {error}. Reply with a corrected JSON object only."
            ),
        ]
    raise ScenarioParseError(f"could not turn the description into a valid scenario: {error}")
