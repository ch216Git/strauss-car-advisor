"""The agent's tools as the LLM sees them: specs, argument validation, guards and JSON results.

Everything here runs in Python before and after the existing tools (app.agent.tools), so the
critical rules hold whatever the model does:
- only the six known tools, with exactly their declared arguments and types;
- employee facts (salary, percentages, credit points, occupation, children, participation, waiver)
  must appear in the employee's own messages (app.agent.grounding); a well-typed value is not enough;
- a car can be used only when the employee's messages name it: identified by find_car and named by
  the employee, or chosen from shown candidates/alternatives by the employee's next message;
- results are already rounded for display, and estimates are labelled in the data.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from typing import Any

from app.agent.grounding import employee_texts, number_grounded, occupation_grounded, waiver_grounded
from app.agent.llm import ToolCall, ToolSpec
from app.agent.render import electric_benefit
from app.agent.state import ConversationState
from app.agent.tools import (
    Alternatives,
    CarMatches,
    ComparisonSummary,
    ToolContext,
    ToolError,
    ToolOutcome,
    ask_representative,
    compare_alternative,
    find_car,
    list_alternatives,
    save_employee_details,
    set_current_car,
)
from app.car_search import MatchStatus
from app.comparison import CompanyCar, SituationDetails, Waiver, display_shekels
from app.rules import Rules
from app.salary import NetBreakdown

# --- Tool specs ----------------------------------------------------------------------

_NUMBER = {"type": "number", "minimum": 0}
# find_car purposes; only "alternative" continues, to the existing compare_alternative(car=query) path.
FIND_CAR_PURPOSES = ("alternative", "current", "information")
_CAR_WORDS = {"type": "string", "description": "One car exactly as the employee wrote it, Hebrew or English"}

SPECS = (
    ToolSpec(
        "save_employee_details",
        "Save details the employee stated. Pass only what they said; omit anything not said. "
        "Percentages as numbers (6 means 6%). Returns what is still missing.",
        {
            "type": "object",
            "properties": {
                "occupation": {"type": "string", "description": "The occupation as the employee wrote it"},
                "base_salary": _NUMBER,
                "commissions": {**_NUMBER, "description": "0 if the employee said there are none"},
                "bonus": {**_NUMBER, "description": "Monthly bonus; 0 if none"},
                "pension_percent": _NUMBER,
                "training_fund_percent": {**_NUMBER, "description": "קרן השתלמות; 0 if none"},
                "credit_points": _NUMBER,
                "eligible_children": {
                    "type": "integer", "minimum": 0,
                    "description": "Children up to the family-track age limit; only for a family-track car",
                },
            },
        },
    ),
    ToolSpec(
        "find_car",
        "Look up ONE catalogue car from the employee's own words, in Hebrew or English, as written. purpose says "
        "why, and is required: 'alternative' runs the whole comparison for that car in this same call (exactly as "
        "compare_alternative with car) - do not look the car up first and compare in another call. "
        "'current' and 'information' only return the matching cars.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "purpose": {
                    "type": "string",
                    "enum": list(FIND_CAR_PURPOSES),
                    "description": (
                        "alternative: the employee asks about getting, switching to or the net-pay effect of this car. "
                        "current: this is the employee's car today (then set it with set_current_car). "
                        "information: the employee only asks what this car is, without choosing it. "
                        "If unsure, use information."
                    ),
                },
            },
            "required": ["query", "purpose"],
        },
    ),
    ToolSpec(
        "set_current_car",
        "Record today's situation: the car (the employee's own words, or a car_id) with the participation the "
        "employee stated, or waiver=true. An unclear car is not set: the candidates come back to ask about.",
        {
            "type": "object",
            "properties": {
                "car": _CAR_WORDS, "car_id": {"type": "string"}, "participation": _NUMBER, "waiver": {"type": "boolean"},
            },
        },
    ),
    ToolSpec(
        "list_alternatives",
        "Cars and waiver option the employee may choose. Participation shown is a catalogue default (estimate).",
        {"type": "object", "properties": {}},
    ),
    ToolSpec(
        "compare_alternative",
        "Calculate net pay today vs the alternative: the car (the employee's own words, or a car_id) or "
        "waiver=true. Pass participation only if the employee or the representative stated it; otherwise the "
        "result is an estimate. An unclear car is not compared: the candidates come back to ask about.",
        {
            "type": "object",
            "properties": {
                "car": _CAR_WORDS, "car_id": {"type": "string"}, "waiver": {"type": "boolean"}, "participation": _NUMBER,
            },
        },
    ),
    ToolSpec(
        "ask_representative",
        "Send a question to the transport representative when information is missing or unknown.",
        {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "enum": ["alternative_participation", "current_car_not_in_catalogue", "other"]},
                "question": {"type": "string"},
                "car_id": {"type": "string", "description": "The alternative car, for alternative_participation"},
            },
            "required": ["topic", "question"],
        },
    ),
)

_RUN: Mapping[str, Callable[..., ToolOutcome]] = {
    "save_employee_details": save_employee_details,
    "find_car": find_car,
    "set_current_car": set_current_car,
    "list_alternatives": list_alternatives,
    "compare_alternative": compare_alternative,
    "ask_representative": ask_representative,
}
_SPEC_BY_NAME = {spec.name: spec for spec in SPECS}
assert set(_RUN) == set(_SPEC_BY_NAME)

# Tools whose car_id must pass the selection guard.
_GUARDED = {"set_current_car", "compare_alternative", "ask_representative"}


@dataclass(frozen=True, slots=True)
class Execution:
    state: ConversationState
    content: dict[str, Any]  # JSON-compatible ToolResult content
    comparison: ComparisonSummary | None = None  # set when a comparison was calculated
    tool: str = ""  # the tool that actually ran: compare_alternative for find_car(purpose="alternative")
    error: ToolError | None = None  # the structured error behind content, when the tool itself refused


def execute(call: ToolCall, state: ConversationState, ctx: ToolContext) -> Execution:
    """Validate and run one tool call; failures come back as {"ok": false, "error": ...}.

    find_car with purpose "alternative" runs as compare_alternative(car=query): the model stated the
    intent, and the comparison goes through the very same path (car guard, grounding, eligibility gate).
    The result still answers the find_car call (same name and call ID); Execution.tool says what ran.
    """
    spec = _SPEC_BY_NAME.get(call.name)
    if spec is None:
        return Execution(
            state, _failure("unknown_tool", f"no tool named {call.name!r}; tools: {list(_SPEC_BY_NAME)}"), tool=call.name
        )
    try:
        arguments = validate_arguments(spec.parameters, call.arguments)
    except ArgumentError as e:
        return Execution(state, _failure("invalid_arguments", str(e)), tool=call.name)

    tool = call.name
    if tool == "find_car":
        if arguments.pop("purpose") == "alternative":
            tool, arguments = "compare_alternative", {"car": arguments["query"]}
    execution = replace(_run(tool, arguments, state, ctx), tool=tool)
    if tool != call.name:
        execution = replace(execution, content={**execution.content, "ran": tool})
    return execution


def _run(tool: str, arguments: dict[str, Any], state: ConversationState, ctx: ToolContext) -> Execution:
    if "car" in arguments:
        resolved = _resolve_car_words(tool, arguments, state, ctx)
        if isinstance(resolved, Execution):
            return resolved
        state, arguments = resolved

    if tool in _GUARDED and arguments.get("car_id") is not None:
        car_id = _catalogue_id(arguments["car_id"], ctx)
        if car_id not in state.usable_car_ids:
            return Execution(
                state,
                _failure(
                    "car_not_confirmed",
                    f"{arguments['car_id']!r} is not a car the employee named; identify it with find_car using "
                    "the employee's words, and if there are several candidates ask the employee to name one",
                ),
            )
        arguments["car_id"] = car_id

    ungrounded = _ungrounded_facts(tool, arguments, employee_texts(state))
    if ungrounded:
        return Execution(
            state,
            _failure(
                "not_stated_by_employee",
                f"{ungrounded} do not appear in the employee's messages; ask the employee instead of assuming",
            ),
        )

    outcome = _RUN[tool](state, ctx, **arguments)
    new_state = _track_cars(outcome.state, outcome.result, ctx)
    if isinstance(outcome.result, ToolError):
        return Execution(new_state, _error_content(outcome.result, ctx.rules), error=outcome.result)
    comparison = outcome.result if isinstance(outcome.result, ComparisonSummary) else None
    return Execution(new_state, {"ok": True, **to_json(outcome.result, ctx.rules)}, comparison)


def _resolve_car_words(
    tool: str, arguments: dict[str, Any], state: ConversationState, ctx: ToolContext
) -> tuple[ConversationState, dict[str, Any]] | Execution:
    """car (employee's words) -> car_id, through find_car and the same guard as a find_car result.

    Only an identified car that the employee's own messages name is used; candidates are recorded
    as offered (so the employee's next message can choose) and returned instead of being used.
    """
    if "car_id" in arguments or arguments.get("waiver"):
        return Execution(state, _failure("invalid_arguments", "give one of car, car_id or waiver"))
    outcome = _RUN["find_car"](state, ctx, query=arguments["car"])
    state = _track_cars(outcome.state, outcome.result, ctx)
    if isinstance(outcome.result, ToolError):
        return Execution(state, _error_content(outcome.result, ctx.rules))
    matches = outcome.result
    if matches.status == MatchStatus.IDENTIFIED and matches.cars[0].car_id in state.usable_car_ids:
        rest = {name: value for name, value in arguments.items() if name != "car"}
        return state, {**rest, "car_id": matches.cars[0].car_id}

    if matches.status == MatchStatus.CANDIDATES:
        code, detail = "car_needs_choice", "several cars match; show these to the employee and ask which one"
    elif matches.status == MatchStatus.NOT_FOUND:
        code, detail = "car_not_found", "not in the catalogue; it cannot be calculated - offer the representative"
    else:
        code, detail = "car_not_confirmed", "the employee's own words do not name this car; ask the employee"
    content = _failure(code, f"{tool} was not run: {detail}")
    content["error"]["matches"] = to_json(matches, ctx.rules)
    return Execution(state, content)


def start_turn(state: ConversationState, user_text: str, ctx: ToolContext) -> ConversationState:
    """The employee replied: a shown candidate/alternative becomes usable only if the reply names exactly it."""
    if not state.offered_car_ids:
        return state
    chosen = ctx.car_search.mentioned(user_text, among=state.offered_car_ids)
    if len(chosen) != 1:
        return state
    return replace(state, usable_car_ids=state.usable_car_ids | chosen, offered_car_ids=state.offered_car_ids - chosen)


# --- Fact provenance ----------------------------------------------------------------------

# Numeric employee facts per tool; each must be a number the employee wrote.
_NUMERIC_FACTS = {
    "save_employee_details": (
        "base_salary", "commissions", "bonus", "pension_percent", "training_fund_percent", "credit_points",
        "eligible_children",
    ),
    "set_current_car": ("participation",),
    "compare_alternative": ("participation",),
}


def _ungrounded_facts(tool: str, arguments: Mapping[str, Any], texts: tuple[str, ...]) -> list[str]:
    ungrounded = [
        name for name in _NUMERIC_FACTS.get(tool, ()) if name in arguments and not number_grounded(arguments[name], texts)
    ]
    if tool == "save_employee_details" and "occupation" in arguments and not occupation_grounded(arguments["occupation"], texts):
        ungrounded.append("occupation")
    if tool == "set_current_car" and arguments.get("waiver") and not waiver_grounded(texts):
        ungrounded.append("waiver")
    return ungrounded


# --- Argument validation ------------------------------------------------------------------

class ArgumentError(ValueError):
    pass


def validate_arguments(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Check arguments against the tool's (flat) schema; null means "not given" and is dropped."""
    properties = schema.get("properties", {})
    unknown = sorted(set(arguments) - set(properties))
    if unknown:
        raise ArgumentError(f"unknown arguments {unknown}; allowed: {sorted(properties)}")
    cleaned = {name: _checked(name, value, properties[name]) for name, value in arguments.items() if value is not None}
    missing = [name for name in schema.get("required", []) if name not in cleaned]
    if missing:
        raise ArgumentError(f"missing required arguments {missing}")
    return cleaned


def _checked(name: str, value: Any, prop: Mapping[str, Any]) -> Any:
    kind = prop["type"]
    if kind == "string":
        ok = isinstance(value, str)
    elif kind == "boolean":
        ok = isinstance(value, bool)
    elif kind == "integer":
        if isinstance(value, float) and value.is_integer():
            value = int(value)  # JSON has one number type; 5.0 children is 5
        ok = isinstance(value, int) and not isinstance(value, bool)
    else:  # number
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not ok:
        raise ArgumentError(f"{name} must be a {kind}, got {value!r}")
    if "minimum" in prop and value < prop["minimum"]:
        raise ArgumentError(f"{name} must be >= {prop['minimum']}, got {value!r}")
    if "enum" in prop and value not in prop["enum"]:
        raise ArgumentError(f"{name} must be one of {list(prop['enum'])}, got {value!r}")
    return value


# --- Car selection guard -------------------------------------------------------------------

def _catalogue_id(car_id: str, ctx: ToolContext) -> str:
    return car_id if car_id in ctx.catalog.cars else (ctx.car_search.car_id(car_id) or car_id)


def _track_cars(state: ConversationState, result: object, ctx: ToolContext) -> ConversationState:
    if isinstance(result, CarMatches):
        ids = frozenset(car.car_id for car in result.cars)
        if result.status == MatchStatus.IDENTIFIED:
            # The LLM wrote the query; the car counts only if the employee's own words name it.
            named = frozenset().union(*(ctx.car_search.mentioned(text) for text in employee_texts(state)))
            return replace(state, usable_car_ids=state.usable_car_ids | (ids & named))
        return replace(state, offered_car_ids=state.offered_car_ids | ids)
    if isinstance(result, Alternatives):
        return replace(state, offered_car_ids=state.offered_car_ids | {option.car.car_id for option in result.cars})
    return state


# --- JSON results ------------------------------------------------------------------------

def _failure(code: str, detail: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "detail": detail}}


def _error_content(error: ToolError, rules: Rules) -> dict[str, Any]:
    content = _failure(error.code.value, error.detail)
    for name in ("missing", "question_ids"):
        if getattr(error, name):
            content["error"][name] = list(getattr(error, name))
    if error.reason:
        content["error"]["reason"] = error.reason
    if error.stage:
        content["error"]["stage"] = error.stage
    if error.facts:
        content["error"]["facts"] = dict(error.facts)
    if "eligible_children" in error.missing:
        content["error"]["family_track_rule"] = _family_track_rule(rules)
    return content


# Hebrew labels for the comparison's change items, so the model reads meanings rather than field names.
CHANGE_LABELS = {
    "pensionable_income": "הכנסה פנסיונית (בסיס, עמלות ובונוס)",
    "maintenance": "אחזקת רכב (חייבת במס)",
    "electric_gross": "החזר חשמל (מגולם): הרכיב בברוטו; העובד מקבל electric_net_bonus נטו",
    "gross_income": "סך הברוטו המשולם",
    "use_value": "שווי שימוש",
    "taxable_income": "הכנסה חייבת במס",
    "tax_before_credits": "מס לפני זיכויים",
    "credit_points_credit": "זיכוי נקודות זיכוי",
    "pension_credit": "זיכוי פנסיה",
    "income_tax": "מס הכנסה",
    "national_insurance": "ביטוח לאומי",
    "health": "דמי בריאות",
    "pension": "הפרשת פנסיה של העובד",
    "training_fund": "קרן השתלמות של העובד",
    "participation": "השתתפות העובד ברכב",
    "family_share": "חלק העובד בפער במסלול המשפחתי",
    "net": "נטו",
}
assert set(CHANGE_LABELS) == {f.name for f in fields(NetBreakdown)}


def to_json(value: Any, rules: Rules) -> Any:
    """Tool results as JSON-compatible data; comparisons are reduced to displayed whole shekels."""
    if isinstance(value, ComparisonSummary):
        return _comparison_json(value, rules)
    if isinstance(value, Alternatives):
        content = {f.name: to_json(getattr(value, f.name), rules) for f in fields(value)}
        if value.family_track_needs_children:
            content["family_track_rule"] = _family_track_rule(rules)
        return content
    if isinstance(value, (CompanyCar, Waiver)):
        return _situation_json(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: to_json(getattr(value, f.name), rules) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_json(item, rules) for key, item in value.items()}
    if isinstance(value, (list, tuple, frozenset, set)):
        return [to_json(item, rules) for item in value]
    return value


def _family_track_rule(rules: Rules) -> dict[str, int]:
    return {"min_children": rules.family_track.min_children, "child_max_age": rules.family_track.child_max_age}


def _situation_json(situation: CompanyCar | Waiver) -> dict[str, Any]:
    if isinstance(situation, Waiver):
        return {"kind": "waiver"}
    return {"kind": "company_car", "car_id": situation.car_id, "participation": situation.participation}


def _comparison_json(summary: ComparisonSummary, rules: Rules) -> dict[str, Any]:
    result = summary.result
    changes = []
    for change in result.changes:
        current, alternative = display_shekels(change.current), display_shekels(change.alternative)
        if current != alternative:
            changes.append({
                "item": change.item, "label": CHANGE_LABELS[change.item],
                "current": current, "alternative": alternative, "difference": alternative - current,
            })
    return {
        "status": summary.status.value,
        "estimated_fields": list(summary.estimated_fields),
        "eligibility_basis": result.eligibility_basis,
        "current": {"net": summary.current_net, **_details_json(result.current_situation, rules)},
        "alternative": {"net": summary.alternative_net, **_details_json(result.alternative_situation, rules)},
        "net_difference": summary.net_difference,
        "changes": changes,
    }


def _details_json(details: SituationDetails, rules: Rules) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": details.kind}
    if details.kind == "waiver":
        out["maintenance"] = details.maintenance
        return out
    out.update(
        car_id=details.car_id,
        commercial_name=details.commercial_name,
        drive=details.drive,
        use_value=details.use_value,
        participation=display_shekels(details.participation),
        participation_is_estimate=details.participation_source == "catalogue_default",
        electric=details.electric,
    )
    if details.electric:
        out["electric_net_bonus"] = rules.electric.net_bonus  # what the employee receives net (rules.json)
        out["electric_benefit"] = electric_benefit(rules)  # the official wording, to quote as written
    if details.family_track:
        out["family_track_employee_share"] = display_shekels(details.family_track.employee_share)
    return out
