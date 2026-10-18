"""The only place that knows the rules.json keys.

Reads config/rules.json once, validates it, and exposes it as a typed, immutable Rules object.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar

from app.csv_columns import COLUMN_TO_FIELD
from app.errors import RulesConfigError

DEFAULT_RULES_PATH = Path(__file__).parent.parent / "config" / "rules.json"
TOP_LEVEL = "<top level>"  # key path for errors about the file as a whole

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ElectricRules:
    net_bonus: int
    method: str


@dataclass(frozen=True, slots=True)
class FamilyTrackRules:
    min_children: int
    child_max_age: int
    cars: tuple[str, ...]
    price_source: str
    gap_split_employee: float


@dataclass(frozen=True, slots=True)
class WaiverRules:
    allowed_by: str
    maintenance_is_taxable: bool
    maintenance_is_pensionable: bool


@dataclass(frozen=True, slots=True)
class TaxBracket:
    upper_limit: int | None  # None = no upper limit (top bracket)
    rate: float


@dataclass(frozen=True, slots=True)
class ContributionRules:
    """Two-tier contribution: low_rate up to threshold, high_rate up to ceiling."""

    threshold: int
    ceiling: int
    low_rate: float
    high_rate: float


@dataclass(frozen=True, slots=True)
class PensionCreditRules:
    rate: float
    max_contribution_rate: float
    income_ceiling: int


@dataclass(frozen=True, slots=True)
class ParticipationRules:
    treatment: str
    reduces_value_of_use: bool


@dataclass(frozen=True, slots=True)
class RoundingRules:
    during: str
    display: str


@dataclass(frozen=True, slots=True)
class NetMethodRules:
    basis: str
    tax_year: int
    credit_point_value: int
    brackets: tuple[TaxBracket, ...]
    national_insurance: ContributionRules
    health: ContributionRules
    pension_credit: PensionCreditRules
    contributions_base: str
    participation: ParticipationRules
    rounding: RoundingRules


@dataclass(frozen=True, slots=True)
class ValueOfUseRules:
    source: str
    key: tuple[str, ...]
    drive_field: Mapping[int, str]


@dataclass(frozen=True, slots=True)
class Rules:
    version: str
    budgets: Mapping[str, int]
    group_order: tuple[str, ...]
    choose_from: str
    electric: ElectricRules
    family_track: FamilyTrackRules
    waiver: WaiverRules
    unused_budget: str
    net_method: NetMethodRules
    value_of_use: ValueOfUseRules


def load_rules(path: Path = DEFAULT_RULES_PATH) -> Rules:
    """Read and parse the rules file; later calls with the same path return the same object."""
    return _load(path.resolve())


@cache
def _load(path: Path) -> Rules:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RulesConfigError(TOP_LEVEL, "an existing rules file", "no such file", file=path) from None
    except UnicodeDecodeError as e:
        raise RulesConfigError(TOP_LEVEL, "UTF-8 text", f"undecodable byte at position {e.start}", file=path) from None
    except json.JSONDecodeError as e:
        raise RulesConfigError(
            TOP_LEVEL, "valid JSON", f"{e.msg} at line {e.lineno}, column {e.colno}", file=path
        ) from None

    try:
        return _parse(_Section(raw, ""))
    except RulesConfigError as e:
        e.file = path
        raise


def _parse(root: "_Section") -> Rules:
    electric = root.section("electric")
    family = root.section("familyTrack")
    waiver = root.section("waiver")
    net = root.section("netMethod")
    pension = net.section("pensionCredit")
    participation = net.section("participation")
    rounding = net.section("rounding")
    value_of_use = root.section("valueOfUse")

    group_order = root.list_of("groupOrder", _str, non_empty=True)
    budgets = root.mapping_of("budgets", _positive_int)
    if any(group not in budgets for group in group_order):
        raise RulesConfigError(
            "budgets",
            f"a budget for every group in groupOrder {_shown(list(group_order))}",
            f"budgets for {_shown(list(budgets))}",
        )

    brackets = net.list_of("brackets", _bracket, non_empty=True)
    _check_bracket_order(brackets, "netMethod.brackets")

    return Rules(
        version=root.get("version", str),
        budgets=budgets,
        group_order=group_order,
        choose_from=root.get("chooseFrom", str),
        electric=ElectricRules(
            net_bonus=electric.parse("netBonus", _positive_int),
            method=electric.get("method", str),
        ),
        family_track=FamilyTrackRules(
            min_children=family.get("minChildren", int),
            child_max_age=family.get("childMaxAge", int),
            cars=family.list_of("cars", _str, non_empty=True),
            price_source=family.get("priceSource", str),
            gap_split_employee=family.parse("gapSplitEmployee", _rate),
        ),
        waiver=WaiverRules(
            allowed_by=waiver.get("allowedBy", str),
            maintenance_is_taxable=waiver.get("maintenanceIsTaxable", bool),
            maintenance_is_pensionable=waiver.get("maintenanceIsPensionable", bool),
        ),
        unused_budget=root.get("unusedBudget", str),
        net_method=NetMethodRules(
            basis=net.get("basis", str),
            tax_year=net.get("taxYear", int),
            credit_point_value=net.parse("creditPointValue", _positive_int),
            brackets=brackets,
            national_insurance=_contribution(net.section("nationalInsurance")),
            health=_contribution(net.section("health")),
            pension_credit=PensionCreditRules(
                rate=pension.parse("rate", _rate),
                max_contribution_rate=pension.parse("maxContributionRate", _rate),
                income_ceiling=pension.get("incomeCeiling", int),
            ),
            contributions_base=net.get("contributionsBase", str),
            participation=ParticipationRules(
                treatment=participation.get("treatment", str),
                reduces_value_of_use=participation.get("reducesValueOfUse", bool),
            ),
            rounding=RoundingRules(
                during=rounding.get("during", str),
                display=rounding.get("display", str),
            ),
        ),
        value_of_use=ValueOfUseRules(
            source=value_of_use.get("source", str),
            key=value_of_use.list_of("key", _csv_column, non_empty=True),
            drive_field=_drive_field(value_of_use.mapping_of("driveField", _str), "valueOfUse.driveField"),
        ),
    )


def _contribution(section: "_Section") -> ContributionRules:
    return ContributionRules(
        threshold=section.get("threshold", int),
        ceiling=section.get("ceiling", int),
        low_rate=section.parse("lowRate", _rate),
        high_rate=section.parse("highRate", _rate),
    )


def _bracket(value: Any, where: str) -> TaxBracket:
    if not isinstance(value, list) or len(value) != 2:
        raise RulesConfigError(where, "[upper limit or null, rate]", _shown(value))
    upper_limit, rate = value
    return TaxBracket(
        upper_limit=None if upper_limit is None else _check(upper_limit, f"{where}[0]", int),
        rate=_rate(rate, f"{where}[1]"),
    )


def _check_bracket_order(brackets: tuple[TaxBracket, ...], where: str) -> None:
    """Upper limits must strictly increase; only the last bracket may be open-ended (null)."""
    for i, bracket in enumerate(brackets):
        if bracket.upper_limit is None:
            if i != len(brackets) - 1:
                raise RulesConfigError(f"{where}[{i}][0]", "a limit (only the last bracket may be null)", "null")
        elif i > 0 and bracket.upper_limit <= (previous := brackets[i - 1].upper_limit):
            raise RulesConfigError(
                f"{where}[{i}][0]", f"a limit above the previous one ({previous})", _shown(bracket.upper_limit)
            )


def _str(value: Any, where: str) -> str:
    return _check(value, where, str)


def _positive_int(value: Any, where: str) -> int:
    number = _check(value, where, int)
    if number <= 0:
        raise RulesConfigError(where, "an integer > 0", _shown(value))
    return number


def _rate(value: Any, where: str) -> float:
    number = _check(value, where, float)
    if not 0 <= number <= 1:
        raise RulesConfigError(where, "rate 0..1", _shown(value))
    return number


def _csv_column(value: Any, where: str) -> str:
    column = _check(value, where, str)
    if column not in COLUMN_TO_FIELD:
        raise RulesConfigError(where, "a column name known to csv_columns.COLUMN_TO_FIELD", _shown(column))
    return column


def _drive_field(names_by_code: Mapping[str, str], where: str) -> Mapping[int, str]:
    """Code -> drive name; names must be unique so the mapping can be read in both directions."""
    drive_field: dict[int, str] = {}
    for code, name in names_by_code.items():
        if not code.isdigit():
            raise RulesConfigError(f"{where}.{code}", "an integer code as key", _shown(code))
        if name in drive_field.values():
            raise RulesConfigError(f"{where}.{code}", "a drive name not used by another code", _shown(name))
        drive_field[int(code)] = name
    return MappingProxyType(drive_field)


def _shown(value: Any, limit: int = 80) -> str:
    """The value as it appears in JSON, shortened so a whole section doesn't flood the message."""
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."


_TYPE_NAMES = {
    int: "an integer",
    float: "a number",
    str: "a string",
    bool: "true or false",
    list: "a list",
    dict: "an object",
}


def _check(value: Any, where: str, kind: type) -> Any:
    """Return value if it is of the expected JSON type (ints are accepted as numbers)."""
    is_bool = isinstance(value, bool)
    if kind is float and isinstance(value, int) and not is_bool:
        return float(value)
    if not isinstance(value, kind) or (is_bool and kind is not bool):
        raise RulesConfigError(where, _TYPE_NAMES[kind], _shown(value))
    return value


class _Section:
    """A JSON object plus its dotted path, so errors can name the exact key."""

    def __init__(self, data: Any, path: str) -> None:
        self._data: dict[str, Any] = _check(data, path or TOP_LEVEL, dict)
        self._path = path

    def parse(self, key: str, parse: Callable[[Any, str], T]) -> T:
        where = self._where(key)
        if key not in self._data:
            raise RulesConfigError(where, "a required key", "nothing (key is missing)")
        return parse(self._data[key], where)

    def get(self, key: str, kind: type) -> Any:
        return self.parse(key, lambda value, where: _check(value, where, kind))

    def section(self, key: str) -> "_Section":
        return _Section(self.get(key, dict), self._where(key))

    def list_of(self, key: str, parse: Callable[[Any, str], T], *, non_empty: bool = False) -> tuple[T, ...]:
        where = self._where(key)
        values = self.get(key, list)
        if non_empty and not values:
            raise RulesConfigError(where, "a non-empty list", "[]")
        return tuple(parse(value, f"{where}[{i}]") for i, value in enumerate(values))

    def mapping_of(self, key: str, parse: Callable[[Any, str], T]) -> Mapping[str, T]:
        section = self.section(key)
        return MappingProxyType({name: section.parse(name, parse) for name in section._data})

    def _where(self, key: str) -> str:
        return f"{self._path}.{key}" if self._path else key
