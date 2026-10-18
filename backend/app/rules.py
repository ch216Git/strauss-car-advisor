"""The only place that knows the rules.json keys.

Reads config/rules.json once and exposes it as a typed, immutable Rules object.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar

from app.errors import RulesConfigError

DEFAULT_RULES_PATH = Path(__file__).parent.parent / "config" / "rules.json"

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
        raise RulesConfigError(f"Rules file not found: {path}") from None
    except json.JSONDecodeError as e:
        raise RulesConfigError(f"Rules file {path} is not valid JSON: {e}") from None

    try:
        return _parse(_Section(raw, ""))
    except RulesConfigError as e:
        raise RulesConfigError(f"Rules file {path}: {e}") from None


def _parse(root: "_Section") -> Rules:
    electric = root.section("electric")
    family = root.section("familyTrack")
    waiver = root.section("waiver")
    net = root.section("netMethod")
    pension = net.section("pensionCredit")
    participation = net.section("participation")
    rounding = net.section("rounding")
    value_of_use = root.section("valueOfUse")

    return Rules(
        version=root.get("version", str),
        budgets=root.mapping_of("budgets", int),
        group_order=root.list_of("groupOrder", _str),
        choose_from=root.get("chooseFrom", str),
        electric=ElectricRules(
            net_bonus=electric.get("netBonus", int),
            method=electric.get("method", str),
        ),
        family_track=FamilyTrackRules(
            min_children=family.get("minChildren", int),
            child_max_age=family.get("childMaxAge", int),
            cars=family.list_of("cars", _str),
            price_source=family.get("priceSource", str),
            gap_split_employee=family.get("gapSplitEmployee", float),
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
            credit_point_value=net.get("creditPointValue", int),
            brackets=net.list_of("brackets", _bracket),
            national_insurance=_contribution(net.section("nationalInsurance")),
            health=_contribution(net.section("health")),
            pension_credit=PensionCreditRules(
                rate=pension.get("rate", float),
                max_contribution_rate=pension.get("maxContributionRate", float),
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
            key=value_of_use.list_of("key", _str),
            drive_field=MappingProxyType({
                _int_code(code, "valueOfUse.driveField"): name
                for code, name in value_of_use.mapping_of("driveField", str).items()
            }),
        ),
    )


def _contribution(section: "_Section") -> ContributionRules:
    return ContributionRules(
        threshold=section.get("threshold", int),
        ceiling=section.get("ceiling", int),
        low_rate=section.get("lowRate", float),
        high_rate=section.get("highRate", float),
    )


def _bracket(value: Any, where: str) -> TaxBracket:
    if not isinstance(value, list) or len(value) != 2:
        raise RulesConfigError(f"'{where}' must be [upper limit or null, rate], got {value!r}")
    upper_limit, rate = value
    return TaxBracket(
        upper_limit=None if upper_limit is None else _check(upper_limit, f"{where}[0]", int),
        rate=_check(rate, f"{where}[1]", float),
    )


def _str(value: Any, where: str) -> str:
    return _check(value, where, str)


def _int_code(key: str, where: str) -> int:
    if not key.isdigit():
        raise RulesConfigError(f"'{where}' keys must be integer codes, got {key!r}")
    return int(key)


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
        raise RulesConfigError(f"'{where}' must be {_TYPE_NAMES[kind]}, got {value!r}")
    return value


class _Section:
    """A JSON object plus its dotted path, so errors can name the exact key."""

    def __init__(self, data: Any, path: str) -> None:
        self._data: dict[str, Any] = _check(data, path or "top level", dict)
        self._path = path

    def get(self, key: str, kind: type) -> Any:
        where = self._where(key)
        if key not in self._data:
            raise RulesConfigError(f"missing required key '{where}'")
        return _check(self._data[key], where, kind)

    def section(self, key: str) -> "_Section":
        return _Section(self.get(key, dict), self._where(key))

    def list_of(self, key: str, parse: Callable[[Any, str], T]) -> tuple[T, ...]:
        where = self._where(key)
        return tuple(parse(value, f"{where}[{i}]") for i, value in enumerate(self.get(key, list)))

    def mapping_of(self, key: str, kind: type) -> Mapping[str, Any]:
        section = self.section(key)
        return MappingProxyType({name: section.get(name, kind) for name in section._data})

    def _where(self, key: str) -> str:
        return f"{self._path}.{key}" if self._path else key
