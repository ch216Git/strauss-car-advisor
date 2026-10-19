"""The only place that knows the rules.json keys.

Reads config/rules.json once, validates it, and exposes it as a typed, immutable Rules object.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

from app.config_file import Section, check_type, load_json_config, parse_positive_int, parse_rate, parse_str, shown
from app.csv_columns import COLUMN_TO_FIELD
from app.errors import ConfigError

DEFAULT_RULES_PATH = Path(__file__).parent.parent / "config" / "rules.json"


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
    return load_json_config(path, _parse)


def _parse(root: "Section") -> Rules:
    electric = root.section("electric")
    family = root.section("familyTrack")
    waiver = root.section("waiver")
    net = root.section("netMethod")
    pension = net.section("pensionCredit")
    participation = net.section("participation")
    rounding = net.section("rounding")
    value_of_use = root.section("valueOfUse")

    group_order = root.list_of("groupOrder", parse_str, non_empty=True)
    budgets = root.mapping_of("budgets", parse_positive_int)
    if any(group not in budgets for group in group_order):
        raise ConfigError(
            "budgets",
            f"a budget for every group in groupOrder {shown(list(group_order))}",
            f"budgets for {shown(list(budgets))}",
        )

    brackets = net.list_of("brackets", _bracket, non_empty=True)
    _check_bracket_order(brackets, "netMethod.brackets")

    return Rules(
        version=root.get("version", str),
        budgets=budgets,
        group_order=group_order,
        choose_from=root.get("chooseFrom", str),
        electric=ElectricRules(
            net_bonus=electric.parse("netBonus", parse_positive_int),
            method=electric.get("method", str),
        ),
        family_track=FamilyTrackRules(
            min_children=family.get("minChildren", int),
            child_max_age=family.get("childMaxAge", int),
            cars=family.list_of("cars", parse_str, non_empty=True),
            price_source=family.get("priceSource", str),
            gap_split_employee=family.parse("gapSplitEmployee", parse_rate),
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
            credit_point_value=net.parse("creditPointValue", parse_positive_int),
            brackets=brackets,
            national_insurance=_contribution(net.section("nationalInsurance")),
            health=_contribution(net.section("health")),
            pension_credit=PensionCreditRules(
                rate=pension.parse("rate", parse_rate),
                max_contribution_rate=pension.parse("maxContributionRate", parse_rate),
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
            drive_field=_drive_field(value_of_use.mapping_of("driveField", parse_str), "valueOfUse.driveField"),
        ),
    )


def _contribution(section: "Section") -> ContributionRules:
    return ContributionRules(
        threshold=section.get("threshold", int),
        ceiling=section.get("ceiling", int),
        low_rate=section.parse("lowRate", parse_rate),
        high_rate=section.parse("highRate", parse_rate),
    )


def _bracket(value: Any, where: str) -> TaxBracket:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(where, "[upper limit or null, rate]", shown(value))
    upper_limit, rate = value
    return TaxBracket(
        upper_limit=None if upper_limit is None else check_type(upper_limit, f"{where}[0]", int),
        rate=parse_rate(rate, f"{where}[1]"),
    )


def _check_bracket_order(brackets: tuple[TaxBracket, ...], where: str) -> None:
    """Upper limits must strictly increase; only the last bracket may be open-ended (null)."""
    for i, bracket in enumerate(brackets):
        if bracket.upper_limit is None:
            if i != len(brackets) - 1:
                raise ConfigError(f"{where}[{i}][0]", "a limit (only the last bracket may be null)", "null")
        elif i > 0 and bracket.upper_limit <= (previous := brackets[i - 1].upper_limit):
            raise ConfigError(
                f"{where}[{i}][0]", f"a limit above the previous one ({previous})", shown(bracket.upper_limit)
            )


def _csv_column(value: Any, where: str) -> str:
    column = check_type(value, where, str)
    if column not in COLUMN_TO_FIELD:
        raise ConfigError(where, "a column name known to csv_columns.COLUMN_TO_FIELD", shown(column))
    return column


def _drive_field(names_by_code: Mapping[str, str], where: str) -> Mapping[int, str]:
    """Code -> drive name; names must be unique so the mapping can be read in both directions."""
    drive_field: dict[int, str] = {}
    for code, name in names_by_code.items():
        if not code.isdigit():
            raise ConfigError(f"{where}.{code}", "an integer code as key", shown(code))
        if name in drive_field.values():
            raise ConfigError(f"{where}.{code}", "a drive name not used by another code", shown(name))
        drive_field[int(code)] = name
    return MappingProxyType(drive_field)
