"""Step 3 comparison: an employee's current situation against one alternative, both through salary.py.

Deterministic: the same employee, alternative, rules, catalogue and CSV always give the same result.
No natural-language text is produced here; ComparisonResult is structured data for later steps.

Known limitations:
- A current car must be a catalogue car; cars outside the catalogue (official-key reference) are not
  supported yet.
- A current waiver can only be calculated when the occupation allows waiving: the workbook gives no
  maintenance amount otherwise, and none is invented. Such an employee gets a ValueError.
- Children are given as a count up to rules familyTrack.childMaxAge (the workbook's form), not as ages.
- Only electric.method "gross_up" is implemented. The briefs mention a "250 gross" second option in
  the official rules file, but the rules.json we have does not define it (see salary.py).
- The catalogue vehicle type for the CSV lookup is a dataset assumption (lookup.json), not a rule.
- Pension credit: qualifying income is min(pensionable income, pensionCredit.incomeCeiling); the
  workbook employees all earn above the ceiling, so the lower case follows the rules text (salary.py).
"""

from dataclasses import dataclass, fields
from decimal import ROUND_HALF_UP, Decimal
from functools import cache

from app.catalog import Catalog, CatalogCar, Occupation, load_catalog, lookup_vehicle
from app.errors import NotEligibleError
from app.family_track import FamilyTrackShare, check_family_track_eligibility, family_track_share, is_family_track_car
from app.lookup_config import LookupConfig, load_lookup_config
from app.rules import Rules, load_rules
from app.salary import NetBreakdown, SalaryInput, net_breakdown
from app.vehicle_repository import VehicleRepository

# rules.json valueOfUse.driveField name of the drive that gets the electric reimbursement.
ELECTRIC_DRIVE = "חשמלי"

# The rules settings this module implements; anything else is refused, not ignored.
SUPPORTED_SETTINGS = {
    "chooseFrom": "own_group_or_lower",
    "waiver.allowedBy": "occupation",
    "unusedBudget": "not_paid",  # nothing to add: unused budget never reaches the salary
    "netMethod.rounding.display": "shekel",
}


# --- Input -------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CompanyCar:
    car_id: str  # catalogue car ID
    participation: float | None = None  # monthly; None = the catalogue default (alternatives only)


@dataclass(frozen=True, slots=True)
class Waiver:
    """No company car; car maintenance (אחזקת רכב) by occupation instead."""


Situation = CompanyCar | Waiver


@dataclass(frozen=True, slots=True)
class Employee:
    occupation: str  # name in the workbook occupations table; gives the group and the waiver amount
    base_salary: float
    commissions: float
    bonus: float
    pension_rate: float  # fraction, 0.06 = 6%
    training_fund_rate: float  # fraction
    credit_points: float
    eligible_children: int  # children up to rules familyTrack.childMaxAge
    current: Situation  # participation must be stated for a current car


# --- Output ------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SituationDetails:
    kind: str  # "company_car" or "waiver"
    car_id: str | None
    commercial_name: str | None
    drive: str | None  # rules valueOfUse.driveField name of the CSV drive code
    use_value: int  # from the Tax Authority CSV (Step 1); 0 for a waiver
    participation: float
    participation_source: str | None  # "stated" or "catalogue_default"
    electric: bool
    family_track: FamilyTrackShare | None
    maintenance: int  # occupation's waiver amount; 0 with a car


@dataclass(frozen=True, slots=True)
class Change:
    item: str  # NetBreakdown field name
    current: float
    alternative: float

    @property
    def difference(self) -> float:
        return self.alternative - self.current


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    group: str  # the employee's group, from the occupation
    eligibility_basis: str  # "own_group_or_lower", "family_track", "waiver_allowed" or "keeps_current_situation"
    current_situation: SituationDetails
    alternative_situation: SituationDetails
    current: NetBreakdown
    alternative: NetBreakdown
    net_difference: int  # displayed alternative net - displayed current net, in whole shekels
    exact_net_difference: float
    changes: tuple[Change, ...]  # every NetBreakdown item that differs, in NetBreakdown order


# --- Comparison --------------------------------------------------------------

def compare(
    employee: Employee,
    alternative: Situation,
    *,
    rules: Rules | None = None,
    catalog: Catalog | None = None,
    lookup: LookupConfig | None = None,
    repository: VehicleRepository | None = None,
) -> ComparisonResult:
    """Compare the employee's current situation with the alternative; NotEligibleError if not allowed."""
    sources = _Sources(
        rules=rules or load_rules(),
        catalog=catalog or load_catalog(),
        lookup=lookup or load_lookup_config(),
        repository=repository or _default_repository(),
    )
    _check_supported(sources.rules)
    occupation = _occupation(employee, sources)
    if isinstance(employee.current, CompanyCar) and employee.current.participation is None:
        raise ValueError("The current car's participation must be stated; catalogue defaults are for alternatives only")

    eligibility_basis = (
        "keeps_current_situation"
        if _same_situation(employee.current, alternative)
        else _check_eligibility(employee, occupation, alternative, sources)
    )
    current_input, current_details = _resolve(employee, occupation, employee.current, sources)
    alternative_input, alternative_details = _resolve(employee, occupation, alternative, sources)

    current = net_breakdown(current_input, sources.rules)
    alternative_result = net_breakdown(alternative_input, sources.rules)
    return ComparisonResult(
        group=occupation.group,
        eligibility_basis=eligibility_basis,
        current_situation=current_details,
        alternative_situation=alternative_details,
        current=current,
        alternative=alternative_result,
        net_difference=display_shekels(alternative_result.net) - display_shekels(current.net),
        exact_net_difference=alternative_result.net - current.net,
        changes=_changes(current, alternative_result),
    )


def display_shekels(amount: float) -> int:
    """Whole shekels for display (rules netMethod.rounding.display), half rounded up."""
    return int(Decimal(str(amount)).quantize(Decimal(1), rounding=ROUND_HALF_UP))


@dataclass(frozen=True, slots=True)
class _Sources:
    rules: Rules
    catalog: Catalog
    lookup: LookupConfig
    repository: VehicleRepository


@cache
def _default_repository() -> VehicleRepository:
    return VehicleRepository()


def _check_supported(rules: Rules) -> None:
    actual = {
        "chooseFrom": rules.choose_from,
        "waiver.allowedBy": rules.waiver.allowed_by,
        "unusedBudget": rules.unused_budget,
        "netMethod.rounding.display": rules.net_method.rounding.display,
    }
    unsupported = {key: value for key, value in actual.items() if value != SUPPORTED_SETTINGS[key]}
    if unsupported:
        raise NotImplementedError(f"rules.json settings not supported by the comparison: {unsupported}")
    if ELECTRIC_DRIVE not in rules.value_of_use.drive_field.values():
        raise NotImplementedError(f"rules.json valueOfUse.driveField has no {ELECTRIC_DRIVE!r} drive")


def _occupation(employee: Employee, sources: _Sources) -> Occupation:
    occupation = sources.catalog.occupations.get(employee.occupation)
    if occupation is None:
        raise ValueError(f"Unknown occupation {employee.occupation!r}; known: {list(sources.catalog.occupations)}")
    if occupation.group not in sources.rules.group_order:
        raise ValueError(f"Occupation {occupation.name!r} has group {occupation.group!r}, not in rules.json groupOrder")
    return occupation


def _car(car_id: str, sources: _Sources) -> CatalogCar:
    car = sources.catalog.cars.get(car_id)
    if car is None:
        raise ValueError(f"Unknown catalogue car {car_id!r}")
    return car


def _same_situation(current: Situation, alternative: Situation) -> bool:
    """Keeping the current car (any participation) or the current waiver is not a new selection."""
    if isinstance(current, Waiver) and isinstance(alternative, Waiver):
        return True
    return isinstance(current, CompanyCar) and isinstance(alternative, CompanyCar) and current.car_id == alternative.car_id


def _check_eligibility(employee: Employee, occupation: Occupation, alternative: Situation, sources: _Sources) -> str:
    rules = sources.rules
    if isinstance(alternative, Waiver):
        if not occupation.waiver_allowed:
            raise NotEligibleError("waiver_not_allowed", f"Occupation {occupation.name!r} does not allow waiving the car")
        return "waiver_allowed"

    car = _car(alternative.car_id, sources)
    if is_family_track_car(car, rules):
        check_family_track_eligibility(car, employee.eligible_children, rules)
        return "family_track"
    if car.group not in rules.group_order:
        raise ValueError(f"Catalogue car {car.car_id} has group {car.group!r}, not in rules.json groupOrder")
    if rules.group_order.index(car.group) > rules.group_order.index(occupation.group):
        raise NotEligibleError(
            "group_too_high", f"{car.car_id} is in group {car.group}; group {occupation.group} may choose its own group or lower"
        )
    return "own_group_or_lower"


def _resolve(
    employee: Employee, occupation: Occupation, situation: Situation, sources: _Sources
) -> tuple[SalaryInput, SituationDetails]:
    """Turn a situation into salary-engine input, plus the facts behind it.

    Calculates the situation as it is; it does not check eligibility. Only a newly selected
    alternative is checked (_check_eligibility), so an existing car is never rejected because the
    employee could not select it today.
    """
    if isinstance(situation, Waiver):
        if not occupation.waiver_allowed:
            raise ValueError(f"Occupation {occupation.name!r} has no car-maintenance amount to receive")
        details = SituationDetails(
            kind="waiver", car_id=None, commercial_name=None, drive=None, use_value=0, participation=0,
            participation_source=None, electric=False, family_track=None, maintenance=occupation.waiver_maintenance,
        )
        return _salary_input(employee, details), details

    rules = sources.rules
    car = _car(situation.car_id, sources)
    record = lookup_vehicle(car, sources.repository, sources.lookup)
    drive = rules.value_of_use.drive_field.get(record.propulsion_code)
    if drive is None:
        raise ValueError(f"CSV drive code {record.propulsion_code} of {car.car_id} is not in rules.json valueOfUse.driveField")
    family = family_track_share(car, occupation.group, rules) if is_family_track_car(car, rules) else None
    stated = situation.participation is not None
    details = SituationDetails(
        kind="company_car",
        car_id=car.car_id,
        commercial_name=record.commercial_name,
        drive=drive,
        use_value=record.use_value,
        participation=situation.participation if stated else car.default_participation,
        participation_source="stated" if stated else "catalogue_default",
        electric=drive == ELECTRIC_DRIVE,
        family_track=family,
        maintenance=0,
    )
    return _salary_input(employee, details), details


def _salary_input(employee: Employee, details: SituationDetails) -> SalaryInput:
    return SalaryInput(
        base_salary=employee.base_salary,
        commissions=employee.commissions,
        bonus=employee.bonus,
        pension_rate=employee.pension_rate,
        training_fund_rate=employee.training_fund_rate,
        credit_points=employee.credit_points,
        use_value=details.use_value,
        participation=details.participation,
        maintenance=details.maintenance,
        electric_car=details.electric,
        family_share=details.family_track.employee_share if details.family_track else 0,
    )


def _changes(current: NetBreakdown, alternative: NetBreakdown) -> tuple[Change, ...]:
    return tuple(
        Change(item=f.name, current=getattr(current, f.name), alternative=getattr(alternative, f.name))
        for f in fields(NetBreakdown)
        if getattr(current, f.name) != getattr(alternative, f.name)
    )
