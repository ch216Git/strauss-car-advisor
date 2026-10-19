"""Family track (מסלול ברוכת ילדים): the employee's share of a family car's cost above the group budget.

Everything comes from rules.json familyTrack and the group budgets, and the car cost from the catalogue.
Eligibility (may the employee select this car now?) is separate from the share (what does a family
car cost the employee?), so an existing family car is still calculated after eligibility lapses.
"""

from dataclasses import dataclass

from app.catalog import CatalogCar
from app.errors import NotEligibleError
from app.rules import Rules

# The only familyTrack.priceSource implemented: the car's monthly cost in the workbook catalogue.
CATALOGUE_PRICE = "catalogue"


@dataclass(frozen=True, slots=True)
class FamilyTrackShare:
    car_id: str
    car_cost: int  # catalogue monthly cost to the company
    budget: int  # the employee's group budget
    gap: int  # car cost above the budget; 0 when the budget covers it
    employee_share: float  # gap x gapSplitEmployee; a net deduction


def check_family_track_eligibility(car: CatalogCar, eligible_children: int, rules: Rules) -> None:
    """Raise NotEligibleError unless the employee may select this car on the family track now.

    eligible_children counts children up to rules familyTrack.childMaxAge; the age limit is applied
    when the count is collected.
    """
    family = rules.family_track
    if eligible_children < 0:
        raise ValueError(f"Number of children must not be negative: {eligible_children}")
    if not is_family_track_car(car, rules):
        raise NotEligibleError(
            "not_family_track_car", f"{car.car_id} is not a family-track car; family-track cars: {list(family.cars)}"
        )
    if eligible_children < family.min_children:
        raise NotEligibleError(
            "not_enough_children",
            f"Family track needs at least {family.min_children} children up to age {family.child_max_age}; "
            f"got {eligible_children}",
        )


def family_track_share(car: CatalogCar, employee_group: str, rules: Rules) -> FamilyTrackShare:
    """The employee's share for a family-track car. Does not check eligibility to select it."""
    family = rules.family_track
    if family.price_source != CATALOGUE_PRICE:
        raise NotImplementedError(
            f"rules.json familyTrack.priceSource {family.price_source!r} is not supported; only {CATALOGUE_PRICE!r}"
        )
    if employee_group not in rules.budgets:
        raise ValueError(f"Unknown employee group {employee_group!r}; rules.json groups: {list(rules.group_order)}")
    if not is_family_track_car(car, rules):
        raise ValueError(f"{car.car_id} is not a family-track car; it has no family-track share")

    budget = rules.budgets[employee_group]
    gap = max(0, car.monthly_cost - budget)
    return FamilyTrackShare(
        car_id=car.car_id,
        car_cost=car.monthly_cost,
        budget=budget,
        gap=gap,
        employee_share=gap * family.gap_split_employee,
    )


def is_family_track_car(car: CatalogCar, rules: Rules) -> bool:
    return car.car_id in rules.family_track.cars
