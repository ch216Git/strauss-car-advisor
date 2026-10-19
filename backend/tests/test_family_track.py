from dataclasses import FrozenInstanceError, replace

import pytest

from app.catalog import Catalog, CatalogCar, load_catalog, lookup_vehicle
from app.errors import NotEligibleError
from app.family_track import check_family_track_eligibility, family_track_share, is_family_track_car
from app.lookup_config import load_lookup_config
from app.rules import Rules, load_rules
from app.salary import SalaryInput, net_breakdown
from app.vehicle_repository import VehicleRepository


@pytest.fixture(scope="module")
def rules() -> Rules:
    return load_rules()


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return load_catalog()


@pytest.fixture(scope="module")
def family_cars(rules: Rules, catalog: Catalog) -> list[CatalogCar]:
    return [catalog.cars[car_id] for car_id in rules.family_track.cars]


def car_named(catalog: Catalog, commercial_name: str) -> CatalogCar:
    return next(car for car in catalog.cars.values() if car.commercial_name == commercial_name)


# --- Workbook reference ('עובדים לדוגמה' + 'חישובים', E-002 alternative) ------
# Hand-copied from the workbook: an independent oracle. E-002 is in group ב with 5 children up to 23,
# drives a NIRO today and asks about the CARNIVAL on the family track.
E002 = dict(group="ב", children=5, base=9800, commissions=2400, pension=6, training=0, points=2.25)
E002_ALTERNATIVE = dict(car="CARNIVAL", employee_share=1150, net=5078, tax=2861, insurance=1879)


def test_e002_employee_share_matches_workbook(rules: Rules, catalog: Catalog) -> None:
    share = family_track_share(car_named(catalog, E002_ALTERNATIVE["car"]), E002["group"], rules)

    assert share.employee_share == E002_ALTERNATIVE["employee_share"]


def test_e002_alternative_net_matches_workbook(rules: Rules, catalog: Catalog) -> None:
    car = car_named(catalog, E002_ALTERNATIVE["car"])
    share = family_track_share(car, E002["group"], rules)
    use_value = lookup_vehicle(car, VehicleRepository(), load_lookup_config()).use_value

    result = net_breakdown(
        SalaryInput(
            base_salary=E002["base"], commissions=E002["commissions"], bonus=0, pension_rate=E002["pension"] / 100,
            training_fund_rate=E002["training"] / 100, credit_points=E002["points"], use_value=use_value,
            participation=car.default_participation, maintenance=0, electric_car=False,
            family_share=share.employee_share,
        ),
        rules,
    )

    assert round(result.net) == E002_ALTERNATIVE["net"]
    assert round(result.income_tax) == E002_ALTERNATIVE["tax"]
    assert round(result.national_insurance + result.health) == E002_ALTERNATIVE["insurance"]


# --- Share calculation -------------------------------------------------------

def test_share_for_every_family_car_and_group(rules: Rules, family_cars: list[CatalogCar]) -> None:
    for car in family_cars:
        for group in rules.group_order:
            share = family_track_share(car, group, rules)
            budget = rules.budgets[group]
            assert (share.car_id, share.car_cost, share.budget) == (car.car_id, car.monthly_cost, budget)
            assert share.gap == max(0, car.monthly_cost - budget)
            assert share.employee_share == share.gap * rules.family_track.gap_split_employee


def test_no_share_when_the_budget_covers_the_car(rules: Rules, family_cars: list[CatalogCar]) -> None:
    car = family_cars[0]
    group = rules.group_order[0]
    generous = replace(rules, budgets={**rules.budgets, group: car.monthly_cost + 1_000})

    share = family_track_share(car, group, generous)

    assert (share.gap, share.employee_share) == (0, 0)


def test_split_comes_from_rules(rules: Rules, family_cars: list[CatalogCar]) -> None:
    car, group = family_cars[0], rules.group_order[0]
    quarter = replace(rules, family_track=replace(rules.family_track, gap_split_employee=0.25))

    default = family_track_share(car, group, rules)
    changed = family_track_share(car, group, quarter)

    assert changed.gap == default.gap > 0
    assert changed.employee_share == default.gap * 0.25


def test_share_is_immutable(rules: Rules, family_cars: list[CatalogCar]) -> None:
    share = family_track_share(family_cars[0], rules.group_order[0], rules)

    with pytest.raises(FrozenInstanceError):
        share.employee_share = 0  # type: ignore[misc]


# --- Eligibility to select a family car -------------------------------------

def test_minimum_children_is_eligible(rules: Rules, family_cars: list[CatalogCar]) -> None:
    check_family_track_eligibility(family_cars[0], rules.family_track.min_children, rules)


def test_too_few_children_is_not_eligible(rules: Rules, family_cars: list[CatalogCar]) -> None:
    with pytest.raises(NotEligibleError) as info:
        check_family_track_eligibility(family_cars[0], rules.family_track.min_children - 1, rules)

    assert info.value.reason == "not_enough_children"
    assert f"up to age {rules.family_track.child_max_age}" in str(info.value)


def test_regular_car_is_not_a_family_track_car(rules: Rules, catalog: Catalog) -> None:
    regular = next(car for car in catalog.cars.values() if not is_family_track_car(car, rules))

    with pytest.raises(NotEligibleError) as info:
        check_family_track_eligibility(regular, rules.family_track.min_children, rules)

    assert info.value.reason == "not_family_track_car"


def test_allowed_cars_come_from_rules(rules: Rules, catalog: Catalog, family_cars: list[CatalogCar]) -> None:
    regular = next(car for car in catalog.cars.values() if not is_family_track_car(car, rules))
    only_regular = replace(rules, family_track=replace(rules.family_track, cars=(regular.car_id,)))

    check_family_track_eligibility(regular, rules.family_track.min_children, only_regular)
    with pytest.raises(NotEligibleError):
        check_family_track_eligibility(family_cars[0], rules.family_track.min_children, only_regular)


def test_minimum_children_comes_from_rules(rules: Rules, family_cars: list[CatalogCar]) -> None:
    stricter = replace(rules, family_track=replace(rules.family_track, min_children=rules.family_track.min_children + 1))

    with pytest.raises(NotEligibleError):
        check_family_track_eligibility(family_cars[0], rules.family_track.min_children, stricter)


def test_negative_children_raises(rules: Rules, family_cars: list[CatalogCar]) -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        check_family_track_eligibility(family_cars[0], -1, rules)


# --- Share without eligibility -----------------------------------------------

def test_share_does_not_depend_on_eligibility(rules: Rules, family_cars: list[CatalogCar]) -> None:
    # The share takes no children input at all; only check_family_track_eligibility looks at children.
    car, group = family_cars[0], rules.group_order[1]
    with pytest.raises(NotEligibleError):
        check_family_track_eligibility(car, 0, rules)

    share = family_track_share(car, group, rules)

    assert share.employee_share == max(0, car.monthly_cost - rules.budgets[group]) * rules.family_track.gap_split_employee


def test_share_of_a_regular_car_raises(rules: Rules, catalog: Catalog) -> None:
    regular = next(car for car in catalog.cars.values() if not is_family_track_car(car, rules))

    with pytest.raises(ValueError, match="not a family-track car"):
        family_track_share(regular, rules.group_order[0], rules)


# --- Refusals ----------------------------------------------------------------

def test_unknown_group_raises(rules: Rules, family_cars: list[CatalogCar]) -> None:
    with pytest.raises(ValueError, match="Unknown employee group"):
        family_track_share(family_cars[0], "ז", rules)


def test_unsupported_price_source_is_refused(rules: Rules, family_cars: list[CatalogCar]) -> None:
    other_source = replace(rules, family_track=replace(rules.family_track, price_source="list_price"))

    with pytest.raises(NotImplementedError, match="priceSource 'list_price'"):
        family_track_share(family_cars[0], rules.group_order[0], other_source)
