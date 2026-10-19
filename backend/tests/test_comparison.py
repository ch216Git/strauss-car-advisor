from dataclasses import FrozenInstanceError, replace

import pytest

from app.catalog import Catalog, load_catalog, lookup_vehicle
from app.comparison import (
    CompanyCar,
    ComparisonResult,
    Employee,
    Waiver,
    compare,
    display_shekels,
)
from app.errors import NotEligibleError
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


def car_id(catalog: Catalog, commercial_name: str) -> str:
    return next(car.car_id for car in catalog.cars.values() if car.commercial_name == commercial_name)


# --- Workbook golden tests ('עובדים לדוגמה' + 'חישובים') ----------------------
# Hand-copied from the challenge workbook: an independent oracle. Cars are given by the workbook's
# commercial name ("WAIVER" = no car, maintenance by occupation); alternatives use the catalogue
# default participation, as the workbook does.
WORKBOOK_EMPLOYEES = {
    # id: occupation, base, commissions, bonus, pension%, training%, points, children, car today, participation today
    "E-001": ("מנהל/ת מוצר", 18500, 0, 0, 6, 2.5, 2.75, 2, "COROLLA CROSS", 300),
    "E-002": ("נציג/ת מכירות שטח", 9800, 2400, 0, 6, 0, 2.25, 5, "NIRO", 250),
    "E-003": ("מפתח/ת תוכנה", 21000, 0, 0, 6.5, 2.5, 2.75, 0, "WAIVER", 0),
    "E-004": ("מנהל/ת אזור מכירות", 16000, 3000, 0, 6, 2.5, 2.25, 3, "TUCSON", 400),
    "E-005": ("מנהל/ת פיתוח", 27000, 0, 0, 7, 2.5, 2.75, 1, "MAZDA 3", 350),
    "E-006": ("מנהל/ת מחלקה", 32000, 0, 2000, 6, 2.5, 2.25, 4, "OCTAVIA", 500),
}
WORKBOOK_RESULTS = [
    # id, group, alternative, basis, net today, net alt, difference, tax today, tax alt, NI+health today,
    #     NI+health alt, electric gross alt, family share alt
    ("E-001", "ג", "ATTO 3", "own_group_or_lower", 11032, 11693, 661, 3443, 3283, 2153, 2092, 440, 0),
    ("E-002", "ב", "CARNIVAL", "family_track", 8600, 5078, -3522, 1369, 2861, 1250, 1879, 0, 1150),
    ("E-003", "ב", "MG4", "own_group_or_lower", 15047, 13109, -1938, 3419, 3910, 2144, 2315, 473, 0),
    ("E-004", "ג", "EV3", "own_group_or_lower", 10693, 11900, 1207, 3991, 3497, 2301, 2128, 440, 0),
    ("E-005", "ג", "WAIVER", "waiver_allowed", 14244, 17806, 3562, 6593, 5694, 3248, 2935, 0, 0),
    ("E-006", "ד", "MODEL 3", "own_group_or_lower", 17073, 17794, 721, 9367, 9183, 4170, 4106, 473, 0),
]


def situation(catalog: Catalog, name: str, participation: float | None = None):
    return Waiver() if name == "WAIVER" else CompanyCar(car_id(catalog, name), participation)


def workbook_employee(catalog: Catalog, employee_id: str) -> Employee:
    occupation, base, commissions, bonus, pension, training, points, children, car, participation = (
        WORKBOOK_EMPLOYEES[employee_id]
    )
    return Employee(
        occupation=occupation, base_salary=base, commissions=commissions, bonus=bonus,
        pension_rate=pension / 100, training_fund_rate=training / 100, credit_points=points,
        eligible_children=children, current=situation(catalog, car, participation),
    )


@pytest.mark.parametrize(
    "employee_id, group, alternative, basis, net_today, net_alt, difference, tax_today, tax_alt, "
    "insurance_today, insurance_alt, electric_gross, family_share",
    WORKBOOK_RESULTS,
)
def test_workbook_alternative(
    catalog: Catalog, employee_id, group, alternative, basis, net_today, net_alt, difference, tax_today, tax_alt,
    insurance_today, insurance_alt, electric_gross, family_share,
) -> None:
    result = compare(workbook_employee(catalog, employee_id), situation(catalog, alternative))

    assert (result.group, result.eligibility_basis) == (group, basis)
    assert (display_shekels(result.current.net), display_shekels(result.alternative.net)) == (net_today, net_alt)
    assert result.net_difference == difference
    assert (round(result.current.income_tax), round(result.alternative.income_tax)) == (tax_today, tax_alt)
    assert round(result.current.national_insurance + result.current.health) == insurance_today
    assert round(result.alternative.national_insurance + result.alternative.health) == insurance_alt
    assert round(result.alternative.electric_gross) == electric_gross
    assert round(result.alternative.family_share) == family_share


# --- Engine wiring -----------------------------------------------------------

@pytest.fixture(scope="module")
def e001(catalog: Catalog) -> Employee:
    return workbook_employee(catalog, "E-001")


def test_both_sides_use_the_salary_engine(catalog: Catalog, e001: Employee) -> None:
    # Reconstructing each side's input from the structured details gives the very same breakdown.
    result = compare(e001, CompanyCar(car_id(catalog, "ATTO 3")))
    alt = result.alternative_situation

    assert alt.electric and alt.participation_source == "catalogue_default"
    assert result.alternative == net_breakdown(
        salary_input(e001, use_value=alt.use_value, participation=alt.participation, electric_car=True)
    )


def salary_input(employee: Employee, **car) -> SalaryInput:
    return SalaryInput(
        base_salary=employee.base_salary, commissions=employee.commissions, bonus=employee.bonus,
        pension_rate=employee.pension_rate, training_fund_rate=employee.training_fund_rate,
        credit_points=employee.credit_points, maintenance=0, family_share=0, **car,
    )


def test_use_value_comes_from_step_1(catalog: Catalog, e001: Employee) -> None:
    result = compare(e001, CompanyCar(car_id(catalog, "ATTO 3")))
    car = catalog.cars[result.alternative_situation.car_id]
    record = lookup_vehicle(car, VehicleRepository(), load_lookup_config())

    assert result.alternative_situation.use_value == record.use_value
    assert result.alternative_situation.commercial_name == record.commercial_name
    assert result.alternative.use_value == record.use_value


def test_drive_comes_from_csv_through_rules(catalog: Catalog, e001: Employee, rules: Rules) -> None:
    result = compare(e001, CompanyCar(car_id(catalog, "ATTO 3")))
    record = lookup_vehicle(catalog.cars[result.current_situation.car_id], VehicleRepository(), load_lookup_config())

    assert result.current_situation.drive == rules.value_of_use.drive_field[record.propulsion_code]
    assert not result.current_situation.electric


def test_stated_participation_overrides_catalogue_default(catalog: Catalog, e001: Employee) -> None:
    default = compare(e001, CompanyCar(car_id(catalog, "ATTO 3")))
    stated = compare(e001, CompanyCar(car_id(catalog, "ATTO 3"), participation=350))

    assert stated.alternative_situation.participation_source == "stated"
    assert stated.alternative.participation == 350
    assert stated.alternative.net == pytest.approx(default.alternative.net - (350 - default.alternative.participation))


def test_current_electric_car_is_grossed_up_too(catalog: Catalog, e001: Employee) -> None:
    electric_today = replace(e001, current=CompanyCar(car_id(catalog, "ATTO 3"), participation=300))

    result = compare(electric_today, CompanyCar(car_id(catalog, "COROLLA CROSS")))

    assert result.current.electric_gross > 0
    assert result.alternative.electric_gross == 0


def test_current_family_car_keeps_its_share(catalog: Catalog) -> None:
    family_today = replace(workbook_employee(catalog, "E-002"), current=CompanyCar(car_id(catalog, "CARNIVAL"), 500))

    result = compare(family_today, CompanyCar(car_id(catalog, "NIRO")))

    assert result.current_situation.family_track is not None
    assert result.current.family_share == result.current_situation.family_track.employee_share
    assert result.alternative.family_share == 0


# --- Current situation is calculated, not re-validated -----------------------
# E-002 drives a CARNIVAL on the family track but now has too few children to select it again.

@pytest.fixture(scope="module")
def lapsed_family_employee(catalog: Catalog, rules: Rules) -> Employee:
    return replace(
        workbook_employee(catalog, "E-002"),
        current=CompanyCar(car_id(catalog, "CARNIVAL"), participation=500),
        eligible_children=rules.family_track.min_children - 1,
    )


def test_existing_family_car_is_calculated_without_eligibility(
    catalog: Catalog, lapsed_family_employee: Employee
) -> None:
    result = compare(lapsed_family_employee, CompanyCar(car_id(catalog, "NIRO")))
    family = result.current_situation.family_track

    assert family is not None and family.employee_share > 0
    assert result.current.family_share == family.employee_share
    assert result.alternative.family_share == 0


def test_existing_family_share_is_the_same_as_when_eligible(
    catalog: Catalog, lapsed_family_employee: Employee, rules: Rules
) -> None:
    eligible = replace(lapsed_family_employee, eligible_children=rules.family_track.min_children)
    niro = CompanyCar(car_id(catalog, "NIRO"))

    assert compare(lapsed_family_employee, niro).current == compare(eligible, niro).current


def test_keeping_the_existing_family_car_is_not_a_new_selection(
    catalog: Catalog, lapsed_family_employee: Employee
) -> None:
    result = compare(lapsed_family_employee, CompanyCar(car_id(catalog, "CARNIVAL")))

    assert result.eligibility_basis == "keeps_current_situation"
    assert result.net_difference == 0


def test_new_family_car_still_needs_eligibility(catalog: Catalog, lapsed_family_employee: Employee) -> None:
    # Switching to the other family car is a new selection: the full check applies.
    with pytest.raises(NotEligibleError) as info:
        compare(lapsed_family_employee, CompanyCar(car_id(catalog, "SANTA FE")))

    assert info.value.reason == "not_enough_children"


def test_new_family_car_with_enough_children_is_eligible(
    catalog: Catalog, lapsed_family_employee: Employee, rules: Rules
) -> None:
    eligible = replace(lapsed_family_employee, eligible_children=rules.family_track.min_children)

    assert compare(eligible, CompanyCar(car_id(catalog, "SANTA FE"))).eligibility_basis == "family_track"


def test_waiver_maintenance_comes_from_occupation(catalog: Catalog) -> None:
    employee = workbook_employee(catalog, "E-005")

    result = compare(employee, Waiver())

    assert result.alternative_situation.maintenance == catalog.occupations[employee.occupation].waiver_maintenance
    assert result.alternative.maintenance == result.alternative_situation.maintenance
    assert (result.alternative.use_value, result.alternative.participation) == (0, 0)


def test_same_situation_has_no_changes(catalog: Catalog, e001: Employee) -> None:
    result = compare(e001, e001.current)

    assert result.eligibility_basis == "keeps_current_situation"
    assert result.net_difference == 0
    assert result.changes == ()


def test_changes_list_only_what_differs(catalog: Catalog, e001: Employee) -> None:
    result = compare(e001, CompanyCar(car_id(catalog, "ATTO 3")))
    items = {change.item: change for change in result.changes}

    assert {"use_value", "electric_gross", "income_tax", "national_insurance", "health", "net"} <= set(items)
    assert not {"pension", "training_fund", "pensionable_income", "maintenance", "family_share"} & set(items)
    assert items["net"].difference == pytest.approx(result.exact_net_difference)
    assert items["use_value"].current == result.current.use_value


def test_net_difference_uses_displayed_nets(catalog: Catalog) -> None:
    # E-003: exact difference rounds to -1939, but the workbook shows 13,109 - 15,047 = -1,938.
    result = compare(workbook_employee(catalog, "E-003"), CompanyCar(car_id(catalog, "MG4")))

    assert round(result.exact_net_difference) == -1939
    assert result.net_difference == -1938


def test_display_rounds_half_up() -> None:
    assert (display_shekels(10.5), display_shekels(11.5), display_shekels(10.49)) == (11, 12, 10)


def test_result_is_immutable(catalog: Catalog, e001: Employee) -> None:
    result: ComparisonResult = compare(e001, CompanyCar(car_id(catalog, "ATTO 3")))

    with pytest.raises(FrozenInstanceError):
        result.net_difference = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.alternative_situation.use_value = 0  # type: ignore[misc]


# --- Eligibility -------------------------------------------------------------

def test_car_from_a_higher_group_is_not_eligible(catalog: Catalog, e001: Employee) -> None:
    with pytest.raises(NotEligibleError) as info:
        compare(e001, CompanyCar(car_id(catalog, "MODEL 3")))  # group ד; E-001 is group ג

    assert info.value.reason == "group_too_high"


def test_car_from_a_lower_group_is_eligible(catalog: Catalog, e001: Employee) -> None:
    assert compare(e001, CompanyCar(car_id(catalog, "PICANTO"))).eligibility_basis == "own_group_or_lower"


def test_family_car_without_enough_children_is_not_eligible(catalog: Catalog, e001: Employee) -> None:
    with pytest.raises(NotEligibleError) as info:
        compare(e001, CompanyCar(car_id(catalog, "CARNIVAL")))  # E-001 has 2 children

    assert info.value.reason == "not_enough_children"


def test_waiver_for_a_mandatory_car_occupation_is_not_eligible(catalog: Catalog) -> None:
    with pytest.raises(NotEligibleError) as info:
        compare(workbook_employee(catalog, "E-002"), Waiver())  # field sales: car required

    assert info.value.reason == "waiver_not_allowed"


# --- Refusals ----------------------------------------------------------------

def test_current_car_needs_stated_participation(catalog: Catalog, e001: Employee) -> None:
    unstated = replace(e001, current=CompanyCar(car_id(catalog, "COROLLA CROSS")))

    with pytest.raises(ValueError, match="participation must be stated"):
        compare(unstated, Waiver())


def test_unknown_car_raises(e001: Employee) -> None:
    with pytest.raises(ValueError, match="Unknown catalogue car"):
        compare(e001, CompanyCar("CAR-999"))


def test_unknown_occupation_raises(e001: Employee) -> None:
    with pytest.raises(ValueError, match="Unknown occupation"):
        compare(replace(e001, occupation="אסטרונאוט/ית"), Waiver())


def test_unsupported_choose_from_is_refused(catalog: Catalog, e001: Employee, rules: Rules) -> None:
    with pytest.raises(NotImplementedError, match="chooseFrom"):
        compare(e001, CompanyCar(car_id(catalog, "ATTO 3")), rules=replace(rules, choose_from="any_group"))
