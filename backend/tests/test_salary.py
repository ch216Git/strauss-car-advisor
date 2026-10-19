from dataclasses import FrozenInstanceError, replace

import pytest

from app.catalog import load_catalog
from app.rules import Rules, load_rules
from app.salary import SalaryInput, bracket_tax, electric_gross_up, net_breakdown, two_tier_contribution


@pytest.fixture(scope="module")
def rules() -> Rules:
    return load_rules()


def salary(**changes) -> SalaryInput:
    """A plain employee with no car; tests change only what they are about."""
    base = SalaryInput(
        base_salary=20_000, commissions=0, bonus=0, pension_rate=0.06, training_fund_rate=0.025,
        credit_points=2.25, use_value=0, participation=0, maintenance=0, electric_car=False,
        family_share=0,
    )
    return replace(base, **changes)


# --- Workbook reference ('עובדים לדוגמה' + 'חישובים', today's situation) ------
# Inputs and expected results hand-copied from the challenge workbook: an independent oracle.
# A car row gives the current car's use value and participation; E-003 has no car today and
# gets car maintenance instead, taken from her occupation in the workbook (not typed in here).
WORKBOOK_TODAY = [
    # id, base, commissions, bonus, pension%, training%, points, use value, participation, waiver occupation,
    #     net, tax, NI+health
    ("E-001", 18500, 0, 0, 6, 2.5, 2.75, 4070, 300, None, 11032, 3443, 2153),
    ("E-002", 9800, 2400, 0, 6, 0, 2.25, 2950, 250, None, 8600, 1369, 1250),
    ("E-003", 21000, 0, 0, 6.5, 2.5, 2.75, 0, 0, "מפתח/ת תוכנה", 15047, 3419, 2144),
    ("E-004", 16000, 3000, 0, 6, 2.5, 2.25, 4790, 400, None, 10693, 3991, 2301),
    ("E-005", 27000, 0, 0, 7, 2.5, 2.75, 4570, 350, None, 14244, 6593, 3248),
    ("E-006", 32000, 0, 2000, 6, 2.5, 2.25, 5150, 500, None, 17073, 9367, 4170),
]


@pytest.mark.parametrize(
    "employee, base, commissions, bonus, pension, training, points, use_value, participation, waiver_occupation, "
    "net, tax, insurance",
    WORKBOOK_TODAY,
)
def test_matches_workbook_reference(
    rules: Rules, employee, base, commissions, bonus, pension, training, points, use_value, participation,
    waiver_occupation, net, tax, insurance,
) -> None:
    maintenance = load_catalog().occupations[waiver_occupation].waiver_maintenance if waiver_occupation else 0

    result = net_breakdown(
        SalaryInput(
            base_salary=base, commissions=commissions, bonus=bonus, pension_rate=pension / 100,
            training_fund_rate=training / 100, credit_points=points, use_value=use_value, participation=participation,
            maintenance=maintenance, electric_car=False, family_share=0,
        ),
        rules,
    )

    assert round(result.net) == net
    assert round(result.income_tax) == tax
    assert round(result.national_insurance + result.health) == insurance


# Electric alternatives from the same sheets: the car's use value and the catalogue's default
# participation, with the workbook's grossed-up reimbursement ('החזר חשמל ברוטו').
WORKBOOK_ELECTRIC = [
    # id, base, commissions, bonus, pension%, training%, points, use value, participation,
    #     electric gross, net, tax, NI+health
    ("E-001", 18500, 0, 0, 6, 2.5, 2.75, 3130, 300, 440, 11693, 3283, 2092),  # ATTO 3
    ("E-003", 21000, 0, 0, 6.5, 2.5, 2.75, 2430, 250, 473, 13109, 3910, 2315),  # MG4
    ("E-004", 16000, 3000, 0, 6, 2.5, 2.25, 2930, 300, 440, 11900, 3497, 2128),  # EV3
    ("E-006", 32000, 0, 2000, 6, 2.5, 2.25, 4150, 500, 473, 17794, 9183, 4106),  # MODEL 3
]


@pytest.mark.parametrize(
    "employee, base, commissions, bonus, pension, training, points, use_value, participation, "
    "electric_gross, net, tax, insurance",
    WORKBOOK_ELECTRIC,
)
def test_electric_matches_workbook_reference(
    rules: Rules, employee, base, commissions, bonus, pension, training, points, use_value, participation,
    electric_gross, net, tax, insurance,
) -> None:
    result = net_breakdown(
        SalaryInput(
            base_salary=base, commissions=commissions, bonus=bonus, pension_rate=pension / 100,
            training_fund_rate=training / 100, credit_points=points, use_value=use_value, participation=participation,
            maintenance=0, electric_car=True, family_share=0,
        ),
        rules,
    )

    assert round(result.electric_gross) == electric_gross
    assert round(result.net) == net
    assert round(result.income_tax) == tax
    assert round(result.national_insurance + result.health) == insurance


# --- Gross and taxable income -----------------------------------------------

def test_gross_income_is_base_plus_commissions_plus_bonus(rules: Rules) -> None:
    result = net_breakdown(salary(base_salary=10_000, commissions=1_500, bonus=500), rules)

    assert result.gross_income == 12_000


def test_use_value_is_taxable_but_not_paid(rules: Rules) -> None:
    without_car = net_breakdown(salary(), rules)
    with_car = net_breakdown(salary(use_value=4_000), rules)

    assert with_car.taxable_income == without_car.taxable_income + 4_000
    assert with_car.gross_income == without_car.gross_income
    assert with_car.income_tax > without_car.income_tax
    assert with_car.national_insurance > without_car.national_insurance
    assert with_car.health > without_car.health
    # The net drops only by the extra tax and insurance; the use value itself is not deducted.
    extra_charges = (with_car.income_tax + with_car.national_insurance + with_car.health) - (
        without_car.income_tax + without_car.national_insurance + without_car.health
    )
    assert with_car.net == pytest.approx(without_car.net - extra_charges)


def test_participation_is_a_net_deduction_only(rules: Rules) -> None:
    without = net_breakdown(salary(use_value=4_000), rules)
    with_participation = net_breakdown(salary(use_value=4_000, participation=350), rules)

    assert with_participation.net == pytest.approx(without.net - 350)
    assert with_participation.taxable_income == without.taxable_income  # reducesValueOfUse = false
    assert with_participation.income_tax == without.income_tax


# --- Car maintenance (waiver) -----------------------------------------------

def test_maintenance_is_paid_and_taxable(rules: Rules) -> None:
    without = net_breakdown(salary(), rules)
    with_maintenance = net_breakdown(salary(maintenance=1_500), rules)

    assert with_maintenance.gross_income == without.gross_income + 1_500
    assert with_maintenance.taxable_income == without.taxable_income + 1_500
    assert with_maintenance.income_tax > without.income_tax
    assert with_maintenance.national_insurance > without.national_insurance
    assert with_maintenance.health > without.health
    extra_charges = (with_maintenance.income_tax + with_maintenance.national_insurance + with_maintenance.health) - (
        without.income_tax + without.national_insurance + without.health
    )
    assert with_maintenance.net == pytest.approx(without.net + 1_500 - extra_charges)


def test_maintenance_is_not_pensionable(rules: Rules) -> None:
    without = net_breakdown(salary(), rules)
    with_maintenance = net_breakdown(salary(maintenance=1_500), rules)

    assert with_maintenance.pensionable_income == without.pensionable_income
    assert with_maintenance.pension == without.pension
    assert with_maintenance.training_fund == without.training_fund
    assert with_maintenance.pension_credit == without.pension_credit


def test_maintenance_amount_comes_from_the_occupation(rules: Rules) -> None:
    occupation = load_catalog().occupations["מנהל/ת מחלקה"]

    result = net_breakdown(salary(maintenance=occupation.waiver_maintenance), rules)

    assert occupation.waiver_allowed
    assert result.maintenance == occupation.waiver_maintenance


def test_maintenance_with_a_company_car_is_refused(rules: Rules) -> None:
    with pytest.raises(ValueError, match="instead of a company car"):
        net_breakdown(salary(maintenance=1_500, use_value=4_000, participation=300), rules)


# --- Electric gross-up -------------------------------------------------------

def net_gain_from_electric(salary_input: SalaryInput, rules: Rules) -> float:
    return net_breakdown(replace(salary_input, electric_car=True), rules).net - net_breakdown(salary_input, rules).net


@pytest.mark.parametrize(
    "case",
    [
        {"base_salary": 5_000, "credit_points": 4},  # no income tax at all: only insurance is grossed up
        {"base_salary": 9_000},
        {"base_salary": 20_000, "use_value": 4_000, "participation": 300},
        {"base_salary": 22_300},  # gross-up crosses the 31% -> 35% bracket limit
        {"base_salary": 60_000},  # above the national insurance and health ceilings
    ],
)
def test_employee_receives_exactly_the_net_bonus(rules: Rules, case) -> None:
    assert net_gain_from_electric(salary(**case), rules) == pytest.approx(rules.electric.net_bonus, abs=1e-6)


def test_gross_up_crossing_a_bracket_limit(rules: Rules) -> None:
    limit = next(b.upper_limit for b in rules.net_method.brackets if b.upper_limit and b.upper_limit > 22_000)
    result = net_breakdown(salary(base_salary=limit - 100, electric_car=True), rules)

    assert result.taxable_income - result.electric_gross < limit < result.taxable_income
    assert net_gain_from_electric(salary(base_salary=limit - 100), rules) == pytest.approx(rules.electric.net_bonus, abs=1e-6)


def test_net_bonus_comes_from_rules(rules: Rules) -> None:
    bigger_bonus = replace(rules, electric=replace(rules.electric, net_bonus=rules.electric.net_bonus * 2))

    assert net_gain_from_electric(salary(), bigger_bonus) == pytest.approx(bigger_bonus.electric.net_bonus, abs=1e-6)
    assert net_breakdown(salary(electric_car=True), bigger_bonus).electric_gross > net_breakdown(
        salary(electric_car=True), rules
    ).electric_gross


def test_gross_up_exceeds_net_bonus(rules: Rules) -> None:
    result = net_breakdown(salary(electric_car=True), rules)

    assert result.electric_gross > rules.electric.net_bonus
    assert result.electric_gross == electric_gross_up(salary(electric_car=True), rules)


def test_electric_gross_is_taxable_but_not_pensionable(rules: Rules) -> None:
    without = net_breakdown(salary(), rules)
    electric = net_breakdown(salary(electric_car=True), rules)

    assert electric.gross_income == pytest.approx(without.gross_income + electric.electric_gross)
    assert electric.taxable_income == pytest.approx(without.taxable_income + electric.electric_gross)
    assert electric.income_tax > without.income_tax
    assert electric.national_insurance > without.national_insurance
    assert electric.health > without.health
    assert electric.pensionable_income == without.pensionable_income
    assert (electric.pension, electric.training_fund, electric.pension_credit) == (
        without.pension, without.training_fund, without.pension_credit
    )


def test_non_electric_car_has_no_gross_up(rules: Rules) -> None:
    assert net_breakdown(salary(use_value=4_000, participation=300), rules).electric_gross == 0


def test_unsupported_electric_method_is_refused_only_for_electric_cars(rules: Rules) -> None:
    other_method = replace(rules, electric=replace(rules.electric, method="gross"))

    with pytest.raises(NotImplementedError, match="electric.method 'gross'"):
        net_breakdown(salary(electric_car=True), other_method)
    assert net_breakdown(salary(), other_method) == net_breakdown(salary(), rules)


def test_family_share_is_a_net_deduction_only(rules: Rules) -> None:
    without = net_breakdown(salary(use_value=8_000, participation=500), rules)
    with_share = net_breakdown(salary(use_value=8_000, participation=500, family_share=1_000), rules)

    assert with_share.family_share == 1_000
    assert (with_share.taxable_income, with_share.income_tax, with_share.national_insurance, with_share.health) == (
        without.taxable_income, without.income_tax, without.national_insurance, without.health
    )
    assert with_share.net == pytest.approx(without.net - 1_000)


def test_family_share_with_maintenance_is_refused(rules: Rules) -> None:
    with pytest.raises(ValueError, match="instead of a company car"):
        net_breakdown(salary(maintenance=1_500, family_share=1_000), rules)


def test_electric_with_maintenance_is_refused(rules: Rules) -> None:
    with pytest.raises(ValueError, match="instead of a company car"):
        net_breakdown(salary(maintenance=1_500, electric_car=True), rules)


# --- Income tax --------------------------------------------------------------

def test_bracket_tax_at_bracket_limits(rules: Rules) -> None:
    brackets = rules.net_method.brackets
    first, second = brackets[0], brackets[1]

    assert bracket_tax(0, brackets) == 0
    assert bracket_tax(first.upper_limit, brackets) == pytest.approx(first.upper_limit * first.rate)
    assert bracket_tax(second.upper_limit, brackets) == pytest.approx(
        first.upper_limit * first.rate + (second.upper_limit - first.upper_limit) * second.rate
    )


def test_bracket_tax_top_bracket_is_open_ended(rules: Rules) -> None:
    brackets = rules.net_method.brackets
    top_limit = brackets[-2].upper_limit

    assert bracket_tax(top_limit + 1_000, brackets) == pytest.approx(
        bracket_tax(top_limit, brackets) + 1_000 * brackets[-1].rate
    )


def test_credit_points_reduce_tax(rules: Rules) -> None:
    fewer = net_breakdown(salary(credit_points=2.25), rules)
    more = net_breakdown(salary(credit_points=3.25), rules)

    assert more.credit_points_credit == pytest.approx(fewer.credit_points_credit + rules.net_method.credit_point_value)
    assert more.income_tax == pytest.approx(fewer.income_tax - rules.net_method.credit_point_value)


def test_income_tax_is_never_negative(rules: Rules) -> None:
    result = net_breakdown(salary(base_salary=5_000, credit_points=4), rules)

    assert result.tax_before_credits < result.credit_points_credit
    assert result.income_tax == 0


# --- Pension and training fund ----------------------------------------------

def test_pension_and_training_fund_are_on_gross_income(rules: Rules) -> None:
    result = net_breakdown(salary(base_salary=10_000, commissions=2_000, pension_rate=0.06, training_fund_rate=0.025), rules)

    assert result.pension == pytest.approx(12_000 * 0.06)
    assert result.training_fund == pytest.approx(12_000 * 0.025)


def test_pension_credit_is_capped(rules: Rules) -> None:
    credit = rules.net_method.pension_credit
    cap = credit.rate * credit.max_contribution_rate * credit.income_ceiling

    high_earner = net_breakdown(salary(base_salary=30_000, pension_rate=0.07), rules)

    assert high_earner.pension > credit.max_contribution_rate * credit.income_ceiling
    assert high_earner.pension_credit == pytest.approx(cap)


def test_pension_credit_below_cap_follows_contribution(rules: Rules) -> None:
    credit = rules.net_method.pension_credit

    low_contribution = net_breakdown(salary(base_salary=20_000, pension_rate=0.02), rules)

    assert low_contribution.pension_credit == pytest.approx(credit.rate * 20_000 * 0.02)


def test_pension_credit_for_income_below_ceiling(rules: Rules) -> None:
    credit = rules.net_method.pension_credit
    income = credit.income_ceiling / 2

    result = net_breakdown(salary(base_salary=income, pension_rate=0.07), rules)

    assert result.pension_credit == pytest.approx(credit.rate * credit.max_contribution_rate * income)


def test_training_fund_is_a_net_deduction_only(rules: Rules) -> None:
    without = net_breakdown(salary(training_fund_rate=0), rules)
    with_fund = net_breakdown(salary(training_fund_rate=0.025), rules)

    assert with_fund.income_tax == without.income_tax
    assert with_fund.net == pytest.approx(without.net - with_fund.training_fund)


# --- National insurance and health ------------------------------------------

def test_two_tier_contribution(rules: Rules) -> None:
    tiers = rules.net_method.national_insurance

    assert two_tier_contribution(tiers.threshold, tiers) == pytest.approx(tiers.threshold * tiers.low_rate)
    assert two_tier_contribution(tiers.threshold + 1_000, tiers) == pytest.approx(
        tiers.threshold * tiers.low_rate + 1_000 * tiers.high_rate
    )
    assert two_tier_contribution(tiers.ceiling + 10_000, tiers) == two_tier_contribution(tiers.ceiling, tiers)


def test_insurance_uses_taxable_income(rules: Rules) -> None:
    result = net_breakdown(salary(use_value=4_000), rules)
    method = rules.net_method

    assert result.national_insurance == pytest.approx(two_tier_contribution(result.taxable_income, method.national_insurance))
    assert result.health == pytest.approx(two_tier_contribution(result.taxable_income, method.health))


# --- Net ---------------------------------------------------------------------

def test_net_is_gross_minus_every_deduction(rules: Rules) -> None:
    r = net_breakdown(salary(commissions=1_000, use_value=4_000, participation=300), rules)

    assert r.net == pytest.approx(
        r.gross_income - r.income_tax - r.national_insurance - r.health - r.pension - r.training_fund - r.participation
        - r.family_share
    )
    assert r.total_deductions == pytest.approx(r.gross_income - r.net)


def test_no_rounding_during_calculation(rules: Rules) -> None:
    result = net_breakdown(salary(base_salary=18_500, use_value=4_070, participation=300, credit_points=2.75), rules)

    assert result.net != round(result.net)


def test_breakdown_is_immutable(rules: Rules) -> None:
    with pytest.raises(FrozenInstanceError):
        net_breakdown(salary(), rules).net = 0  # type: ignore[misc]


def test_default_rules_are_the_loaded_rules(rules: Rules) -> None:
    assert net_breakdown(salary()) == net_breakdown(salary(), rules)


# --- Refusals ----------------------------------------------------------------

@pytest.mark.parametrize(
    "changes",
    [
        {"base_salary": -1}, {"use_value": -100}, {"participation": -50}, {"maintenance": -1_500}, {"family_share": -10},
        {"pension_rate": 6}, {"training_fund_rate": -0.1},
    ],
)
def test_invalid_input_raises(rules: Rules, changes) -> None:
    with pytest.raises(ValueError):
        net_breakdown(salary(**changes), rules)


def test_unsupported_rules_setting_is_refused(rules: Rules) -> None:
    yearly = replace(rules, net_method=replace(rules.net_method, basis="yearly"))

    with pytest.raises(NotImplementedError, match="netMethod.basis"):
        net_breakdown(salary(), yearly)


def test_unsupported_waiver_setting_is_refused(rules: Rules) -> None:
    pensionable_maintenance = replace(rules, waiver=replace(rules.waiver, maintenance_is_pensionable=True))

    with pytest.raises(NotImplementedError, match="waiver.maintenanceIsPensionable"):
        net_breakdown(salary(maintenance=1_500), pensionable_maintenance)
