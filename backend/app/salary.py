"""Deterministic monthly net-pay calculation, driven by rules.json netMethod.

No rounding happens here (rules netMethod.rounding.during = "none"); round only for display.
"""

from dataclasses import dataclass

from app.rules import ContributionRules, Rules, TaxBracket, load_rules

# The netMethod settings this module implements; anything else in rules.json is refused, not ignored.
SUPPORTED_SETTINGS = {
    "netMethod.basis": "monthly",
    "netMethod.contributionsBase": "gross",
    "netMethod.participation.treatment": "net_deduction",
    "netMethod.participation.reducesValueOfUse": False,
    "netMethod.rounding.during": "none",
    "waiver.maintenanceIsTaxable": True,
    "waiver.maintenanceIsPensionable": False,
}

# The only electric.method implemented; checked only when an electric car is calculated.
ELECTRIC_GROSS_UP = "gross_up"


@dataclass(frozen=True, slots=True)
class SalaryInput:
    """One employee's month. Amounts in ₪ per month; rates as fractions (0.06 = 6%)."""

    base_salary: float
    commissions: float
    bonus: float
    pension_rate: float  # employee pension contribution
    training_fund_rate: float  # employee training-fund (קרן השתלמות) contribution
    credit_points: float
    use_value: float  # vehicle use value (שווי שימוש); 0 without a company car
    participation: float  # employee's vehicle participation; 0 without a company car
    maintenance: float  # car maintenance (אחזקת רכב) for waiving the car: the occupation's amount; else 0
    electric_car: bool  # the company car is electric: adds the grossed-up electricity reimbursement
    family_share: float  # employee's family-track share (see family_track.py); 0 outside the family track


@dataclass(frozen=True, slots=True)
class NetBreakdown:
    """Every step of the calculation, unrounded, so it can be shown and explained."""

    pensionable_income: float  # base salary + commissions + bonus; base for pension and training fund
    maintenance: float  # taxable, not pensionable
    electric_gross: float  # grossed-up electricity reimbursement; taxable, not pensionable
    gross_income: float  # pensionable income + maintenance + electric gross: everything paid
    use_value: float
    taxable_income: float  # gross income + use value; base for income tax, national insurance and health
    tax_before_credits: float
    credit_points_credit: float
    pension_credit: float
    income_tax: float  # never below 0
    national_insurance: float
    health: float
    pension: float
    training_fund: float
    participation: float
    family_share: float  # net deduction only
    net: float  # gross income - tax - national insurance - health - pension - training fund - participation - family share

    @property
    def total_deductions(self) -> float:
        return self.gross_income - self.net


def net_breakdown(salary: SalaryInput, rules: Rules | None = None) -> NetBreakdown:
    rules = rules or load_rules()
    _check_supported(rules)
    _check_input(salary)
    electric_gross = electric_gross_up(salary, rules) if salary.electric_car else 0.0
    return _breakdown(salary, rules, electric_gross)


def electric_gross_up(salary: SalaryInput, rules: Rules) -> float:
    """The gross addition G whose net effect is exactly electric.netBonus.

    Solves net(G) - net(0) = netBonus after the extra income tax, national insurance and health.
    Net rises with G (each extra shekel loses less than a shekel), but the slope changes at
    every bracket and threshold, so G is found by bisection rather than a single formula.
    """
    if rules.electric.method != ELECTRIC_GROSS_UP:
        raise NotImplementedError(
            f"rules.json electric.method {rules.electric.method!r} is not supported; only {ELECTRIC_GROSS_UP!r}"
        )
    target = rules.electric.net_bonus
    net_without = _breakdown(salary, rules, 0.0).net

    def net_gain(gross: float) -> float:
        return _breakdown(salary, rules, gross).net - net_without

    low, high = 0.0, float(target)
    for _ in range(64):
        if net_gain(high) >= target:
            break
        low, high = high, high * 2
    else:
        raise ValueError(f"No gross amount yields a net gain of {target}: marginal deductions reach 100%")

    while True:  # stops once the interval cannot shrink further in floating point
        middle = (low + high) / 2
        if middle in (low, high):
            return high
        if net_gain(middle) < target:
            low = middle
        else:
            high = middle


def _breakdown(salary: SalaryInput, rules: Rules, electric_gross: float) -> NetBreakdown:
    method = rules.net_method

    pensionable_income = salary.base_salary + salary.commissions + salary.bonus  # contributionsBase = "gross"
    # Maintenance (waiver.maintenanceIsTaxable, not ...IsPensionable) and the electric gross-up are
    # taxable but not pensionable; the gross-up treatment matches the workbook's reference employees.
    gross_income = pensionable_income + salary.maintenance + electric_gross
    pension = pensionable_income * salary.pension_rate
    training_fund = pensionable_income * salary.training_fund_rate

    taxable_income = gross_income + salary.use_value
    tax_before_credits = bracket_tax(taxable_income, method.brackets)
    credit_points_credit = salary.credit_points * method.credit_point_value
    pension_credit = _pension_credit(pension, pensionable_income, rules)
    income_tax = max(0.0, tax_before_credits - credit_points_credit - pension_credit)

    national_insurance = two_tier_contribution(taxable_income, method.national_insurance)
    health = two_tier_contribution(taxable_income, method.health)

    net = (
        gross_income
        - income_tax
        - national_insurance
        - health
        - pension
        - training_fund
        - salary.participation
        - salary.family_share
    )
    return NetBreakdown(
        pensionable_income=pensionable_income,
        maintenance=salary.maintenance,
        electric_gross=electric_gross,
        gross_income=gross_income,
        use_value=salary.use_value,
        taxable_income=taxable_income,
        tax_before_credits=tax_before_credits,
        credit_points_credit=credit_points_credit,
        pension_credit=pension_credit,
        income_tax=income_tax,
        national_insurance=national_insurance,
        health=health,
        pension=pension,
        training_fund=training_fund,
        participation=salary.participation,
        family_share=salary.family_share,
        net=net,
    )


def bracket_tax(income: float, brackets: tuple[TaxBracket, ...]) -> float:
    """Progressive tax: each bracket's rate applies to the part of income inside that bracket."""
    tax = 0.0
    lower = 0.0
    for bracket in brackets:
        upper = income if bracket.upper_limit is None else min(income, bracket.upper_limit)
        if upper > lower:
            tax += (upper - lower) * bracket.rate
        if bracket.upper_limit is None or income <= bracket.upper_limit:
            break
        lower = bracket.upper_limit
    return tax


def two_tier_contribution(income: float, tiers: ContributionRules) -> float:
    """low_rate up to the threshold, high_rate from there up to the ceiling; nothing above the ceiling."""
    capped = min(income, tiers.ceiling)
    low_part = min(capped, tiers.threshold)
    high_part = max(0.0, capped - tiers.threshold)
    return low_part * tiers.low_rate + high_part * tiers.high_rate


def _pension_credit(pension: float, pensionable_income: float, rules: Rules) -> float:
    """Credit rate on the employee's contribution, counted up to max rate x qualifying income (capped)."""
    credit = rules.net_method.pension_credit
    qualifying_income = min(pensionable_income, credit.income_ceiling)
    return credit.rate * min(pension, credit.max_contribution_rate * qualifying_income)


def _check_supported(rules: Rules) -> None:
    method = rules.net_method
    actual = {
        "netMethod.basis": method.basis,
        "netMethod.contributionsBase": method.contributions_base,
        "netMethod.participation.treatment": method.participation.treatment,
        "netMethod.participation.reducesValueOfUse": method.participation.reduces_value_of_use,
        "netMethod.rounding.during": method.rounding.during,
        "waiver.maintenanceIsTaxable": rules.waiver.maintenance_is_taxable,
        "waiver.maintenanceIsPensionable": rules.waiver.maintenance_is_pensionable,
    }
    unsupported = {key: value for key, value in actual.items() if value != SUPPORTED_SETTINGS[key]}
    if unsupported:
        raise NotImplementedError(f"rules.json settings not supported by the net calculation: {unsupported}")


def _check_input(salary: SalaryInput) -> None:
    amounts = {
        "base_salary": salary.base_salary,
        "commissions": salary.commissions,
        "bonus": salary.bonus,
        "credit_points": salary.credit_points,
        "use_value": salary.use_value,
        "participation": salary.participation,
        "maintenance": salary.maintenance,
        "family_share": salary.family_share,
    }
    negative = {name: value for name, value in amounts.items() if value < 0}
    if negative:
        raise ValueError(f"Salary input must not be negative: {negative}")
    rates = {"pension_rate": salary.pension_rate, "training_fund_rate": salary.training_fund_rate}
    out_of_range = {name: value for name, value in rates.items() if not 0 <= value <= 1}
    if out_of_range:
        raise ValueError(f"Salary rates must be fractions between 0 and 1 (0.06 = 6%): {out_of_range}")
    if salary.maintenance and (salary.use_value or salary.participation or salary.electric_car or salary.family_share):
        raise ValueError(
            "Car maintenance is paid instead of a company car; "
            "use_value, participation, electric_car and family_share must be unset"
        )
