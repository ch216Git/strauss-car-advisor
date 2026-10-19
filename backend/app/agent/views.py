"""Screen-ready views of a conversation: cars, the last comparison, details given (presentation only).

Built only from the engine's own results and data: amounts are the comparison's own figures rounded
for display with display_shekels (as the tools show them), the electric benefit is rules.json's,
car facts come from the catalogue and the Tax Authority CSV. Nothing is calculated or decided here.
"""

from dataclasses import dataclass, fields

from app.agent.state import EmployeeDetails, ShownComparison
from app.agent.tools import ToolContext, _car_summary
from app.comparison import ELECTRIC_DRIVE, CompanyCar, SituationDetails, display_shekels
from app.family_track import is_family_track_car

# The comparison items a screen explains, in the order the replies explain them (app.agent.render);
# the electric reimbursement is shown apart, by its net amount (ElectricBenefitView), never by its gross.
EXPLAINED_CHANGES = ("use_value", "maintenance", "participation", "family_share")


@dataclass(frozen=True, slots=True)
class CarView:
    car_id: str
    manufacturer: str  # Tax Authority CSV
    commercial_name: str  # Tax Authority CSV
    registration_year: int
    group: str
    electric: bool
    family_track: bool


@dataclass(frozen=True, slots=True)
class SideView:
    kind: str  # "company_car" or "waiver"
    car: CarView | None
    net: int
    use_value: int
    participation: int
    participation_source: str | None  # "stated" or "catalogue_default"; None for a waiver
    maintenance: int
    family_share: int | None  # family track only: the employee's net share of the gap


@dataclass(frozen=True, slots=True)
class ChangeView:
    item: str  # one of EXPLAINED_CHANGES
    current: int
    alternative: int


@dataclass(frozen=True, slots=True)
class ElectricBenefitView:
    net: int  # rules.json electric.netBonus: what the employee receives net
    grossed_up: bool  # rules.json electric.method == "gross_up": the employer bears the tax
    applies_to: str  # "alternative" (gained) or "current" (left)


@dataclass(frozen=True, slots=True)
class ComparisonView:
    status: str  # ComparisonStatus value: "estimate" or "confirmed"
    estimated_fields: tuple[str, ...]
    current: SideView
    alternative: SideView
    net_difference: int
    changes: tuple[ChangeView, ...]
    electric_benefit: ElectricBenefitView | None


def car_view(car_id: str | None, ctx: ToolContext) -> CarView | None:
    car = ctx.catalog.cars.get(car_id or "")
    if car is None:
        return None
    summary = _car_summary(car, ctx)
    return CarView(
        car_id=car.car_id,
        manufacturer=summary.manufacturer,
        commercial_name=summary.commercial_name,
        registration_year=summary.registration_year,
        group=summary.group,
        electric=summary.drive == ELECTRIC_DRIVE,
        family_track=is_family_track_car(car, ctx.rules),
    )


def comparison_view(shown: ShownComparison, ctx: ToolContext) -> ComparisonView:
    result = shown.result
    displayed = {
        change.item: (display_shekels(change.current), display_shekels(change.alternative)) for change in result.changes
    }
    changes = tuple(
        ChangeView(item, *displayed[item])
        for item in EXPLAINED_CHANGES
        if item in displayed and displayed[item][0] != displayed[item][1]
    )
    electric = None
    gross = displayed.get("electric_gross")
    if gross is not None and gross[0] != gross[1]:  # the reimbursement is gained or left (as the replies say)
        electric = ElectricBenefitView(
            net=display_shekels(ctx.rules.electric.net_bonus),
            grossed_up=ctx.rules.electric.method == "gross_up",
            applies_to="alternative" if result.alternative_situation.electric else "current",
        )
    return ComparisonView(
        status=shown.status.value,
        estimated_fields=shown.estimated_fields,
        current=_side(result.current_situation, display_shekels(result.current.net), ctx),
        alternative=_side(result.alternative_situation, display_shekels(result.alternative.net), ctx),
        net_difference=result.net_difference,
        changes=changes,
        electric_benefit=electric,
    )


def provided_details(details: EmployeeDetails) -> tuple[str, ...]:
    """Names of the employee details given so far (never their values)."""
    return tuple(f.name for f in fields(details) if getattr(details, f.name) is not None)


def current_car_id(details: EmployeeDetails) -> str | None:
    return details.current.car_id if isinstance(details.current, CompanyCar) else None


def _side(details: SituationDetails, net: int, ctx: ToolContext) -> SideView:
    return SideView(
        kind=details.kind,
        car=car_view(details.car_id, ctx),
        net=net,
        use_value=details.use_value,
        participation=display_shekels(details.participation),
        participation_source=details.participation_source,
        maintenance=details.maintenance,
        family_share=display_shekels(details.family_track.employee_share) if details.family_track else None,
    )
