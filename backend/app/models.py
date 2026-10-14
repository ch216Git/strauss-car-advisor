from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class VehicleRecord:
    """A single row from the use-value (שווי שימוש) dataset."""

    tax_year: int
    registration_year: int
    vehicle_type_code: int
    manufacturer_code: int
    manufacturer: str
    model_code: int
    model_description: str
    fuel_type_code: int
    commercial_name: str
    is_automatic: bool
    # Observed in the data: 0-regular, 1-hybrid, 2-plug-in, 3-electric.
    # The CSV header ("1-חשמלי.2-היברידי.3-פלג-אין") does not match the actual values.
    propulsion_code: int
    green_score: int
    pollution_level: int
    engine_volume_cc: int
    weight_kg: int
    effective_date: date
    list_price: int
    adjusted_price: int
    use_value: int
