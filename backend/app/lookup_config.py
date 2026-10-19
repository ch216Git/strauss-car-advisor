"""The only place that knows the lookup.json keys.

Dataset settings for finding catalogue cars in the use-value CSV; not business rules.
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

from app.config_file import Section, load_json_config, parse_positive_int

DEFAULT_LOOKUP_PATH = Path(__file__).parent.parent / "config" / "lookup.json"


@dataclass(frozen=True, slots=True)
class LookupConfig:
    tax_year: int
    # Dataset assumption (see catalogueVehicleTypeNote in lookup.json): the catalogue has no vehicle-type column.
    catalogue_vehicle_type: int


def load_lookup_config(path: Path = DEFAULT_LOOKUP_PATH) -> LookupConfig:
    """Read and parse the lookup file; later calls with the same path return the same object."""
    return _load(path.resolve())


@cache
def _load(path: Path) -> LookupConfig:
    return load_json_config(path, _parse)


def _parse(root: Section) -> LookupConfig:
    return LookupConfig(
        tax_year=root.parse("taxYear", parse_positive_int),
        catalogue_vehicle_type=root.parse("catalogueVehicleType", parse_positive_int),
    )
