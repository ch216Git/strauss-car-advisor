import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from app.errors import ConfigError
from app.lookup_config import DEFAULT_LOOKUP_PATH, LookupConfig, load_lookup_config

# The two values agreed for Step 3; hand-copied as an oracle for the parsing.
AGREED_TAX_YEAR = 2026
AGREED_CATALOGUE_VEHICLE_TYPE = 1


def raw_lookup() -> dict:
    return json.loads(DEFAULT_LOOKUP_PATH.read_text(encoding="utf-8"))


def lookup_error(tmp_path: Path, mutate) -> ConfigError:
    data = raw_lookup()
    mutate(data)
    path = tmp_path / "lookup.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        load_lookup_config(path)
    assert info.value.file == path.resolve()
    return info.value


def test_real_lookup_file_loads() -> None:
    lookup = load_lookup_config()

    assert lookup == LookupConfig(tax_year=AGREED_TAX_YEAR, catalogue_vehicle_type=AGREED_CATALOGUE_VEHICLE_TYPE)
    assert load_lookup_config() is lookup


def test_vehicle_type_is_documented_as_dataset_assumption() -> None:
    note = raw_lookup()["catalogueVehicleTypeNote"]

    assert "DATASET ASSUMPTION, NOT A BUSINESS RULE" in note


def test_lookup_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        load_lookup_config().tax_year = 2025  # type: ignore[misc]


@pytest.mark.parametrize(
    "mutate, path, expected, found",
    [
        (lambda d: d.pop("taxYear"), "taxYear", "a required key", "nothing (key is missing)"),
        (lambda d: d.pop("catalogueVehicleType"), "catalogueVehicleType", "a required key", "nothing (key is missing)"),
        (lambda d: d.update(taxYear="2026"), "taxYear", "an integer", '"2026"'),
        (lambda d: d.update(catalogueVehicleType=0), "catalogueVehicleType", "an integer > 0", "0"),
    ],
)
def test_invalid_lookup_raises(tmp_path: Path, mutate, path: str, expected: str, found: str) -> None:
    error = lookup_error(tmp_path, mutate)

    assert (error.path, error.expected, error.found) == (path, expected, found)
