import csv
from pathlib import Path

import pytest

from app.csv_columns import COLUMN_TO_FIELD, CSV_ENCODING
from app.errors import AmbiguousVehicleError, VehicleNotFoundError
from app.rules import load_rules
from app.vehicle_repository import DEFAULT_CSV_PATH, VehicleRepository

# Rows copied by hand from use_value_2026.csv; they act as an independent oracle,
# so a wrong column mapping (e.g. reading list price as use value) is caught.
KNOWN_VEHICLES = [
    # (key, manufacturer, commercial_name, use_value)
    ((2026, 2010, 1, 19, 290), "אאודי", "S5", 14800),
    ((2026, 2012, 1, 650, 2041), "סובארו יפן", "FORESTER", 5920),
    ((2026, 2021, 1, 588, 346), "מזדה יפן", "MAZDA CX-30", 5120),
    ((2026, 2026, 2, 1740, 5), "פולקסווגן טורק", "CARAVELLE", 9180),
]
AUDI_S5_KEY = KNOWN_VEHICLES[0][0]
NONEXISTENT_MODEL_CODE = 999_999


def find(repo: VehicleRepository, key: tuple[int, int, int, int, int]):
    tax_year, registration_year, vehicle_type_code, manufacturer_code, model_code = key
    return repo.find_exact(
        tax_year=tax_year,
        registration_year=registration_year,
        vehicle_type_code=vehicle_type_code,
        manufacturer_code=manufacturer_code,
        model_code=model_code,
    )


def read_raw_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding=CSV_ENCODING, newline="") as f:
        header, *rows = csv.reader(f)
    return header, [row for row in rows if row]


@pytest.fixture(scope="module")
def repo() -> VehicleRepository:
    return VehicleRepository()


# --- Loading -----------------------------------------------------------------

def test_csv_loads_successfully(repo: VehicleRepository) -> None:
    records = repo.get_all()

    assert records, "repository returned no records"


def test_record_count_matches_rows_in_csv(repo: VehicleRepository) -> None:
    _, raw_rows = read_raw_rows(DEFAULT_CSV_PATH)

    assert len(raw_rows) > 50_000  # sanity: the 2026 dataset has tens of thousands of rows
    assert len(repo.get_all()) == len(raw_rows)


# --- Exact match -------------------------------------------------------------

@pytest.mark.parametrize("key, manufacturer, commercial_name, use_value", KNOWN_VEHICLES)
def test_valid_key_returns_the_matching_vehicle(
    repo: VehicleRepository, key, manufacturer: str, commercial_name: str, use_value: int
) -> None:
    vehicle = find(repo, key)

    assert (
        vehicle.tax_year,
        vehicle.registration_year,
        vehicle.vehicle_type_code,
        vehicle.manufacturer_code,
        vehicle.model_code,
    ) == key
    assert vehicle.manufacturer == manufacturer
    assert vehicle.commercial_name == commercial_name


@pytest.mark.parametrize("key, manufacturer, commercial_name, use_value", KNOWN_VEHICLES)
def test_use_value_matches_csv(
    repo: VehicleRepository, key, manufacturer: str, commercial_name: str, use_value: int
) -> None:
    assert find(repo, key).use_value == use_value


# --- No match ----------------------------------------------------------------

def test_unknown_key_raises_not_found(repo: VehicleRepository) -> None:
    key = (*AUDI_S5_KEY[:4], NONEXISTENT_MODEL_CODE)

    with pytest.raises(VehicleNotFoundError):
        find(repo, key)


def test_lookup_key_comes_from_rules(repo: VehicleRepository) -> None:
    key_columns = load_rules().value_of_use.key

    assert repo.key_fields == tuple(COLUMN_TO_FIELD[column] for column in key_columns)


def test_partial_key_is_rejected(repo: VehicleRepository) -> None:
    tax_year, registration_year, vehicle_type_code, manufacturer_code, _ = AUDI_S5_KEY

    with pytest.raises(TypeError, match="model_code"):
        repo.find_exact(
            tax_year=tax_year,
            registration_year=registration_year,
            vehicle_type_code=vehicle_type_code,
            manufacturer_code=manufacturer_code,
        )


def test_custom_key_columns_change_the_lookup(tmp_path: Path) -> None:
    # A repository built with a different key looks up by that key only.
    header, rows = read_raw_rows(DEFAULT_CSV_PATH)
    csv_path = tmp_path / "two_rows.csv"
    write_rows(csv_path, header, [rows[0], rows[1]])
    repo = VehicleRepository(csv_path, key_columns=["קוד תוצר", "קוד דגם"])
    _, _, _, manufacturer_code, model_code = AUDI_S5_KEY

    assert repo.find_exact(manufacturer_code=manufacturer_code, model_code=model_code).commercial_name == "S5"
    with pytest.raises(TypeError):
        find(repo, AUDI_S5_KEY)


def test_no_fallback_to_commercial_name(repo: VehicleRepository) -> None:
    # Same manufacturer as the Audi S5 but a wrong model code: the S5 exists in the
    # data, yet the lookup must not fall back to it (or to any other vehicle).
    key = (*AUDI_S5_KEY[:4], NONEXISTENT_MODEL_CODE)

    with pytest.raises(VehicleNotFoundError):
        find(repo, key)

    with pytest.raises(TypeError):
        repo.find_exact(commercial_name="S5")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "key",
    [
        (2026, 1900, 1, 19, 290),  # registration year not in data
        (2025, 2010, 1, 19, 290),  # tax year not in data
        (2026, 2010, 9, 19, 290),  # vehicle type not in data
    ],
)
def test_no_default_use_value_when_not_found(repo: VehicleRepository, key) -> None:
    with pytest.raises(VehicleNotFoundError):
        find(repo, key)


def test_empty_csv_raises_not_found(tmp_path: Path) -> None:
    header, _ = read_raw_rows(DEFAULT_CSV_PATH)
    csv_path = tmp_path / "empty.csv"
    write_rows(csv_path, header, [])

    with pytest.raises(VehicleNotFoundError):
        find(VehicleRepository(csv_path), AUDI_S5_KEY)


# --- Multiple matches --------------------------------------------------------

def write_rows(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding=CSV_ENCODING, newline="") as f:
        csv.writer(f).writerows([header, *rows])


def test_duplicate_rows_raise_ambiguous(tmp_path: Path) -> None:
    header, rows = read_raw_rows(DEFAULT_CSV_PATH)
    csv_path = tmp_path / "duplicate.csv"
    write_rows(csv_path, header, [rows[0], rows[0], rows[1]])

    with pytest.raises(AmbiguousVehicleError):
        find(VehicleRepository(csv_path), AUDI_S5_KEY)


def test_same_key_with_different_use_values_raises_ambiguous(tmp_path: Path) -> None:
    header, rows = read_raw_rows(DEFAULT_CSV_PATH)
    use_value_index = [COLUMN_TO_FIELD.get(h.strip()) for h in header].index("use_value")
    conflicting = list(rows[0])
    conflicting[use_value_index] = str(int(rows[0][use_value_index]) + 1)
    csv_path = tmp_path / "conflict.csv"
    write_rows(csv_path, header, [rows[0], conflicting])

    with pytest.raises(AmbiguousVehicleError):
        find(VehicleRepository(csv_path), AUDI_S5_KEY)
