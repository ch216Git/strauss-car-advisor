from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.csv_columns import COLUMN_TO_FIELD, CSV_DATE_FORMAT, CSV_ENCODING
from app.errors import AmbiguousVehicleError, VehicleNotFoundError
from app.models import VehicleRecord

DEFAULT_CSV_PATH = Path(__file__).parent.parent / "data" / "use_value_2026.csv"

# (tax_year, registration_year, vehicle_type_code, manufacturer_code, model_code)
VehicleKey = tuple[int, int, int, int, int]


class VehicleRepository:
    """Reads the use-value CSV once and keeps the records in memory."""

    def __init__(self, csv_path: Path = DEFAULT_CSV_PATH) -> None:
        self._csv_path = csv_path
        self._records: list[VehicleRecord] | None = None
        self._by_key: dict[VehicleKey, list[VehicleRecord]] | None = None

    def get_all(self) -> list[VehicleRecord]:
        if self._records is None:
            self._records = self._load()
        return self._records

    def find_exact(
        self,
        *,
        tax_year: int,
        registration_year: int,
        vehicle_type_code: int,
        manufacturer_code: int,
        model_code: int,
    ) -> VehicleRecord:
        """Return the single record matching the key exactly, or raise."""
        key: VehicleKey = (tax_year, registration_year, vehicle_type_code, manufacturer_code, model_code)
        matches = self._index().get(key, [])

        if not matches:
            raise VehicleNotFoundError(f"No vehicle found for {_describe(key)}")
        if len(matches) > 1:
            raise AmbiguousVehicleError(f"{len(matches)} vehicles match {_describe(key)}")
        return matches[0]

    def _index(self) -> dict[VehicleKey, list[VehicleRecord]]:
        if self._by_key is None:
            by_key: dict[VehicleKey, list[VehicleRecord]] = defaultdict(list)
            for record in self.get_all():
                by_key[_key_of(record)].append(record)
            self._by_key = dict(by_key)
        return self._by_key

    def _load(self) -> list[VehicleRecord]:
        df = pd.read_csv(self._csv_path, encoding=CSV_ENCODING)
        df.columns = df.columns.str.strip()

        missing = set(COLUMN_TO_FIELD) - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")

        df = df[list(COLUMN_TO_FIELD)].rename(columns=COLUMN_TO_FIELD)
        return [_to_record(row) for row in df.to_dict("records")]


def _key_of(record: VehicleRecord) -> VehicleKey:
    return (
        record.tax_year,
        record.registration_year,
        record.vehicle_type_code,
        record.manufacturer_code,
        record.model_code,
    )


def _describe(key: VehicleKey) -> str:
    tax_year, registration_year, vehicle_type_code, manufacturer_code, model_code = key
    return (
        f"tax_year={tax_year}, registration_year={registration_year}, "
        f"vehicle_type_code={vehicle_type_code}, manufacturer_code={manufacturer_code}, "
        f"model_code={model_code}"
    )


def _to_record(row: dict) -> VehicleRecord:
    return VehicleRecord(
        **{
            **row,
            "manufacturer": row["manufacturer"].strip(),
            "model_description": row["model_description"].strip(),
            "commercial_name": row["commercial_name"].strip(),
            "is_automatic": row["is_automatic"] == 1,
            "effective_date": datetime.strptime(row["effective_date"], CSV_DATE_FORMAT).date(),
        }
    )
