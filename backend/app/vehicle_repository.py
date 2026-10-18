from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.csv_columns import COLUMN_TO_FIELD, CSV_DATE_FORMAT, CSV_ENCODING
from app.errors import AmbiguousVehicleError, VehicleNotFoundError
from app.models import VehicleRecord
from app.rules import load_rules

DEFAULT_CSV_PATH = Path(__file__).parent.parent / "data" / "use_value_2026.csv"

# Values of the lookup key fields, in the order rules.json valueOfUse.key lists them.
VehicleKey = tuple[int, ...]


class VehicleRepository:
    """Reads the use-value CSV once and keeps the records in memory.

    The lookup key columns come from rules.json (valueOfUse.key) unless key_columns is given.
    """

    def __init__(self, csv_path: Path = DEFAULT_CSV_PATH, key_columns: Sequence[str] | None = None) -> None:
        if key_columns is None:
            key_columns = load_rules().value_of_use.key
        self._csv_path = csv_path
        self._key_fields = tuple(COLUMN_TO_FIELD[column] for column in key_columns)
        self._records: list[VehicleRecord] | None = None
        self._by_key: dict[VehicleKey, list[VehicleRecord]] | None = None

    @property
    def key_fields(self) -> tuple[str, ...]:
        """VehicleRecord field names that make up the lookup key."""
        return self._key_fields

    def get_all(self) -> list[VehicleRecord]:
        if self._records is None:
            self._records = self._load()
        return self._records

    def find_exact(self, **key: int) -> VehicleRecord:
        """Return the single record matching the key exactly, or raise.

        Takes exactly the key fields as keyword arguments, e.g. tax_year=2026, model_code=290.
        """
        if set(key) != set(self._key_fields):
            raise TypeError(f"find_exact() takes exactly the key fields {list(self._key_fields)}, got {sorted(key)}")
        values: VehicleKey = tuple(key[field] for field in self._key_fields)
        matches = self._index().get(values, [])

        if not matches:
            raise VehicleNotFoundError(f"No vehicle found for {self._describe(values)}")
        if len(matches) > 1:
            raise AmbiguousVehicleError(f"{len(matches)} vehicles match {self._describe(values)}")
        return matches[0]

    def _index(self) -> dict[VehicleKey, list[VehicleRecord]]:
        if self._by_key is None:
            by_key: dict[VehicleKey, list[VehicleRecord]] = defaultdict(list)
            for record in self.get_all():
                by_key[self._key_of(record)].append(record)
            self._by_key = dict(by_key)
        return self._by_key

    def _key_of(self, record: VehicleRecord) -> VehicleKey:
        return tuple(getattr(record, field) for field in self._key_fields)

    def _describe(self, values: VehicleKey) -> str:
        return ", ".join(f"{field}={value}" for field, value in zip(self._key_fields, values))

    def _load(self) -> list[VehicleRecord]:
        df = pd.read_csv(self._csv_path, encoding=CSV_ENCODING)
        df.columns = df.columns.str.strip()

        missing = set(COLUMN_TO_FIELD) - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")

        df = df[list(COLUMN_TO_FIELD)].rename(columns=COLUMN_TO_FIELD)
        return [_to_record(row) for row in df.to_dict("records")]


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
