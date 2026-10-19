"""Check every vehicle in the challenge catalog against the Tax Authority CSV.

Read-only: reports mismatches, never fixes data.
Run from backend/:  python -m scripts.verify_catalog
"""

from dataclasses import dataclass
from pathlib import Path

import openpyxl

from app.catalog import find_workbook
from app.errors import AmbiguousVehicleError, VehicleNotFoundError
from app.lookup_config import load_lookup_config
from app.models import VehicleRecord
from app.rules import load_rules
from app.vehicle_repository import VehicleRepository
from app.workbook_columns import CATALOG_COLUMN_TO_FIELD, CATALOG_CROSS_CHECK_COLUMNS, CATALOG_SHEET

# Deliberately tries every vehicle type in the CSV, not just lookup.json catalogueVehicleType:
# this script is what checks that dataset assumption.
CANDIDATE_VEHICLE_TYPE_CODES = (1, 2)


@dataclass
class CatalogVehicle:
    vehicle_id: str
    manufacturer: str
    commercial_name: str
    manufacturer_code: int
    model_code: int
    registration_year: int
    propulsion: str
    list_price: int
    use_value: int


@dataclass
class CheckResult:
    catalog: CatalogVehicle
    record: VehicleRecord | None
    vehicle_type_code: int | None
    error: str | None


def read_catalog(path: Path) -> list[CatalogVehicle]:
    sheet = openpyxl.load_workbook(path, read_only=True, data_only=True)[CATALOG_SHEET]
    header, *rows = sheet.iter_rows(values_only=True)
    column_to_field = {**CATALOG_COLUMN_TO_FIELD, **CATALOG_CROSS_CHECK_COLUMNS}
    col = {column_to_field[name]: i for i, name in enumerate(header) if name in column_to_field}
    return [
        CatalogVehicle(
            vehicle_id=row[col["car_id"]],
            manufacturer=row[col["manufacturer"]],
            commercial_name=row[col["commercial_name"]],
            manufacturer_code=int(row[col["manufacturer_code"]]),
            model_code=int(row[col["model_code"]]),
            registration_year=int(row[col["registration_year"]]),
            propulsion=row[col["propulsion"]],
            list_price=int(row[col["list_price"]]),
            use_value=int(row[col["use_value"]]),
        )
        for row in rows
        if row[col["car_id"]]
    ]


def check(repo: VehicleRepository, vehicle: CatalogVehicle, tax_year: int) -> CheckResult:
    found: list[tuple[int, VehicleRecord]] = []
    for type_code in CANDIDATE_VEHICLE_TYPE_CODES:
        try:
            record = repo.find_exact(
                tax_year=tax_year,
                registration_year=vehicle.registration_year,
                vehicle_type_code=type_code,
                manufacturer_code=vehicle.manufacturer_code,
                model_code=vehicle.model_code,
            )
        except VehicleNotFoundError:
            continue
        except AmbiguousVehicleError as e:
            return CheckResult(vehicle, None, type_code, f"ambiguous: {e}")
        found.append((type_code, record))

    if not found:
        return CheckResult(vehicle, None, None, "not found in Tax Authority CSV")
    if len(found) > 1:
        candidates = "; ".join(
            f"type {t}: {r.manufacturer} {r.commercial_name}, use value {r.use_value:,}, list price {r.list_price:,}"
            for t, r in found
        )
        return CheckResult(vehicle, None, None, f"AMBIGUOUS - key matches several vehicle types -> {candidates}")
    type_code, record = found[0]
    return CheckResult(vehicle, record, type_code, None)


def propulsion_codes() -> dict[str, int]:
    """Catalog propulsion label -> CSV code, from rules.json valueOfUse.driveField."""
    return {name: code for code, name in load_rules().value_of_use.drive_field.items()}


def anomalies(result: CheckResult, propulsion_codes: dict[str, int]) -> list[str]:
    if result.record is None:
        return [result.error or "not found"]
    c, r = result.catalog, result.record
    issues = []
    if c.use_value != r.use_value:
        issues.append(f"use value: catalog {c.use_value:,} vs CSV {r.use_value:,} (diff {c.use_value - r.use_value:+,})")
    if c.list_price != r.list_price:
        issues.append(f"list price: catalog {c.list_price:,} vs CSV {r.list_price:,} (diff {c.list_price - r.list_price:+,})")
    if propulsion_codes.get(c.propulsion) != r.propulsion_code:
        issues.append(f"propulsion: catalog '{c.propulsion}' vs CSV code {r.propulsion_code}")
    if c.manufacturer != r.manufacturer:
        issues.append(f"manufacturer name: catalog '{c.manufacturer}' vs CSV '{r.manufacturer}'")
    if c.commercial_name != r.commercial_name:
        issues.append(f"commercial name: catalog '{c.commercial_name}' vs CSV '{r.commercial_name}'")
    return issues


def main() -> None:
    catalog_path = find_workbook()
    tax_year = load_lookup_config().tax_year
    repo = VehicleRepository()
    results = [check(repo, v, tax_year) for v in read_catalog(catalog_path)]

    print(f"Catalog: {catalog_path.name} / {CATALOG_SHEET}")
    print(f"Lookup key: tax_year={tax_year}, vehicle_type in {CANDIDATE_VEHICLE_TYPE_CODES}\n")

    columns = ["Vehicle", "Product Code", "Model Code", "Found", "Catalog Use Value", "Tax CSV Use Value", "Match"]
    rows = []
    for res in results:
        c, r = res.catalog, res.record
        rows.append([
            f"{c.vehicle_id} {c.commercial_name}",
            str(c.manufacturer_code),
            str(c.model_code),
            "yes" if r else ("AMBIGUOUS" if res.error and res.error.startswith("AMBIGUOUS") else "NO"),
            f"{c.use_value:,}",
            f"{r.use_value:,}" if r else "-",
            ("yes" if r.use_value == c.use_value else "NO") if r else "-",
        ])
    widths = [max(len(x) for x in col) for col in zip(columns, *rows)]
    for line in [columns, ["-" * w for w in widths], *rows]:
        print(" | ".join(cell.ljust(w) for cell, w in zip(line, widths)))

    found = [res for res in results if res.record]
    use_value_matches = [res for res in found if res.record.use_value == res.catalog.use_value]
    type_codes = sorted({res.vehicle_type_code for res in found})

    print("\nSummary")
    print(f"  Checked:            {len(results)}")
    print(f"  Found:              {len(found)}  (vehicle type codes matched: {type_codes})")
    ambiguous = sum(1 for res in results if res.error and res.error.startswith("AMBIGUOUS"))
    print(f"  Ambiguous:          {ambiguous}")
    print(f"  Not found:          {len(results) - len(found) - ambiguous}")
    print(f"  Use value matches:  {len(use_value_matches)} / {len(found)}")
    print(f"  List price matches: {sum(res.record.list_price == res.catalog.list_price for res in found)} / {len(found)}")

    print("\nAnomalies")
    codes = propulsion_codes()
    any_issue = False
    for res in results:
        for issue in anomalies(res, codes):
            any_issue = True
            print(f"  {res.catalog.vehicle_id} {res.catalog.commercial_name}: {issue}")
    if not any_issue:
        print("  none")


if __name__ == "__main__":
    main()
