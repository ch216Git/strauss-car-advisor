"""Check every vehicle in the challenge catalog against the Tax Authority CSV.

Read-only: reports mismatches, never fixes data.
Run from backend/:  python -m scripts.verify_catalog
"""

from dataclasses import dataclass
from pathlib import Path

import openpyxl

from app.errors import AmbiguousVehicleError, VehicleNotFoundError
from app.models import VehicleRecord
from app.vehicle_repository import VehicleRepository

DATA_DIR = Path(__file__).parent.parent / "data"
CATALOG_SHEET = "קטלוג רכבים"

# The workbook states its use values come from the Tax Authority's 2026 dataset.
TAX_YEAR = 2026
# The catalog has no vehicle-type column; these are the only codes present in the CSV.
CANDIDATE_VEHICLE_TYPE_CODES = (1, 2)

# Catalog propulsion label -> CSV code, as observed in the data (see VehicleRecord.propulsion_code).
PROPULSION_CODES = {"בנזין": 0, "היברידי": 1, "פלאג-אין": 2, "חשמלי": 3}


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


def find_catalog_file() -> Path:
    matches = [p for p in DATA_DIR.glob("*.xlsx") if CATALOG_SHEET in openpyxl.load_workbook(p, read_only=True).sheetnames]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one workbook with sheet '{CATALOG_SHEET}' in {DATA_DIR}, found {len(matches)}")
    return matches[0]


def read_catalog(path: Path) -> list[CatalogVehicle]:
    sheet = openpyxl.load_workbook(path, read_only=True, data_only=True)[CATALOG_SHEET]
    header, *rows = sheet.iter_rows(values_only=True)
    col = {name: i for i, name in enumerate(header)}
    return [
        CatalogVehicle(
            vehicle_id=row[col["מזהה"]],
            manufacturer=row[col["תוצר"]],
            commercial_name=row[col["דגם (כינוי מסחרי)"]],
            manufacturer_code=int(row[col["קוד תוצר"]]),
            model_code=int(row[col["קוד דגם"]]),
            registration_year=int(row[col["שנת רישום"]]),
            propulsion=row[col["הנעה"]],
            list_price=int(row[col["מחיר מחירון"]]),
            use_value=int(row[col["שווי שימוש חודשי"]]),
        )
        for row in rows
        if row[col["מזהה"]]
    ]


def check(repo: VehicleRepository, vehicle: CatalogVehicle) -> CheckResult:
    found: list[tuple[int, VehicleRecord]] = []
    for type_code in CANDIDATE_VEHICLE_TYPE_CODES:
        try:
            record = repo.find_exact(
                tax_year=TAX_YEAR,
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


def anomalies(result: CheckResult) -> list[str]:
    if result.record is None:
        return [result.error or "not found"]
    c, r = result.catalog, result.record
    issues = []
    if c.use_value != r.use_value:
        issues.append(f"use value: catalog {c.use_value:,} vs CSV {r.use_value:,} (diff {c.use_value - r.use_value:+,})")
    if c.list_price != r.list_price:
        issues.append(f"list price: catalog {c.list_price:,} vs CSV {r.list_price:,} (diff {c.list_price - r.list_price:+,})")
    if PROPULSION_CODES.get(c.propulsion) != r.propulsion_code:
        issues.append(f"propulsion: catalog '{c.propulsion}' vs CSV code {r.propulsion_code}")
    if c.manufacturer != r.manufacturer:
        issues.append(f"manufacturer name: catalog '{c.manufacturer}' vs CSV '{r.manufacturer}'")
    if c.commercial_name != r.commercial_name:
        issues.append(f"commercial name: catalog '{c.commercial_name}' vs CSV '{r.commercial_name}'")
    return issues


def main() -> None:
    catalog_path = find_catalog_file()
    repo = VehicleRepository()
    results = [check(repo, v) for v in read_catalog(catalog_path)]

    print(f"Catalog: {catalog_path.name} / {CATALOG_SHEET}")
    print(f"Lookup key: tax_year={TAX_YEAR}, vehicle_type in {CANDIDATE_VEHICLE_TYPE_CODES}\n")

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
    any_issue = False
    for res in results:
        for issue in anomalies(res):
            any_issue = True
            print(f"  {res.catalog.vehicle_id} {res.catalog.commercial_name}: {issue}")
    if not any_issue:
        print("  none")


if __name__ == "__main__":
    main()
