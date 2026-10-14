import pandas as pd
from pathlib import Path

CSV_PATH = Path(__file__).parent.parent / "data" / "use_value_2026.csv"

df = pd.read_csv(
    CSV_PATH,
    encoding="windows-1255"
)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumn names:")
for column in df.columns:
    print(repr(column))

print("\nFirst rows:")
print(df.head())