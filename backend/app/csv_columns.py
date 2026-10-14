"""The only place that knows the government CSV column names."""

CSV_ENCODING = "windows-1255"
CSV_DATE_FORMAT = "%d/%m/%Y"

# Government column name (after stripping whitespace) -> VehicleRecord field name.
COLUMN_TO_FIELD: dict[str, str] = {
    "שנת מס": "tax_year",
    "שנת רישום": "registration_year",
    "קוד סוג רכב": "vehicle_type_code",
    "קוד תוצר": "manufacturer_code",
    "תוצר": "manufacturer",
    "קוד דגם": "model_code",
    "תאור דגם": "model_description",
    "סוג דלק": "fuel_type_code",
    "כינוי מסחרי": "commercial_name",
    "לא-0.כן-1 אוטומט": "is_automatic",
    "1-חשמלי.2-היברידי.3-פלג-אין": "propulsion_code",
    "ציון ירוק": "green_score",
    "דרגת זיהום": "pollution_level",
    "נפח מנוע": "engine_volume_cc",
    "משקל": "weight_kg",
    "תאריך תחולה": "effective_date",
    "מחיר מחירון": "list_price",
    "מחיר מתואם": "adjusted_price",
    "שווי שימוש": "use_value",
}
