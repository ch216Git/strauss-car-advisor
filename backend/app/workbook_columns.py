"""The only place that knows the challenge workbook's sheet and column names."""

CATALOG_SHEET = "קטלוג רכבים"
OCCUPATIONS_SHEET = "עיסוקים"

YES = "כן"
NO = "לא"

# Workbook column -> CatalogCar field name.
CATALOG_COLUMN_TO_FIELD: dict[str, str] = {
    "מזהה": "car_id",
    "קבוצה": "group",
    "קוד תוצר": "manufacturer_code",
    "דגם (כינוי מסחרי)": "commercial_name",
    "קוד דגם": "model_code",
    "שנת רישום": "registration_year",
    "עלות חודשית לחברה (דמה)": "monthly_cost",
    "השתתפות עובד ברירת מחדל (דמה)": "default_participation",
}

# Catalog columns whose official values come from the Tax Authority CSV (Step 1 lookup).
# The app must not read them; only scripts/verify_catalog.py does, to cross-check the workbook.
CATALOG_CROSS_CHECK_COLUMNS: dict[str, str] = {
    "תוצר": "manufacturer",
    "הנעה": "propulsion",
    "מחיר מחירון": "list_price",
    "שווי שימוש חודשי": "use_value",
}

# Workbook column -> Occupation field name.
OCCUPATION_COLUMN_TO_FIELD: dict[str, str] = {
    "עיסוק": "name",
    "קבוצת זכאות": "group",
    "רכב חובה": "car_required",
    "מותר לוותר": "waiver_allowed",
    "אחזקת רכב בוויתור (₪)": "waiver_maintenance",
}
