from pathlib import Path
import pandas as pd
import numpy as np
import json

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
BRONZE_FILE = BASE_DIR / "data" / "bronze" / "vehicles_bronze.csv"
SILVER_DIR = BASE_DIR / "data" / "silver"
QUALITY_DIR = BASE_DIR / "reports" / "quality"

SILVER_FILE = SILVER_DIR / "vehicles_silver.csv"
QUALITY_REPORT_JSON = QUALITY_DIR / "quality_report.json"
MISSING_REPORT_CSV = QUALITY_DIR / "missing_values_report.csv"
NUMERIC_SUMMARY_CSV = QUALITY_DIR / "numeric_summary.csv"
DUPLICATES_CSV = QUALITY_DIR / "duplicate_rows.csv"
OUTLIER_REPORT_CSV = QUALITY_DIR / "outlier_report.csv"

SILVER_DIR.mkdir(parents=True, exist_ok=True)
QUALITY_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Schema definitions
# =========================
INTEGER_COLUMNS = [
    "Model_Year",
    "Cylinders",
    "Motor_kW",
    "Fuel_Cons_Comb_mpg",
    "Elec_Range_km",
    "CO2_Emissions_g_km",
    "CO2_Rating",
    "Smog_Rating"
]

FLOAT_COLUMNS = [
    "Engine_Size_L",
    "Fuel_Cons_City_L100km",
    "Fuel_Cons_Hwy_L100km",
    "Fuel_Cons_Comb_L100km",
    "Elec_Cons_City_kWh100km",
    "Elec_Cons_Hwy_kWh100km",
    "Elec_Cons_Comb_kWh100km",
    "Total_Range_km",
    "Recharge_Time_h"
]

STRING_COLUMNS = [
    "Make",
    "Model",
    "Vehicle_Class",
    "Vehicle_Type",
    "Transmission",
    "Fuel_Type_Primary",
    "Fuel_Type_Secondary"
]

# Keep all columns needed for downstream gold + clustering
KEEP_COLUMNS = [
    "Model_Year",
    "Make",
    "Model",
    "Vehicle_Class",
    "Vehicle_Type",
    "Engine_Size_L",
    "Cylinders",
    "Motor_kW",
    "Transmission",
    "Fuel_Type_Primary",
    "Fuel_Type_Secondary",
    "Fuel_Cons_City_L100km",
    "Fuel_Cons_Hwy_L100km",
    "Fuel_Cons_Comb_L100km",
    "Fuel_Cons_Comb_mpg",
    "Elec_Cons_City_kWh100km",
    "Elec_Cons_Hwy_kWh100km",
    "Elec_Cons_Comb_kWh100km",
    "Elec_Range_km",
    "Total_Range_km",
    "Recharge_Time_h",
    "CO2_Emissions_g_km",
    "CO2_Rating",
    "Smog_Rating"
]

# Structural missingness by design
STRUCTURAL_MISSING = {
    "Model_Year": ["Conventional/Hybrid"],
    "Engine_Size_L": ["Battery Electric (BEV)"],
    "Cylinders": ["Battery Electric (BEV)"],
    "Motor_kW": ["Conventional/Hybrid"],
    "Fuel_Type_Secondary": ["Conventional/Hybrid", "Battery Electric (BEV)"],
    "Fuel_Cons_City_L100km": ["Battery Electric (BEV)"],
    "Fuel_Cons_Hwy_L100km": ["Battery Electric (BEV)"],
    "Fuel_Cons_Comb_L100km": ["Battery Electric (BEV)"],
    "Fuel_Cons_Comb_mpg": ["Battery Electric (BEV)", "Plug-in Hybrid (PHEV)"],
    "Elec_Cons_City_kWh100km": ["Conventional/Hybrid", "Plug-in Hybrid (PHEV)"],
    "Elec_Cons_Hwy_kWh100km": ["Conventional/Hybrid", "Plug-in Hybrid (PHEV)"],
    "Elec_Cons_Comb_kWh100km": ["Conventional/Hybrid"],
    "Elec_Range_km": ["Conventional/Hybrid"],
    "Total_Range_km": ["Conventional/Hybrid"],
    "Recharge_Time_h": ["Conventional/Hybrid"],
    "CO2_Rating": ["Conventional/Hybrid"],
    "Smog_Rating": ["Conventional/Hybrid"]
}

VALID_VEHICLE_TYPES = {
    "Conventional/Hybrid",
    "Battery Electric (BEV)",
    "Plug-in Hybrid (PHEV)"
}

VALID_FUEL_TYPES = {"X", "Z", "D", "E", "N", "B"}

VALID_TRANSMISSION_PREFIXES = {"A", "AM", "AS", "AV", "M"}


def safe_to_numeric(df, columns, kind="float"):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if kind == "int":
                df[col] = df[col].astype("Int64")
    return df


def safe_to_string(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    return df


def standardize_column_names(df):
    df.columns = [c.strip() for c in df.columns]
    return df


def create_vehicle_id(df):
    key_cols = ["Make", "Model", "Vehicle_Class", "Vehicle_Type", "Transmission"]
    for col in key_cols:
        if col not in df.columns:
            df[col] = "UNKNOWN"

    df["vehicle_id"] = (
        df["Make"].fillna("NA").astype(str) + "_" +
        df["Model"].fillna("NA").astype(str) + "_" +
        df["Vehicle_Class"].fillna("NA").astype(str) + "_" +
        df["Vehicle_Type"].fillna("NA").astype(str) + "_" +
        df["Transmission"].fillna("NA").astype(str)
    )
    return df


def classify_missingness(df):
    records = []
    total_rows = len(df)

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_pct = round((missing_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

        if missing_count == 0:
            classification = "No missing"
            notes = ""
        elif col in STRUCTURAL_MISSING:
            classification = "Mostly structural missingness"
            notes = f"Expected missing for vehicle types: {STRUCTURAL_MISSING[col]}"
        else:
            classification = "Potential data quality issue"
            notes = "Review needed"

        records.append({
            "column": col,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "classification": classification,
            "notes": notes
        })

    return pd.DataFrame(records)


def check_duplicates(df):
    dup_mask = df.duplicated()
    return df[dup_mask].copy()


def validate_categorical_values(df):
    issues = {}

    if "Vehicle_Type" in df.columns:
        invalid_vehicle_type = ~df["Vehicle_Type"].isin(VALID_VEHICLE_TYPES) & df["Vehicle_Type"].notna()
        issues["invalid_vehicle_type_count"] = int(invalid_vehicle_type.sum())

    if "Fuel_Type_Primary" in df.columns:
        invalid_primary = ~df["Fuel_Type_Primary"].isin(VALID_FUEL_TYPES) & df["Fuel_Type_Primary"].notna()
        issues["invalid_fuel_type_primary_count"] = int(invalid_primary.sum())

    if "Fuel_Type_Secondary" in df.columns:
        invalid_secondary = ~df["Fuel_Type_Secondary"].isin(VALID_FUEL_TYPES) & df["Fuel_Type_Secondary"].notna()
        issues["invalid_fuel_type_secondary_count"] = int(invalid_secondary.sum())

    return issues


def extract_transmission_prefix(series):
    return series.astype("string").str.extract(r"^([A-Z]+)", expand=False)


def validate_transmission_codes(df):
    issues = {}
    if "Transmission" in df.columns:
        prefix = extract_transmission_prefix(df["Transmission"])
        invalid_prefix = ~prefix.isin(VALID_TRANSMISSION_PREFIXES) & prefix.notna()
        issues["invalid_transmission_prefix_count"] = int(invalid_prefix.sum())
    return issues


def range_checks(df):
    issues = {}

    checks = {
        "Model_Year_out_of_expected_range": ((df["Model_Year"] < 2012) | (df["Model_Year"] > 2026)) if "Model_Year" in df.columns else pd.Series([], dtype=bool),
        "Engine_Size_L_negative": (df["Engine_Size_L"] < 0) if "Engine_Size_L" in df.columns else pd.Series([], dtype=bool),
        "Cylinders_nonpositive": (df["Cylinders"] <= 0) if "Cylinders" in df.columns else pd.Series([], dtype=bool),
        "Motor_kW_negative": (df["Motor_kW"] < 0) if "Motor_kW" in df.columns else pd.Series([], dtype=bool),
        "Fuel_Cons_City_L100km_negative": (df["Fuel_Cons_City_L100km"] < 0) if "Fuel_Cons_City_L100km" in df.columns else pd.Series([], dtype=bool),
        "Fuel_Cons_Hwy_L100km_negative": (df["Fuel_Cons_Hwy_L100km"] < 0) if "Fuel_Cons_Hwy_L100km" in df.columns else pd.Series([], dtype=bool),
        "Fuel_Cons_Comb_L100km_negative": (df["Fuel_Cons_Comb_L100km"] < 0) if "Fuel_Cons_Comb_L100km" in df.columns else pd.Series([], dtype=bool),
        "Fuel_Cons_Comb_mpg_nonpositive": (df["Fuel_Cons_Comb_mpg"] <= 0) if "Fuel_Cons_Comb_mpg" in df.columns else pd.Series([], dtype=bool),
        "Elec_Cons_City_kWh100km_negative": (df["Elec_Cons_City_kWh100km"] < 0) if "Elec_Cons_City_kWh100km" in df.columns else pd.Series([], dtype=bool),
        "Elec_Cons_Hwy_kWh100km_negative": (df["Elec_Cons_Hwy_kWh100km"] < 0) if "Elec_Cons_Hwy_kWh100km" in df.columns else pd.Series([], dtype=bool),
        "Elec_Cons_Comb_kWh100km_negative": (df["Elec_Cons_Comb_kWh100km"] < 0) if "Elec_Cons_Comb_kWh100km" in df.columns else pd.Series([], dtype=bool),
        "Elec_Range_km_negative": (df["Elec_Range_km"] < 0) if "Elec_Range_km" in df.columns else pd.Series([], dtype=bool),
        "Total_Range_km_negative": (df["Total_Range_km"] < 0) if "Total_Range_km" in df.columns else pd.Series([], dtype=bool),
        "Recharge_Time_h_negative": (df["Recharge_Time_h"] < 0) if "Recharge_Time_h" in df.columns else pd.Series([], dtype=bool),
        "CO2_Emissions_g_km_negative": (df["CO2_Emissions_g_km"] < 0) if "CO2_Emissions_g_km" in df.columns else pd.Series([], dtype=bool),
        "CO2_Rating_out_of_range": ((df["CO2_Rating"] < 1) | (df["CO2_Rating"] > 10)) if "CO2_Rating" in df.columns else pd.Series([], dtype=bool),
        "Smog_Rating_out_of_range": ((df["Smog_Rating"] < 1) | (df["Smog_Rating"] > 10)) if "Smog_Rating" in df.columns else pd.Series([], dtype=bool),
    }

    for name, mask in checks.items():
        if len(mask) > 0:
            issues[name] = int(mask.fillna(False).sum())

    return issues


def build_outlier_report(df):
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    records = []

    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 5:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_count = int(((df[col] < lower) | (df[col] > upper)).fillna(False).sum())

        records.append({
            "column": col,
            "q1": round(float(q1), 4),
            "q3": round(float(q3), 4),
            "iqr": round(float(iqr), 4),
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
            "outlier_count_iqr": outlier_count
        })

    return pd.DataFrame(records)


def main():
    if not BRONZE_FILE.exists():
        raise FileNotFoundError(f"Bronze file not found: {BRONZE_FILE}")

    # 1. Read bronze
    df = pd.read_csv(BRONZE_FILE)
    original_rows = len(df)

    # 2. Standardize column names
    df = standardize_column_names(df)

    print("Bronze columns:")
    print(df.columns.tolist())

    # 3. Type conversion
    df = safe_to_numeric(df, FLOAT_COLUMNS, kind="float")
    df = safe_to_numeric(df, INTEGER_COLUMNS, kind="int")
    df = safe_to_string(df, STRING_COLUMNS)

    # 4. Create surrogate key
    df = create_vehicle_id(df)

    # 5. Feature selection for silver
    final_keep_cols = [c for c in KEEP_COLUMNS if c in df.columns]
    final_keep_cols.append("vehicle_id")
    df = df[final_keep_cols].copy()

    # 6. Exact duplicate check
    duplicates_df = check_duplicates(df)
    duplicates_df.to_csv(DUPLICATES_CSV, index=False)

    # 7. Drop exact duplicates
    df = df.drop_duplicates().copy()

    # 8. Missing value report
    missing_df = classify_missingness(df)
    missing_df.to_csv(MISSING_REPORT_CSV, index=False)

    # 9. Numeric summary
    numeric_summary = df.describe(include=[np.number]).T
    numeric_summary.to_csv(NUMERIC_SUMMARY_CSV)

    # 10. Rule-based anomaly checks
    range_issue_counts = range_checks(df)
    categorical_issue_counts = validate_categorical_values(df)
    transmission_issue_counts = validate_transmission_codes(df)

    # 11. Statistical outlier report (flag only, do not remove automatically)
    outlier_df = build_outlier_report(df)
    outlier_df.to_csv(OUTLIER_REPORT_CSV, index=False)

    # 12. Save silver
    df.to_csv(SILVER_FILE, index=False)

    # 13. Save quality report
    quality_report = {
        "source_file": str(BRONZE_FILE),
        "silver_file": str(SILVER_FILE),
        "original_row_count": int(original_rows),
        "final_row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "exact_duplicate_rows_found": int(len(duplicates_df)),
        "missing_values_summary_file": str(MISSING_REPORT_CSV),
        "numeric_summary_file": str(NUMERIC_SUMMARY_CSV),
        "duplicate_rows_file": str(DUPLICATES_CSV),
        "outlier_report_file": str(OUTLIER_REPORT_CSV),
        "range_issue_counts": range_issue_counts,
        "categorical_issue_counts": categorical_issue_counts,
        "transmission_issue_counts": transmission_issue_counts,
        "retained_columns": df.columns.tolist(),
        "notes": [
            "Structural missing values are retained because they reflect vehicle-type-specific non-applicability.",
            "Outliers are reported, not automatically removed, because extreme values may represent valid vehicle designs.",
            "Silver retains a wide schema to support downstream gold preparation and clustering feature engineering."
        ]
    }

    with open(QUALITY_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=4)

    print("Silver transformation completed.")
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print(f"Silver file: {SILVER_FILE}")
    print("Saved columns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()
