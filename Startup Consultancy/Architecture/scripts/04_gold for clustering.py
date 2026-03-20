from pathlib import Path
import pandas as pd
import numpy as np
import re

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
GOLD_FILE = BASE_DIR / "data" / "gold" / "vehicles_gold_supervised learning.csv"
OUT_FILE = BASE_DIR / "data" / "gold" / "vehicles_gold_clustering.csv"

def extract_transmission_type(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().upper()
    letters = re.findall(r"[A-Z]+", x)
    return "".join(letters) if letters else np.nan

def extract_num_gears(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().upper()
    numbers = re.findall(r"\d+", x)
    return float(numbers[0]) if numbers else np.nan

def safe_ratio(num, den):
    return np.where(
        pd.notna(num) & pd.notna(den) & (den != 0),
        num / den,
        np.nan
    )

def main():
    if not GOLD_FILE.exists():
        raise FileNotFoundError(f"Gold file not found: {GOLD_FILE}")

    df = pd.read_csv(GOLD_FILE)
    df.columns = [c.strip() for c in df.columns]

    print("Gold columns:")
    print(df.columns.tolist())

    # -------------------------
    # A. Features for all vehicle types
    # -------------------------
    df["Transmission_Type"] = df["Transmission"].apply(extract_transmission_type)
    df["Num_Gears"] = df["Transmission"].apply(extract_num_gears)

    df["Is_SUV"] = df["Vehicle_Class"].astype(str).str.contains("SUV", case=False, na=False).astype(int)
    df["Is_Truck"] = df["Vehicle_Class"].astype(str).str.contains("Truck|Pickup", case=False, na=False).astype(int)

    # -------------------------
    # B. Conventional/Hybrid + PHEV related
    # -------------------------
    df["Fuel_Efficiency_Index"] = np.where(
        pd.notna(df["Fuel_Cons_Comb_L100km"]) & (df["Fuel_Cons_Comb_L100km"] != 0),
        100 / df["Fuel_Cons_Comb_L100km"],
        np.nan
    )

    df["CO2_per_L_engine"] = safe_ratio(df["CO2_Emissions_g_km"], df["Engine_Size_L"])

    df["City_Hwy_Fuel_Ratio"] = safe_ratio(
        df["Fuel_Cons_City_L100km"],
        df["Fuel_Cons_Hwy_L100km"]
    )

    # -------------------------
    # C. BEV + PHEV related
    # -------------------------
    df["Electric_Efficiency_Index"] = np.where(
        pd.notna(df["Elec_Cons_Comb_kWh100km"]) & (df["Elec_Cons_Comb_kWh100km"] != 0),
        100 / df["Elec_Cons_Comb_kWh100km"],
        np.nan
    )

    df["Range_per_ChargeHour"] = safe_ratio(
        df["Elec_Range_km"],
        df["Recharge_Time_h"]
    )

    # -------------------------
    # D. PHEV specific
    # -------------------------
    df["Elec_Range_Share"] = safe_ratio(
        df["Elec_Range_km"],
        df["Total_Range_km"]
    )

    # -------------------------
    # Basic feature selection for clustering preparation
    # -------------------------
    clustering_cols = [
        "vehicle_id",
        "Make",
        "Model",
        "Vehicle_Class",
        "Vehicle_Type",
        "Fuel_Type_Primary",
        "Fuel_Type_Secondary",
        "Transmission",
        "Transmission_Type",
        "Num_Gears",
        "Is_SUV",
        "Is_Truck",
        "Model_Year",
        "Engine_Size_L",
        "Cylinders",
        "Motor_kW",
        "Fuel_Cons_City_L100km",
        "Fuel_Cons_Hwy_L100km",
        "Fuel_Cons_Comb_L100km",
        "Elec_Cons_Comb_kWh100km",
        "Elec_Range_km",
        "Total_Range_km",
        "Recharge_Time_h",
        "CO2_Emissions_g_km",
        "Fuel_Efficiency_Index",
        "CO2_per_L_engine",
        "City_Hwy_Fuel_Ratio",
        "Electric_Efficiency_Index",
        "Range_per_ChargeHour",
        "Elec_Range_Share"
    ]

    existing_cols = [c for c in clustering_cols if c in df.columns]
    cluster_df = df[existing_cols].copy()

    numeric_cols = cluster_df.select_dtypes(include=[np.number]).columns

    cluster_df[numeric_cols] = cluster_df[numeric_cols].fillna(0)

    categorical_cols = cluster_df.select_dtypes(include=["object", "string"]).columns

    cluster_df[categorical_cols] = cluster_df[categorical_cols].fillna("Unknown")

    print("\nMissing values filled:")
    print("Numeric features → 0")
    print("Categorical features → 'Unknown'")


    cluster_df.to_csv(OUT_FILE, index=False)

    print("Clustering feature file created successfully.")
    print(f"Saved to: {OUT_FILE}")
    print("New columns added:")
    print([
        "Transmission_Type",
        "Num_Gears",
        "Is_SUV",
        "Is_Truck",
        "Fuel_Efficiency_Index",
        "CO2_per_L_engine",
        "City_Hwy_Fuel_Ratio",
        "Electric_Efficiency_Index",
        "Range_per_ChargeHour",
        "Elec_Range_Share"
    ])

if __name__ == "__main__":
    main()
