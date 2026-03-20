from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
SILVER_FILE = BASE_DIR / "data" / "silver" / "vehicles_silver.csv"
GOLD_DIR = BASE_DIR / "data" / "gold"
FIG_DIR = BASE_DIR / "reports" / "figures"

GOLD_ML_FILE = GOLD_DIR / "vehicles_gold_supervised learning.csv"
GOLD_SUMMARY_FILE = GOLD_DIR / "vehicle_type_summary.csv"
FIG1 = FIG_DIR / "avg_co2_by_vehicle_type.png"
FIG2 = FIG_DIR / "vehicle_count_by_vehicle_type.png"

GOLD_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not SILVER_FILE.exists():
        raise FileNotFoundError(f"Silver file not found: {SILVER_FILE}")

    df = pd.read_csv(SILVER_FILE)
    df.columns = [c.strip() for c in df.columns]

    print("Silver columns:")
    print(df.columns.tolist())

    # -------------------------
    # 1. Gold ML dataset
    # -------------------------
    ml_cols = [
        "vehicle_id",
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

    existing_ml_cols = [c for c in ml_cols if c in df.columns]
    ml_df = df[existing_ml_cols].copy()

    if "Vehicle_Type" in ml_df.columns:
        ml_df["is_bev"] = (ml_df["Vehicle_Type"] == "Battery Electric (BEV)").astype(int)
        ml_df["is_phev"] = (ml_df["Vehicle_Type"] == "Plug-in Hybrid (PHEV)").astype(int)
        ml_df["is_conv_hybrid"] = (ml_df["Vehicle_Type"] == "Conventional/Hybrid").astype(int)

    ml_df.to_csv(GOLD_ML_FILE, index=False)

    # -------------------------
    # 2. Gold summary
    # -------------------------
    agg_dict = {
        "vehicle_count": ("Vehicle_Type", "size")
    }

    if "CO2_Emissions_g_km" in df.columns:
        agg_dict["avg_co2_g_km"] = ("CO2_Emissions_g_km", "mean")
        agg_dict["median_co2_g_km"] = ("CO2_Emissions_g_km", "median")

    if "Engine_Size_L" in df.columns:
        agg_dict["avg_engine_size_l"] = ("Engine_Size_L", "mean")

    if "Motor_kW" in df.columns:
        agg_dict["avg_motor_kw"] = ("Motor_kW", "mean")

    if "Fuel_Cons_Comb_L100km" in df.columns:
        agg_dict["avg_fuel_cons_l100km"] = ("Fuel_Cons_Comb_L100km", "mean")

    if "Elec_Cons_Comb_kWh100km" in df.columns:
        agg_dict["avg_elec_cons_kwh100km"] = ("Elec_Cons_Comb_kWh100km", "mean")

    if "Elec_Range_km" in df.columns:
        agg_dict["avg_elec_range_km"] = ("Elec_Range_km", "mean")

    if "Recharge_Time_h" in df.columns:
        agg_dict["avg_recharge_time_h"] = ("Recharge_Time_h", "mean")

    summary = (
        df.groupby("Vehicle_Type", dropna=False)
        .agg(**agg_dict)
        .reset_index()
    )

    summary.to_csv(GOLD_SUMMARY_FILE, index=False)

    # -------------------------
    # 3. Charts
    # -------------------------
    if "avg_co2_g_km" in summary.columns:
        plt.figure(figsize=(8, 5))
        plt.bar(summary["Vehicle_Type"], summary["avg_co2_g_km"])
        plt.title("Average CO2 Emissions by Vehicle Type")
        plt.ylabel("CO2 Emissions (g/km)")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(FIG1, dpi=300)
        plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(summary["Vehicle_Type"], summary["vehicle_count"])
    plt.title("Vehicle Count by Vehicle Type")
    plt.ylabel("Count")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIG2, dpi=300)
    plt.close()

    print("Gold layer files created successfully.")
    print(f"ML dataset: {GOLD_ML_FILE}")
    print(f"Summary dataset: {GOLD_SUMMARY_FILE}")
    print(f"Figure 1: {FIG1}")
    print(f"Figure 2: {FIG2}")

if __name__ == "__main__":
    main()
