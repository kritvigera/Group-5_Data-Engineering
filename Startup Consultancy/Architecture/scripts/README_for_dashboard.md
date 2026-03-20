# README for Dashboard

## Overview
This dashboard is designed to run as a Streamlit application for market overview, cluster comparison, and CO₂ emissions estimation.

## Required folder structure
Please keep the relative folder structure unchanged.

```text
architecture/
├─ data/
│  ├─ dashboard/
│  │  ├─ market_overview_data.xlsx
│  │  ├─ co2_target_values_by_model_year.xlsx
│  │  └─ vehicles_with_cluster_and_labels_v1 1.csv
│  └─ gold/
│     └─ vehicles_gold_ml.csv
└─ scripts/
   ├─ dashboard.py
   └─ best_co2_model.pkl
```

## Required files
Place the following files in the locations shown above:

- `architecture/scripts/dashboard.py`
- `architecture/scripts/best_co2_model.pkl`
- `architecture/data/dashboard/market_overview_data.xlsx`
- `architecture/data/dashboard/co2_target_values_by_model_year.xlsx`
- `architecture/data/dashboard/vehicles_with_cluster_and_labels_v1 1.csv`
- `architecture/data/gold/vehicles_gold_ml.csv`

## Python environment
It is recommended to use a virtual environment.

### Install dependencies
```bash
pip install -r requirements_for_dashboard.md
```

If you prefer installing manually:

```bash
pip install streamlit==1.39.0 pandas==2.2.2 numpy==1.26.4 plotly==5.24.1 openpyxl==3.1.5 scikit-learn==1.5.1
```

## How to run
Open a terminal and move to the `scripts` folder:

```bash
cd architecture/scripts
```

Then run:

```bash
python -m streamlit run dashboard.py
```

## Important notes
- Keep the folder structure unchanged, because the dashboard reads files using relative paths.
- Keep file names unchanged, especially:
  - `best_co2_model.pkl`
  - `market_overview_data.xlsx`
  - `co2_target_values_by_model_year.xlsx`
  - `vehicles_gold_ml.csv`
  - `vehicles_with_cluster_and_labels_v1 1.csv`
- If you use a virtual environment, make sure Streamlit is launched from that environment.

## Troubleshooting
### 1. Excel file cannot be read
Install:

```bash
pip install openpyxl==3.1.5
```

### 2. NumPy / pandas binary incompatibility
Reinstall both packages in the same environment:

```bash
pip uninstall -y numpy pandas
pip install numpy==1.26.4 pandas==2.2.2
```

### 3. Dashboard cannot find files
Check that:
- you are running the script from the `architecture/scripts` folder, or
- the relative folder structure is preserved exactly.

### 4. Model cannot be loaded
Check that:
- `best_co2_model.pkl` is in the `scripts` folder
- `scikit-learn==1.5.1` is installed
