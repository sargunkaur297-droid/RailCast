# RailCast — Railway Delay Prediction & Historical Journey Analytics

RailCast is a research and portfolio project for analysing historical Indian railway journey records, forecasting train-delay risk with machine learning, and exploring station-level delay patterns and railway corridor structure.

> **Important:** RailCast is not a live train-tracking service. Its dashboard uses locally prepared historical data and trained model artifacts. Forecasts and analytics should not be interpreted as official real-time railway information.

## Project at a glance

- **Data engineering:** Polars for data preparation and DuckDB for local analytical storage.
- **Machine learning:** LightGBM prediction models, with Ridge, Random Forest and XGBoost benchmark comparisons.
- **Network analysis:** NetworkX-based railway-zone and corridor metrics.
- **Station-level analytics:** Running-delay and dwell-delay decomposition from historical station records.
- **Dashboard:** Streamlit and Plotly.

## Dashboard

The current Streamlit application has four tabs:

1. **Network Overview** — zone-level journey volume, average delay and delay-rate summaries.
2. **Train Forecast** — train selection and model-driven delay estimates using the available historical dataset and trained model.
3. **Corridor Intelligence** — station-to-station corridor and delay-pattern analysis.
4. **Model & Architecture** — model performance and implementation information.

The dashboard expects the local DuckDB database at `db/railway.duckdb` and the trained model bundle at `models/champion_models.pkl`. The benchmark view also uses `models/benchmark_report.json`.

## Repository structure

```text
RailCast/
├── app/
│   └── dashboard.py
├── data/
│   ├── ir_data_dictionary.csv
│   └── station_data/
│       ├── stations_zones_mapping.json
│       └── IRN_edges.csv
├── docs/
├── models/
│   ├── champion_models.pkl
│   ├── benchmark_report.json
│   ├── correlation_summary.json
│   └── eda_results.json
├── scripts/
├── src/
│   ├── station_etl.py
│   ├── trajectory_profiler.py
│   ├── graph_builder.py
│   ├── cascade_features.py
│   ├── predictor.py
│   ├── benchmark.py
│   └── train.py
├── tests/
├── requirements.txt
├── run_app.bat
└── README.md
```

## Requirements

- Python 3.10 or newer (use a Python version supported by the installed packages).
- Windows, macOS or Linux.
- 8 GB RAM minimum; 16 GB recommended for larger data-processing runs.
- Additional disk space for the source datasets and generated DuckDB database.

## Setup

### 1. Clone the repository

```powershell
git clone https://github.com/sargunkaur297-droid/RailCast.git
cd RailCast
```

### 2. Create a virtual environment and install dependencies

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data setup

Large/raw source datasets and the generated DuckDB database are intentionally not committed to this repository. This keeps the Git repository smaller and avoids distributing data files unnecessarily.

To rebuild the station-level database, obtain the source data used by the project and place the following files in `data/station_data/`:

- `train_routes_delays_Sep2024.csv`
- `train_routes_Sep2024.csv`

The repository already includes `stations_zones_mapping.json`. The station ETL script requires both CSV files; it will stop with a clear missing-file error if either is absent.

For the journey-level benchmark/training pipeline, place the required journey dataset at `data/ir_train.csv`. Do not assume this file is included in a fresh clone.

## Build the local DuckDB database

After placing the two station-level CSV files in `data/station_data/`, run:

```powershell
$env:PYTHONPATH = "."
python src/station_etl.py
```

This creates `db/railway.duckdb` locally and builds the station, section-analytics and journey tables used by the dashboard.

## Run the dashboard

Once the local database and required model artifacts are available:

```powershell
streamlit run app/dashboard.py
```

Then open http://localhost:8501.

On Windows, you can alternatively double-click `run_app.bat` after installing dependencies and preparing the local database.

## Run tests

```powershell
$env:PYTHONPATH = "."
python -m pytest -q
```

Some integration tests depend on the locally generated DuckDB database and source data. Prepare the data/database first when running the complete suite. A fresh clone without those local assets may not be able to run every data-dependent test.

## Rebuild models (optional)

The benchmark and training scripts require `data/ir_train.csv` and the database/data inputs expected by the feature-engineering pipeline:

```powershell
$env:PYTHONPATH = "."
python src/benchmark.py
python src/train.py
```

These commands regenerate model artifacts and can take time depending on the dataset and hardware.

## Documentation

- [Architecture and workflow](docs/ARCHITECTURE_AND_WORKFLOW.md)
- [Usage and requirements](docs/USAGE_AND_REQUIREMENTS.md)
- [Scientific benchmarking and findings](docs/SCIENTIFIC_BENCHMARKING_AND_FINDINGS.md)
- [Model benchmark and comparison](docs/MODEL_BENCHMARK_AND_COMPARISON.md)
- [Viva evaluator guide](docs/VIVA_EVALUATOR_DEFENSE_GUIDE.md)
- [Prediction reliability and ground-truth audit](docs/PREDICTION_RELIABILITY_AND_GROUND_TRUTH.md)
- [NetworkX and feature-engineering report](README_NETWORKX_AND_FEATURE_ENGINEERING.md)

## Data, interpretation and limitations

- Results depend on the source dataset, preprocessing choices and the locally generated database.
- The dashboard is a historical-data research prototype, not an official railway information service.
- Model metrics in the repository describe the reported evaluation setup; they are not a guarantee of future performance.
- Do not commit private credentials, API keys, personal datasets or local database files.

## License

No license file is currently included. Unless a license is added, the repository's contents are not automatically granted an open-source reuse license.