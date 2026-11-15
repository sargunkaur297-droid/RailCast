# RailCast — Installation and Usage

## 1. Requirements

- Python 3.10 or newer.
- Windows, macOS or Linux.
- 8 GB RAM minimum; 16 GB recommended for large ETL and model-training workloads.
- Disk space for the source CSV files, virtual environment and generated DuckDB database.

Install all Python dependencies from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The requirements file includes XGBoost because the benchmark module imports it.

## 2. Source data required

The raw source datasets and generated DuckDB database are not included in a fresh clone. Obtain the source data used for this project and place the files at these exact paths:

| File | Used by |
|---|---|
| `data/station_data/train_routes_delays_Sep2024.csv` | Station-level ETL |
| `data/station_data/train_routes_Sep2024.csv` | Station-level ETL |
| `data/ir_train.csv` | Journey-level ETL, feature engineering, benchmark and training |
| `data/ir_test.csv` | Optional dashboard fallback for train selection |

The repository includes the station-to-zone mapping file at `data/station_data/stations_zones_mapping.json`. Large raw CSVs and local DuckDB files are excluded by `.gitignore`; they must be supplied locally. The ETL scripts cannot recreate missing source datasets.

## 3. Build the station-level DuckDB database

From the repository root, after placing both required station CSV files:

```powershell
$env:PYTHONPATH = "."
python src/station_etl.py
```

The script creates `db/railway.duckdb` and writes the `station_stops`, `section_analytics` and `journeys` tables.

If you also need to rebuild the journey-level dataset table from `data/ir_train.csv`, run:

```powershell
$env:PYTHONPATH = "."
python src/etl.py
```

Note: `src/etl.py` rebuilds the DuckDB file. Run the station ETL afterwards if the dashboard needs the station-level tables as well.

## 4. Launch the dashboard

The current dashboard has four tabs: Network Overview, Train Forecast, Corridor Intelligence, and Model & Architecture.

The dashboard expects:
- `db/railway.duckdb` with the tables created by the station ETL.
- `models/champion_models.pkl` for prediction.
- `models/benchmark_report.json` for benchmark display.

Run:

```powershell
streamlit run app/dashboard.py
```

Open http://localhost:8501. On Windows, `run_app.bat` is an alternative launcher after the environment and local data have been prepared.

**This is a historical-data research prototype, not a live train-tracking service.** It does not provide official real-time running status.

## 5. Rebuild benchmark and models (optional)

The feature-engineering, benchmark and training scripts require `data/ir_train.csv` and a compatible local DuckDB database:

```powershell
$env:PYTHONPATH = "."
python src/benchmark.py
python src/train.py
```

These scripts regenerate benchmark/model artifacts and may take several minutes or longer depending on the data and hardware.

## 6. Run tests

```powershell
$env:PYTHONPATH = "."
python -m pytest -q
```

Some tests are integration tests and require the locally generated DuckDB database and source data. A fresh clone without those files may not pass every data-dependent test.

## 7. Troubleshooting

- **Missing CSV:** Verify the exact filenames and paths in the source-data table above.
- **Missing database:** Run `python src/station_etl.py` after adding the two station CSV files.
- **Missing model bundle:** Restore the tracked `models/champion_models.pkl` artifact or regenerate it using `python src/train.py` after preparing its inputs.
- **Import errors:** Activate the virtual environment, install `requirements.txt`, and run commands from the repository root.
- **Port already in use:** Start Streamlit on another port, for example `streamlit run app/dashboard.py --server.port 8502`.
