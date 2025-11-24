# GarminDataReader

GarminDataReader logs into Garmin, downloads your activity history, stores it in a local SQLite database, and exports summarized CSV data.

## Prerequisites

- Python 3.10+ and `pip`
- Garmin account credentials
- Optional: a virtual environment for isolation

## 1. Configure Environment Variables

Set your Garmin credentials as environment variables so the scripts can authenticate without hardcoding secrets. Easiest way to do so is to make a `.env` in the project directory.

```bash
GARMIN_EMAIL="your-email@example.com"
GARMIN_PASSWORD="your-password"
```


## 2. Choose a Start Date

Edit `db_filler.py` to define the earliest activity date you want to ingest. Update the `start_date` definition near the top of the `main` section:

```
db_filler.py lines 492-493
start_date = date(2025, 2, 20)
```

Replace the year, month, and day with your desired starting point for activity download.

## 3. Clear Placeholder Data

Delete the temporary files shipped with the repo so your run starts from a clean slate:

`cache.db` 
`runs_data.csv`

## 4. Run the Ingestion Pipeline

Use the orchestrator script to run the tools in the required order. It will call `get_tokens.py`, `db_filler.py`, and `db_to_csv.py` sequentially.

```bash
python update.py
```

This will:

1. Retrieve a fresh Garmin auth token (`get_tokens.py`)
2. Populate the SQLite database with your activities (`db_filler.py`)
3. Export the database contents to `runs_data.csv` (`db_to_csv.py`)

## 5. Output

- `cache.db`: SQLite database containing all fetched data
- `runs_data.csv`: CSV summary suitable for spreadsheets or BI tools

Re-run `python update.py` whenever you want to refresh the dataset; adjust `start_date` again if you want to backfill earlier runs.