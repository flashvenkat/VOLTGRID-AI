# VOLTGRID AI — Smart Microgrid Load Forecasting & Power Dispatch Engine

## Overview

VOLTGRID AI is an intelligent microgrid energy demand forecasting and operational
dispatch platform. Built with FastAPI and a modern frontend dashboard, it ingests
hourly weather telemetry (ambient temperature, humidity, solar irradiance) to
forecast energy demand and optimize power allocation between renewable battery
storage and the main grid.

## Key Features

- **AI-Powered Load Forecasting:** Integrates with Zoho Catalyst QuickML REST
  endpoints with an automated local regression fallback engine for resilient
  offline capabilities.
- **Batch Telemetry Ingestion:** Supports CSV uploads and JSON batches of up to
  1,000 weather telemetry records.
- **Microgrid Power Dispatch Engine:** Applies threshold-based dispatch rules,
  calculates a battery state-of-charge estimate, and monitors critical overloads.
- **Real-Time Operational Dashboard:** Includes an interactive 24-point load
  curve, peak demand monitoring, status cards, and a complete hourly dispatch
  schedule.
- **Robust API Pipeline:** Built on FastAPI with Pydantic validation and a local
  predictor contract test.

## Tech Stack

- **Backend:** Python 3.10+, FastAPI, Uvicorn, Pydantic
- **Data and forecasting:** Pandas, NumPy, Zoho Catalyst QuickML (optional)
- **Frontend:** HTML5, CSS3, JavaScript Fetch API, Tailwind CSS CDN, Chart.js CDN,
  Lucide Icons CDN
- **Deployment and tooling:** Git, VS Code

## Getting Started

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/flashvenkat/VOLTGRID-AI.git
cd VOLTGRID-AI
python -m pip install -r requirements.txt
```

Python 3.10 or later is recommended. The application expects
`sample_test_input.csv` in the project root; `train_telemetry.csv` is optional.

### 2. Configure QuickML (optional)

When `QUICKML_ENDPOINT_URL` is set, the predictor sends each telemetry record to
that HTTP(S) endpoint. If it is unset, the configured default Zoho Catalyst
endpoint is used. For authentication, set the endpoint key and, if required, a
Zoho OAuth token in your environment:

```powershell
# PowerShell
$env:QUICKML_ENDPOINT_KEY = "your-endpoint-key"
$env:ZOHO_OAUTH_TOKEN = "your-zoho-access-token"
$env:QUICKML_ENDPOINT_URL = "https://your-quickml-endpoint"
```

```bash
# macOS / Linux
export QUICKML_ENDPOINT_KEY="your-endpoint-key"
export ZOHO_OAUTH_TOKEN="your-zoho-access-token"
export QUICKML_ENDPOINT_URL="https://your-quickml-endpoint"
```

`QUICKML_ENDPOINT_URL` can be omitted to use the default endpoint configured in
`predict.py`. Do not commit endpoint keys or OAuth tokens to source control. If
QuickML is unavailable, returns an error, or cannot be parsed, the predictor logs
the failure and uses the local regression fallback.

### 3. Run the web application

From the project root:

```bash
python run.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in a browser. The dashboard
loads the bundled 24-hour sample on startup. Tailwind CSS, Chart.js, and Lucide
Icons are loaded from CDNs, so those dashboard libraries require an internet
connection.

### 4. Run the predictor and tests

Predict from a CSV file using the command-line interface:

```bash
python predict.py --input sample_test_input.csv --output predictions.csv
```

The output CSV contains `timestamp,target_demand_kw`, with one prediction for
each input row. Run the local contract validator with:

```bash
python test_eval.py
```

On success, the validator prints `SUCCESS`.

## Forecasting and Dispatch Rules

Telemetry rows require these fields:

| Field | Description | Validation |
| --- | --- | --- |
| `timestamp` | ISO-formatted date and time | Must parse as a date/time |
| `ambient_temp_c` | Ambient temperature in Celsius | -80 to 80 |
| `humidity_pct` | Relative humidity percentage | 0 to 100 |
| `solar_irradiance_wm2` | Solar irradiance in W/m² | 0 to 2000 |

The predictor fits a robust regression to `train_telemetry.csv` when that file
has sufficient valid rows and the required target column. Otherwise it uses a
deterministic regression prior based on time of day and weather features.
Predictions are finite numeric demand estimates in kW.

Dispatch decisions are calculated independently for each timestamp:

- **Renewable Storage Mode:** Demand is greater than 160 kW, or the local
  timestamp hour is between 18:00 and 22:00 inclusive.
- **Main Grid Mode:** Demand is at most 160 kW and the hour is outside the
  18:00–22:00 window.
- **Critical overload:** Set when demand is at least 200 kW, regardless of the
  selected power mode.

The API returns results for all submitted telemetry rows. The dashboard chart
plots the first 24 points for readability; the dispatch table displays the
complete result batch.

## API Reference

### `GET /api/v1/samples`

Returns the bundled sample telemetry:

```json
{
  "telemetry": [
    {
      "timestamp": "2026-10-24T00:00:00",
      "ambient_temp_c": "24",
      "humidity_pct": "78",
      "solar_irradiance_wm2": "0"
    }
  ]
}
```

The sample endpoint returns CSV-derived values as strings; the forecast endpoint
validates and converts them to numeric fields.

### `POST /api/v1/forecast`

Accepts a JSON body with a `telemetry` array (or a raw JSON array), a multipart
CSV upload under the `file` or `csv` field, or no body to forecast the bundled
sample. The request must contain 1–1,000 valid telemetry records.

Example JSON request:

```json
{
  "telemetry": [
    {
      "timestamp": "2026-10-24T09:00:00",
      "ambient_temp_c": 27.5,
      "humidity_pct": 62,
      "solar_irradiance_wm2": 425
    }
  ]
}
```

The response has a `dispatch` array. Each output record includes the original
weather values and timestamp, plus:

- `target_demand_kw`: predicted demand
- `grid_mode`: `Renewable Storage Mode` or `Main Grid Mode`
- `critical_overload`: Boolean alert flag
- `battery_soc_pct`: estimated state of charge percentage

Invalid fields or batches return HTTP 422 with validation details.

## Project Structure

```text
VOLTGRID-AI/
├── main.py                  # FastAPI routes, validation, and dispatch
├── predict.py               # QuickML integration, regression fallback, CLI
├── run.py                   # Local Uvicorn entry point
├── requirements.txt         # Python dependencies
├── sample_test_input.csv    # Bundled 24-hour sample telemetry
├── test_eval.py             # Local predictor CSV contract validator
└── static/
    └── index.html           # Dashboard interface
```

## Notes

- QuickML requests are made once per telemetry record. The local regression
  fallback keeps forecasts available if the remote service is unavailable.
- `battery_soc_pct` is a dispatch-dashboard estimate, not a battery-management
  system measurement.
- This project provides forecast and decision-support outputs. Validate model
  quality against representative labeled data and operational requirements
  before relying on it for real-world grid control.
