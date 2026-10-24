"""Forecast telemetry with optional QuickML inference and a local robust regressor."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

FEATURE_COLUMNS = ("ambient_temp_c", "humidity_pct", "solar_irradiance_wm2")
TARGET_COLUMN = "target_demand_kw"
TRAINING_FILE = Path(__file__).with_name("train_telemetry.csv")
QUICKML_ENDPOINT_URL = (
    "https://api.catalyst.zoho.in/quickml/v1/project/"
    "105679000000024001/endpoints/predict"
)


def _feature_vector(row: dict[str, Any]) -> list[float]:
    temperature = float(row["ambient_temp_c"])
    humidity = float(row["humidity_pct"])
    irradiance = float(row["solar_irradiance_wm2"])
    timestamp = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
    hour = timestamp.hour + timestamp.minute / 60
    temp = (temperature - 20) / 20
    humid = (humidity - 50) / 50
    solar = irradiance / 1000
    angle = 2 * math.pi * hour / 24
    return [
        1.0,
        temp,
        humid,
        solar,
        temp * temp,
        humid * humid,
        solar * solar,
        temp * humid,
        temp * solar,
        math.sin(angle),
        math.cos(angle),
    ]


def _robust_training_coefficients() -> np.ndarray | None:
    if not TRAINING_FILE.is_file():
        return None
    try:
        with TRAINING_FILE.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) < 8 or not all(
            column in (rows[0] if rows else {}) for column in (*FEATURE_COLUMNS, TARGET_COLUMN, "timestamp")
        ):
            return None
        x = np.asarray([_feature_vector(row) for row in rows], dtype=float)
        y = np.asarray([float(row[TARGET_COLUMN]) for row in rows], dtype=float)
        valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(y) < 8:
            return None

        penalty = np.eye(x.shape[1]) * 0.5
        penalty[0, 0] = 0
        weights = np.ones(len(y))
        coefficients = np.zeros(x.shape[1])
        for _ in range(20):
            weighted_x = x * np.sqrt(weights)[:, None]
            weighted_y = y * np.sqrt(weights)
            coefficients = np.linalg.solve(
                weighted_x.T @ weighted_x + penalty, weighted_x.T @ weighted_y
            )
            residuals = y - x @ coefficients
            scale = max(float(np.median(np.abs(residuals)) * 1.4826), 1.0)
            normalized = np.abs(residuals) / (1.345 * scale)
            weights = np.where(normalized <= 1, 1, 1 / normalized)
        return coefficients
    except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as error:
        print(f"Could not fit train_telemetry.csv; using heuristic regression: {error}", file=sys.stderr)
        return None


def _heuristic_coefficients() -> np.ndarray:
    # A fixed, interpretable regression prior used when no usable training set exists.
    return np.asarray([145, 26, 11, -34, 4, 1, 5, 2, -4, 10, 13], dtype=float)


def _quickml_predictions(rows: list[dict[str, Any]], endpoint: str) -> list[float]:
    parsed_url = urlparse(endpoint)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("QUICKML_ENDPOINT_URL must be a valid HTTP(S) URL")
    headers = {
        "X-QUICKML-ENDPOINT-KEY": os.getenv("QUICKML_ENDPOINT_KEY", ""),
        "CATALYST-ORG": "60092799301",
        "Environment": "Development",
        "Content-Type": "application/json",
    }
    auth_token = os.getenv("ZOHO_OAUTH_TOKEN", "").strip()
    if auth_token:
        headers["Authorization"] = f"Zoho-oauthtoken {auth_token}"
    values = []
    for row in rows:
        telemetry = {
            "timestamp": str(row["timestamp"]),
            "ambient_temp_c": float(row["ambient_temp_c"]),
            "humidity_pct": float(row["humidity_pct"]),
            "solar_irradiance_wm2": float(row["solar_irradiance_wm2"]),
        }
        payload = {"data": telemetry}
        response = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=(5, 20),
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            payload = payload.get(
                "predictions",
                payload.get("output", payload.get("data", payload.get(TARGET_COLUMN))),
            )
        if isinstance(payload, dict):
            payload = payload.get(TARGET_COLUMN)
        if isinstance(payload, list):
            if len(payload) != 1:
                raise ValueError("QuickML must return one prediction per telemetry row")
            payload = payload[0]
        if isinstance(payload, dict):
            payload = payload.get(TARGET_COLUMN)
        values.append(float(payload))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("QuickML returned a non-finite prediction")
    return values


def predict_rows(rows: list[dict[str, Any]], quickml_url: str | None = None) -> list[float]:
    """Return one finite demand estimate for each telemetry row."""
    if quickml_url is None:
        quickml_url = (
            os.environ.get("QUICKML_ENDPOINT_URL", "").strip()
            or QUICKML_ENDPOINT_URL
        )
    if not rows:
        return []
    if quickml_url:
        try:
            return _quickml_predictions(rows, quickml_url)
        except requests.HTTPError:
            print(
                "QuickML API authentication required; using local regression fallback.",
                file=sys.stderr,
            )
        except (requests.RequestException, ValueError, TypeError, KeyError) as error:
            print(f"QuickML inference failed; using local regression: {error}", file=sys.stderr)

    coefficients = _robust_training_coefficients()
    if coefficients is None:
        coefficients = _heuristic_coefficients()
    matrix = np.asarray([_feature_vector(row) for row in rows], dtype=float)
    predictions = matrix @ coefficients
    return [float(np.clip(value, 0, 500)) for value in predictions]


def predict(input_df: pd.DataFrame) -> np.ndarray:
    """Predict demand for telemetry rows, returning a finite 1D float array."""
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("input_df must be a pandas DataFrame")
    if input_df.empty:
        return np.asarray([], dtype=float)

    required = {"timestamp", *FEATURE_COLUMNS}
    missing = required.difference(input_df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {', '.join(sorted(missing))}")
    rows = input_df.to_dict(orient="records")
    for row in rows:
        timestamp = row["timestamp"]
        if isinstance(timestamp, (datetime, pd.Timestamp)):
            row["timestamp"] = timestamp.isoformat()
        for column in FEATURE_COLUMNS:
            row[column] = float(row[column])

    values = np.asarray(predict_rows(rows), dtype=float).reshape(-1)
    if values.size != len(input_df) or not np.isfinite(values).all():
        raise ValueError("Predictions must contain one finite value per input row")
    return values


def _read_telemetry(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"timestamp", *FEATURE_COLUMNS}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or []))
            raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")
        rows = list(reader)
    for index, row in enumerate(rows, start=2):
        try:
            for column in FEATURE_COLUMNS:
                number = float(row[column])
                if not math.isfinite(number):
                    raise ValueError(f"{column} must be finite")
            datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid telemetry at CSV row {index}: {error}") from error
    return rows


def _write_predictions(path: Path, rows: Iterable[dict[str, str]], values: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp", TARGET_COLUMN])
        writer.writeheader()
        for row, value in zip(rows, values, strict=True):
            writer.writerow({"timestamp": row["timestamp"], TARGET_COLUMN: float(value)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast microgrid demand from telemetry CSV.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = _read_telemetry(args.input)
    frame = pd.DataFrame(rows)
    predictions = predict(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_predictions(args.output, rows, predictions.tolist())


if __name__ == "__main__":
    main()
