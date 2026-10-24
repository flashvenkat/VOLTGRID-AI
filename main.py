from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from starlette.datastructures import UploadFile

from predict import predict

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
SAMPLE_FILE = ROOT / "sample_test_input.csv"
EVENING_START = 18
EVENING_END = 22
STORAGE_THRESHOLD_KW = 160
OVERLOAD_THRESHOLD_KW = 200
MAX_TELEMETRY_POINTS = 1000

app = FastAPI(title="VOLTGRID AI", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class TelemetryPoint(BaseModel):
    timestamp: str
    ambient_temp_c: float = Field(ge=-80, le=80)
    humidity_pct: float = Field(ge=0, le=100)
    solar_irradiance_wm2: float = Field(ge=0, le=2000)


class ForecastRequest(BaseModel):
    telemetry: list[TelemetryPoint] = Field(min_length=1, max_length=1000)


def _load_sample() -> list[dict[str, str]]:
    if not SAMPLE_FILE.is_file():
        raise HTTPException(status_code=500, detail="Sample telemetry file is unavailable")
    with SAMPLE_FILE.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _dispatch(timestamp: str, demand: float) -> tuple[str, bool]:
    try:
        hour = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).hour
    except ValueError as error:
        raise HTTPException(status_code=422, detail=f"Invalid timestamp: {timestamp}") from error
    storage_active = bool(demand > STORAGE_THRESHOLD_KW or EVENING_START <= hour <= EVENING_END)
    overload = bool(demand >= OVERLOAD_THRESHOLD_KW)
    return ("Renewable Storage Mode" if storage_active else "Main Grid Mode", overload)


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/v1/samples")
def samples() -> dict[str, list[dict[str, str]]]:
    return {"telemetry": _load_sample()}


@app.post("/api/v1/forecast")
async def forecast(request: Request) -> dict[str, object]:
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file") or form.get("csv")
        if not isinstance(upload, UploadFile):
            raise HTTPException(status_code=422, detail="Upload a CSV using the 'file' field")
        try:
            telemetry = pd.read_csv(io.BytesIO(await upload.read())).to_dict(orient="records")
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
            raise HTTPException(status_code=422, detail=f"Could not parse uploaded CSV: {error}") from error
    elif "application/json" in content_type:
        try:
            payload = await request.json()
            if isinstance(payload, list):
                telemetry = payload
            elif isinstance(payload, dict) and isinstance(payload.get("telemetry"), list):
                telemetry = payload["telemetry"]
            else:
                raise ValueError("JSON body must contain a telemetry array")
        except (ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    else:
        telemetry = _load_sample()

    if not telemetry:
        raise HTTPException(status_code=422, detail="At least one telemetry row is required")
    try:
        batch = ForecastRequest(telemetry=telemetry)
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=error.errors()) from error

    if len(batch.telemetry) > MAX_TELEMETRY_POINTS:
        raise HTTPException(
            status_code=422,
            detail=f"A maximum of {MAX_TELEMETRY_POINTS} telemetry rows is supported",
        )
    try:
        frame = pd.DataFrame([point.dict() for point in batch.telemetry])
        predictions = predict(frame)
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"Invalid telemetry: {error}") from error

    dispatch = []
    for row, demand in zip(frame.to_dict(orient="records"), predictions, strict=True):
        mode, overload = _dispatch(row["timestamp"], demand)
        solar = float(row["solar_irradiance_wm2"])
        dispatch.append(
            {
                **row,
                "target_demand_kw": float(demand),
                "grid_mode": mode,
                "critical_overload": overload,
                "battery_soc_pct": float(
                    round(max(8, min(98, 72 + solar / 40 - max(demand - 150, 0) / 3)), 1)
                ),
            }
        )
    return {"dispatch": dispatch}
