"""Basic checks that the live matched-set files exist and sit on USGS scales."""

from __future__ import annotations

from datetime import date
import math
from typing import cast

import numpy as np
import pandas as pd
import pytest

from streamflow.config import (
    LIVE_DAILY_WARMUP_START,
    RAW_DIR,
    WEEKLY_HIST_PATH,
    WEEKLY_LIVE_PATH,
)
from streamflow.ingest_flow import select_daily_discharge
from streamflow.matched import (
    CLIMATE_PATH,
    GEFS_DAILY,
    NMME_MONTHLY,
    STATIC_PATH,
    SWE_DAILY,
    add_rolls,
    attach_climate,
    attach_static,
    attach_swe,
)

LEES = "09380000"
GEFS_END = date(2026, 9, 14)
pytestmark = pytest.mark.integration


def test_select_daily_discharge_prefers_approved_then_established_series():
    frame = pd.DataFrame(
        [
            {
                "monitoring_location_id": "USGS-00000001",
                "date": "2026-01-01",
                "discharge_cfs": 10.0,
                "qualifier": None,
                "approval_status": "Approved",
                "time_series_id": "established",
            },
            {
                "monitoring_location_id": "USGS-00000001",
                "date": "2026-01-02",
                "discharge_cfs": 11.0,
                "qualifier": None,
                "approval_status": "Approved",
                "time_series_id": "established",
            },
            {
                "monitoring_location_id": "USGS-00000001",
                "date": "2026-01-02",
                "discharge_cfs": 99.0,
                "qualifier": None,
                "approval_status": "Approved",
                "time_series_id": "short-experiment",
            },
            {
                "monitoring_location_id": "USGS-00000002",
                "date": "2026-01-02",
                "discharge_cfs": 20.0,
                "qualifier": None,
                "approval_status": "Approved",
                "time_series_id": "approved",
            },
            {
                "monitoring_location_id": "USGS-00000002",
                "date": "2026-01-02",
                "discharge_cfs": 30.0,
                "qualifier": None,
                "approval_status": "Provisional",
                "time_series_id": "provisional",
            },
        ]
    )
    selected = select_daily_discharge(frame)
    values = selected.set_index(["monitoring_location_id", "date"])["discharge_cfs"]
    assert values.loc[("USGS-00000001", "2026-01-02")] == 11.0
    assert values.loc[("USGS-00000002", "2026-01-02")] == 20.0


def test_climate_monthly_scales_and_coverage():
    df = pd.read_parquet(CLIMATE_PATH)
    assert {"year", "month", "ENSO", "PNA", "PDO", "AMO", "sunspots"} <= set(df.columns)
    assert not df.duplicated(["year", "month"]).any()
    enso = df.dropna(subset=["ENSO"])
    last = enso.iloc[-1]
    assert 24.0 <= float(enso["ENSO"].min()) <= 26.0
    assert 28.0 <= float(enso["ENSO"].max()) <= 31.0
    assert int(last["year"]) >= 2026
    assert int(last["month"]) >= 8
    pdo = df["PDO"].dropna()
    assert not pdo.empty
    assert (pdo.abs() < 90).all()
    amo = df.dropna(subset=["AMO"])
    assert int(amo.iloc[-1]["year"]) >= 2026


def test_swe_season_and_lees():
    df = pd.read_parquet(SWE_DAILY)
    assert df["StaID"].nunique() == 3229
    assert not df.duplicated(["StaID", "date"]).any()
    assert pd.to_datetime(df["date"]).min().date() == LIVE_DAILY_WARMUP_START
    assert pd.to_datetime(df["date"]).max().date() >= date(2026, 8, 31)
    assert float(df["swe_mean"].min()) >= 0.0
    assert float(df["swe_mean"].max()) < 3000.0
    df = df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.month
    jan = float(df.loc[df["month"] == 1, "swe_mean"].mean())
    jul = float(df.loc[df["month"] == 7, "swe_mean"].mean())
    assert jan > 20.0
    assert jul < 2.0
    lees = df.loc[df["StaID"] == LEES]
    jan_lees = float(lees.loc[lees["month"] == 1, "swe_mean"].mean())
    jul_lees = float(lees.loc[lees["month"] == 7, "swe_mean"].mean())
    assert jan_lees > jul_lees
    assert jul_lees < 1.0


def test_nmme_months_and_units():
    df = pd.read_parquet(NMME_MONTHLY)
    assert df["StaID"].nunique() == 3229
    months = df.groupby(["year", "month"]).size()
    assert len(months) >= 25
    assert (LIVE_DAILY_WARMUP_START.year, LIVE_DAILY_WARMUP_START.month) in months.index
    assert (2024, 9) in months.index
    assert (2026, 9) in months.index
    t = df["tref15day_nmme_anom"]
    p = df["precip15day_nmme_anom"]
    assert -6.0 < float(t.min()) < 0.0
    assert 0.0 < float(t.max()) < 8.0
    # millimeters per day, not millimeters per second
    assert -8.0 < float(p.min()) < 0.0
    assert 0.5 < float(p.max()) < 15.0
    assert abs(float(p.mean())) < 1.0


def test_static_file_has_area():
    static = pd.read_csv(STATIC_PATH, usecols=["StaID", "S-SqKm"])
    static["StaID"] = static["StaID"].astype(str).str.zfill(8)
    lees = static.loc[static["StaID"] == LEES, "S-SqKm"]
    assert not lees.empty
    assert 250_000 < float(lees.iloc[0]) < 400_000


def test_gefs_scale_and_no_duplicates():
    if not GEFS_DAILY.exists():
        pytest.skip("GEFS daily file not written yet")
    df = pd.read_parquet(GEFS_DAILY)
    assert df["StaID"].nunique() >= 3220
    assert not df.duplicated(["StaID", "date"]).any()
    tmax = df["tmax7day_gefs_mean"]
    tp = df["tp7day_gefs_mean"]
    assert -40.0 < float(tmax.min()) < 15.0
    assert 20.0 < float(tmax.max()) < 50.0
    assert float(tp.min()) >= 0.0
    assert float(tp.max()) < 40.0
    assert 0.1 < float(tp.mean()) < 3.0
    lees = df.loc[df["StaID"] == LEES]
    assert not lees.empty


def test_gefs_complete_through_target():
    if not GEFS_DAILY.exists():
        pytest.skip("GEFS daily file not written yet")
    df = pd.read_parquet(GEFS_DAILY, columns=["date"])
    last = pd.to_datetime(df["date"]).max().date()
    if last < GEFS_END:
        pytest.skip(f"GEFS still downloading (last day {last})")
    n_days = pd.to_datetime(df["date"]).nunique()
    assert n_days >= 700


def test_attach_helpers_on_tiny_week():
    weekly = pd.DataFrame(
        {
            "StaID": [LEES],
            "Date": [pd.Timestamp("2025-01-13")],
            "week_dt": [pd.Timestamp("2025-01-13")],
            "year": [2025],
            "month": [1],
            "jd": [13],
            "Flow_cfs": [8000.0],
        }
    )
    weekly = attach_static(weekly)
    weekly = attach_swe(weekly)
    weekly = attach_climate(weekly)
    weekly = add_rolls(weekly)
    area = cast(float, weekly.loc[0, "S-SqKm"])
    flow_mm = cast(float, weekly.loc[0, "Flow_mm"])
    swe = cast(float, weekly.loc[0, "swe_mean"])
    enso = cast(float, weekly.loc[0, "ENSO"])
    assert area > 0
    assert math.isfinite(flow_mm)
    assert swe >= 0
    assert 24.0 < enso < 31.0
    assert WEEKLY_HIST_PATH.exists()
    assert WEEKLY_LIVE_PATH.exists()
    assert RAW_DIR.exists()
