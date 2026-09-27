"""Paths and secrets. Secrets come from .env only."""

from datetime import date
from pathlib import Path

from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
META_DIR = DATA_DIR / "meta"

# Lower 48. Alaska/Hawaii/PR are out of DroughtCast and gridMET.
CONUS_BBOX = (-125.0, 24.5, -66.0, 50.0)

USGS_DAILY_URL = (
    "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"
)
USGS_TS_META_URL = (
    "https://api.waterdata.usgs.gov/ogcapi/v0/collections/"
    "time-series-metadata/items"
)

DISCHARGE_PARAMETER = "00060"
DAILY_MEAN_STATISTIC = "00003"

DAILY_FLOW_PATH = RAW_DIR / "usgs_daily_discharge.parquet"
WEEKLY_HIST_PATH = PROCESSED_DIR / "weekly_features.parquet"
WEEKLY_HIST_MATCHED = PROCESSED_DIR / "weekly_hist_matched.parquet"
WEEKLY_LIVE_PATH = PROCESSED_DIR / "weekly_live.parquet"
# First Monday week after the frozen Survey table (ends 2020-03-30).
LIVE_WEEKLY_START = date(2020, 3, 30)
# Daily sources start one year earlier so 365-day averages are filled.
LIVE_DAILY_WARMUP_START = date(2019, 3, 1)
RETRAIN_YEAR_PATH = PROCESSED_DIR / "retrain_design_years.parquet"
RETRAIN_POLICY_PATH = PROCESSED_DIR / "retrain_design_policies.parquet"
DROUGHT_YEAR_PATH = PROCESSED_DIR / "drought_design_years.parquet"
DROUGHT_CUTOFF_PATH = PROCESSED_DIR / "drought_design_cutoffs.parquet"
DROUGHT_RETRAIN_YEAR_PATH = PROCESSED_DIR / "drought_retrain_years.parquet"
DROUGHT_RETRAIN_CUTOFF_PATH = PROCESSED_DIR / "drought_retrain_cutoffs.parquet"
DROUGHT_HOLD_YEAR_PATH = PROCESSED_DIR / "drought_hold_years.parquet"
DROUGHT_HOLD_PATH = PROCESSED_DIR / "drought_hold_summary.parquet"
DROUGHT_MONITOR_YEAR_PATH = PROCESSED_DIR / "drought_monitor_years.parquet"
DROUGHT_GATE_YEAR_PATH = PROCESSED_DIR / "drought_gate_years.parquet"
DROUGHT_GATE_TRAIL_PATH = PROCESSED_DIR / "drought_gate_trail.parquet"
MODELS_DIR = ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"
LIVE_DECISIONS_PATH = PROCESSED_DIR / "live_decisions.parquet"
DOCS_DIR = ROOT / "docs"
# Last time the locked rule retrained in the historical replay.
BOOTSTRAP_INSTALLED = date(2020, 9, 21)

RESERVOIR_BASIN_DAILY = RAW_DIR / "reservoir_basin_daily.parquet"
SSEBOP_BASIN_MONTHLY = RAW_DIR / "ssebop_basin_monthly.parquet"
OPENET_POINT_MONTHLY = RAW_DIR / "openet_point_monthly.parquet"


def load_env() -> None:
    load_dotenv(ROOT / ".env")


def usgs_api_key() -> str:
    load_env()
    key = os.getenv("USGS_WATER_DATA_API_KEY", "").strip()
    if not key:
        raise RuntimeError("USGS_WATER_DATA_API_KEY is missing from .env")
    return key


def earthdata_credentials() -> tuple[str, str]:
    load_env()
    user = os.getenv("EARTHDATA_USERNAME", "").strip()
    password = os.getenv("EARTHDATA_PASSWORD", "").strip()
    if not user or not password:
        raise RuntimeError("EARTHDATA_USERNAME/PASSWORD missing from .env")
    return user, password


def openet_api_key() -> str:
    load_env()
    key = os.getenv("OPENET_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENET_API_KEY is missing from .env")
    return key


def smtp_settings() -> dict[str, str] | None:
    """Return SMTP fields, or None if email is not configured."""
    load_env()
    to_addr = os.getenv("ALERT_EMAIL_TO", "").strip()
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    if not to_addr or not user or not password:
        return None
    return {
        "to": to_addr,
        "host": os.getenv("SMTP_HOST", "smtp.gmail.com").strip() or "smtp.gmail.com",
        "port": os.getenv("SMTP_PORT", "587").strip() or "587",
        "user": user,
        "password": password,
        "from_addr": os.getenv("SMTP_FROM", "").strip() or user,
    }
