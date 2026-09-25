"""Pull CONUS daily mean discharge and write parquet.

Designed to be run on a schedule: pass --since last date, or a full window.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from streamflow.config import (
    CONUS_BBOX,
    DAILY_MEAN_STATISTIC,
    DISCHARGE_PARAMETER,
    RAW_DIR,
    USGS_DAILY_URL,
)
from streamflow.usgs_client import UsgsClient

logger = logging.getLogger(__name__)

OUT_PATH = RAW_DIR / "usgs_daily_discharge.parquet"


def ingest_daily_discharge(
    start: date,
    end: date,
    *,
    client: UsgsClient | None = None,
    out_path: Path = OUT_PATH,
) -> Path:
    """Download daily mean discharge (parameter 00060, statistic 00003)."""
    client = client or UsgsClient()
    windows = list(_month_windows(start, end))
    got_any = False
    for window_start, window_end in windows:
        logger.info("window %s to %s", window_start, window_end)
        try:
            n = _ingest_window(client, window_start, window_end, out_path)
        except RuntimeError:
            if (window_end - window_start).days <= 7:
                raise
            logger.warning(
                "month %s to %s failed; retrying by week",
                window_start,
                window_end,
            )
            n = 0
            for week_start, week_end in _week_windows(window_start, window_end):
                logger.info("week %s to %s", week_start, week_end)
                n += _ingest_window(client, week_start, week_end, out_path)
        got_any = got_any or n > 0
    if not got_any:
        raise RuntimeError(
            f"No daily discharge rows for {start} to {end}. Check the API key."
        )
    return out_path


def _ingest_window(
    client: UsgsClient, start: date, end: date, out_path: Path
) -> int:
    west, south, east, north = CONUS_BBOX
    params = {
        "parameter_code": DISCHARGE_PARAMETER,
        "statistic_id": DAILY_MEAN_STATISTIC,
        "datetime": f"{start.isoformat()}/{end.isoformat()}",
        "bbox": f"{west},{south},{east},{north}",
    }
    rows: list[dict] = []
    for feat in client.iter_features(USGS_DAILY_URL, params):
        props = feat.get("properties") or {}
        value = props.get("value")
        when = props.get("time")
        site = props.get("monitoring_location_id")
        if when is None or site is None:
            continue
        rows.append(
            {
                "monitoring_location_id": site,
                "date": when[:10],
                "discharge_cfs": _to_float(value),
                "qualifier": _first_qualifier(props.get("qualifier")),
                "approval_status": props.get("approval_status"),
                "time_series_id": props.get("time_series_id"),
            }
        )
    if not rows:
        logger.warning("no rows for %s to %s", start, end)
        return 0

    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = (
        frame.sort_values(["monitoring_location_id", "date"])
        .drop_duplicates(
            ["monitoring_location_id", "date", "time_series_id"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        frame = (
            pd.concat([existing, frame], ignore_index=True)
            .sort_values(["monitoring_location_id", "date"])
            .drop_duplicates(
                ["monitoring_location_id", "date", "time_series_id"],
                keep="last",
            )
            .reset_index(drop=True)
        )
    frame.to_parquet(out_path, index=False)
    logger.info(
        "wrote %s rows (%s sites) to %s",
        len(frame),
        frame["monitoring_location_id"].nunique(),
        out_path,
    )
    return len(rows)


def _month_windows(start: date, end: date) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        if cursor.month == 12:
            next_month = date(cursor.year + 1, 1, 1)
        else:
            next_month = date(cursor.year, cursor.month + 1, 1)
        window_end = min(end, next_month - timedelta(days=1))
        windows.append((cursor, window_end))
        cursor = window_end + timedelta(days=1)
    return windows


def _week_windows(start: date, end: date) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        window_end = min(end, cursor + timedelta(days=6))
        windows.append((cursor, window_end))
        cursor = window_end + timedelta(days=1)
    return windows


def _to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_qualifier(raw: object) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        return ",".join(str(x) for x in raw) if raw else None
    return str(raw)


def _parse_date(text: str) -> date:
    return date.fromisoformat(text)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    today = datetime.now(timezone.utc).date()
    parser = argparse.ArgumentParser(
        description="Ingest CONUS USGS daily mean discharge."
    )
    parser.add_argument(
        "--start",
        type=_parse_date,
        default=today - timedelta(days=90),
        help="Inclusive start date (YYYY-MM-DD). Default: 90 days ago.",
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        default=today,
        help="Inclusive end date (YYYY-MM-DD). Default: today (UTC).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_PATH,
        help=f"Parquet path. Default: {OUT_PATH}",
    )
    args = parser.parse_args(argv)
    if args.end < args.start:
        raise SystemExit("--end must be on or after --start")
    ingest_daily_discharge(args.start, args.end, out_path=args.out)


if __name__ == "__main__":
    main()
