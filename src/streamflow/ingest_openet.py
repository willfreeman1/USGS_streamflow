"""Live actual evapotranspiration onto DroughtCast basins.

Two products, on purpose:

1. USGS SSEBop MODIS monthly, 1 km. This is the same family as the
   1980–2020 `aet_mean` column (dictionary: NASA MODIS / SSEBOP). Free
   GeoTIFF download, no query cap. We average each month onto every basin.

2. OpenET Ensemble at a 5-degree point grid. The free OpenET account is
   100 queries/month and 50,000 acres. Whole-basin polygons and a 30 m
   national raster do not fit. One monthly timeseries per grid point uses
   one query and covers 2024–2026. Each gage gets the nearest point.

OpenET is the irrigation-season check. SSEBop is the national series that
can sit next to the historical column.
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import subprocess
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from streamflow.config import RAW_DIR, WEEKLY_HIST_PATH, openet_api_key
from streamflow.zonal import (
    basin_cell_index,
    basin_means,
    load_droughtcast_polygons,
    north_up_lat,
)

logger = logging.getLogger(__name__)

SSEBOP_DIR = RAW_DIR / "ssebop"
OUT_SSEBOP = RAW_DIR / "ssebop_basin_monthly.parquet"
OUT_OPENET = RAW_DIR / "openet_point_monthly.parquet"
SSEBOP_INDEX = (
    "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/"
    "eta/modis_eta/monthly/downloads/"
)
OPENET_POINT = "https://openet-api.org/raster/timeseries/point"
USER_AGENT = "usgs-streamflow-monitor (local)"

# Interior CONUS 5-degree cells plus Lees Ferry (Colorado / Powell).
GRID_LONS = [-122.5, -117.5, -112.5, -107.5, -102.5, -97.5, -92.5, -87.5, -82.5, -77.5]
GRID_LATS = [27.5, 32.5, 37.5, 42.5, 47.5]
EXTRA_POINTS = (("lees_ferry", -111.51, 36.86),)


def ingest_openet(
    start: date,
    end: date,
    *,
    skip_ssebop: bool = False,
    skip_points: bool = False,
) -> None:
    if not skip_ssebop:
        ingest_ssebop(start, end)
    if not skip_points:
        ingest_openet_points(start, end)


def ingest_ssebop(start: date, end: date) -> Path:
    months = _month_starts(start, end)
    zips = []
    for month in months:
        path = _download_ssebop(month)
        if path is not None:
            zips.append((month, path))
    if not zips:
        raise RuntimeError("No SSEBop monthly zips downloaded")

    hist = pd.read_parquet(WEEKLY_HIST_PATH, columns=["StaID"])
    sta_ids = sorted(hist["StaID"].astype(str).unique())
    sample = _open_ssebop_tif(zips[0][1])
    if sample is None:
        raise RuntimeError("Could not open first SSEBop GeoTIFF")
    with sample:
        lon, lat = _ssebop_lonlat(sample)
        res = float(abs(sample.res[0]))
    lat, _ = north_up_lat(lat)
    ordered, cells = basin_cell_index(lon, lat, res=res, sta_ids=sta_ids)

    frames: list[pd.DataFrame] = []
    for month, path in zips:
        tif = _open_ssebop_tif(path)
        if tif is None:
            continue
        with tif:
            grid = tif.read(1).astype(np.float32)
            if tif.transform.e > 0:
                grid = grid[::-1]
        grid = np.where(grid <= 0, np.nan, grid)
        means = basin_means(grid, ordered, cells)
        frames.append(
            pd.DataFrame(
                {
                    "StaID": ordered,
                    "date": pd.Timestamp(month),
                    "aet_mm_month": means,
                }
            )
        )
        logger.info(
            "SSEBop %s: national mean %.1f mm, nulls %s",
            month,
            float(np.nanmean(means)),
            int(np.isnan(means).sum()),
        )
    table = pd.concat(frames, ignore_index=True)
    days = table["date"].dt.days_in_month
    table["aet_mean"] = table["aet_mm_month"] / days
    table = table.sort_values(["StaID", "date"]).reset_index(drop=True)
    OUT_SSEBOP.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT_SSEBOP, index=False)
    logger.info(
        "wrote %s rows (%s gages) %s to %s -> %s (%.1f MB)",
        len(table),
        table["StaID"].nunique(),
        table["date"].min().date(),
        table["date"].max().date(),
        OUT_SSEBOP,
        OUT_SSEBOP.stat().st_size / 1e6,
    )
    return OUT_SSEBOP


def _download_ssebop(month: date) -> Path | None:
    SSEBOP_DIR.mkdir(parents=True, exist_ok=True)
    name = f"m{month:%Y%m}.zip"
    dest = SSEBOP_DIR / name
    if dest.exists() and dest.stat().st_size > 100_000:
        return dest
    url = SSEBOP_INDEX + name
    logger.info("download %s", url)
    cmd = [
        "curl.exe",
        "-fL",
        "-C",
        "-",
        "-A",
        USER_AGENT,
        "--retry",
        "6",
        "--retry-delay",
        "4",
        "--retry-all-errors",
        "-o",
        str(dest),
        url,
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.warning("SSEBop missing or failed: %s", url)
        if dest.exists() and dest.stat().st_size < 100_000:
            dest.unlink()
        return None
    return dest


def _open_ssebop_tif(zip_path: Path):
    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".tif")]
        if not names:
            logger.warning("no tif in %s", zip_path.name)
            return None
        name = names[0]
        data = archive.read(name)
    return rasterio.open(io.BytesIO(data))


def _ssebop_lonlat(src) -> tuple[np.ndarray, np.ndarray]:
    height, width = src.shape
    a, _b, west, _d, e, north = src.transform[:6]
    xs = west + a * (np.arange(width, dtype=np.float64) + 0.5)
    ys = north + e * (np.arange(height, dtype=np.float64) + 0.5)
    return xs, ys


def ingest_openet_points(start: date, end: date) -> Path:
    key = openet_api_key()
    points = [
        (f"grid_{lon}_{lat}", lon, lat) for lon in GRID_LONS for lat in GRID_LATS
    ]
    points.extend(EXTRA_POINTS)
    logger.info("OpenET point queries planned: %s", len(points))
    existing: set[str] = set()
    frames: list[pd.DataFrame] = []
    if OUT_OPENET.exists():
        old = pd.read_parquet(OUT_OPENET)
        existing = set(old["point_id"].astype(str))
        frames.append(old)
        logger.info("OpenET already on disk for %s points; skipping those", len(existing))

    sess = requests.Session()
    sess.headers.update(
        {
            "Authorization": key,
            "Content-Type": "application/json",
            "accept": "application/json",
        }
    )
    for point_id, lon, lat in points:
        if point_id in existing:
            continue
        payload = {
            "date_range": [start.isoformat(), end.isoformat()],
            "interval": "monthly",
            "geometry": [lon, lat],
            "model": "Ensemble",
            "variable": "ET",
            "reference_et": "gridMET",
            "units": "mm",
            "file_format": "JSON",
        }
        logger.info("OpenET %s (%.2f, %.2f)", point_id, lon, lat)
        try:
            resp = sess.post(OPENET_POINT, json=payload, timeout=120)
            if resp.status_code == 429:
                logger.warning("OpenET quota hit after %s new points", point_id)
                break
            resp.raise_for_status()
            payload = resp.json()
            if point_id == points[0][0] or not frames:
                logger.info("OpenET sample payload type=%s preview=%s", type(payload).__name__, str(payload)[:400])
            rows = _parse_openet(payload, point_id, lon, lat)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenET %s failed: %s", point_id, exc)
            continue
        if rows.empty:
            logger.warning("OpenET %s returned no rows", point_id)
            continue
        frames.append(rows)
        table = pd.concat(frames, ignore_index=True)
        table.to_parquet(OUT_OPENET, index=False)

    if not frames:
        raise RuntimeError("No OpenET point rows")
    table = pd.concat(frames, ignore_index=True)
    table = table.drop_duplicates(["point_id", "date"], keep="last")
    table = table.sort_values(["point_id", "date"]).reset_index(drop=True)
    table.to_parquet(OUT_OPENET, index=False)
    logger.info(
        "wrote %s rows (%s points) %s to %s -> %s",
        len(table),
        table["point_id"].nunique(),
        table["date"].min().date(),
        table["date"].max().date(),
        OUT_OPENET,
    )
    return OUT_OPENET


def _parse_openet(payload, point_id: str, lon: float, lat: float) -> pd.DataFrame:
    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    if isinstance(payload, dict) and payload and all(
        str(k)[:4].isdigit() for k in list(payload)[:3]
    ):
        payload = [{"time": k, "et": v} for k, v in payload.items()]
    if not isinstance(payload, list):
        return pd.DataFrame()
    rows = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        when = item.get("time") or item.get("date") or item.get("month")
        value = item.get("et") or item.get("ET") or item.get("value")
        if when is None:
            continue
        rows.append(
            {
                "point_id": point_id,
                "lon": lon,
                "lat": lat,
                "date": str(when)[:10],
                "openet_et_mm": _to_float(value),
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def attach_openet_to_gages(
    monthly: pd.DataFrame, sta_ids: list[str]
) -> pd.DataFrame:
    """Nearest OpenET grid point for each DroughtCast basin centroid."""
    ordered, geoms = load_droughtcast_polygons(sta_ids)
    cents = np.array([[g.centroid.x, g.centroid.y] for g in geoms], dtype=float)
    pts = monthly.drop_duplicates("point_id")[["point_id", "lon", "lat"]]
    xy = pts[["lon", "lat"]].to_numpy(dtype=float)
    # squared degree distance is enough at 5 degrees
    d2 = (
        (cents[:, None, 0] - xy[None, :, 0]) ** 2
        + (cents[:, None, 1] - xy[None, :, 1]) ** 2
    )
    nearest = pts["point_id"].to_numpy()[d2.argmin(axis=1)]
    map_frame = pd.DataFrame({"StaID": ordered, "point_id": nearest})
    out = monthly.merge(map_frame, on="point_id", how="inner")
    return out[["StaID", "date", "openet_et_mm", "point_id"]]


def _month_starts(start: date, end: date) -> list[date]:
    months = []
    cursor = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while cursor <= last:
        months.append(cursor)
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return months


def _to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def list_ssebop_months() -> list[str]:
    resp = requests.get(SSEBOP_INDEX, headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    return sorted(set(re.findall(r"m(20\d{4})\.zip", resp.text)))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Ingest SSEBop monthly AET and OpenET Ensemble points."
    )
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument("--skip-ssebop", action="store_true")
    parser.add_argument("--skip-points", action="store_true")
    args = parser.parse_args(argv)
    ingest_openet(
        date.fromisoformat(args.start),
        date.fromisoformat(args.end),
        skip_ssebop=args.skip_ssebop,
        skip_points=args.skip_points,
    )


if __name__ == "__main__":
    main()
