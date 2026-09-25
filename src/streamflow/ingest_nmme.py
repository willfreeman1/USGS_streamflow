"""NMME ensemble-mean monthly forecasts onto DroughtCast basins.

CPC publishes the ensemble mean as an anomaly (Kelvin for temperature,
mm/day for precipitation). We store the basin-mean anomaly. When we
join it to the weekly table we add each station's 1980–2020 monthly
mean from the USGS file so the column is on the same scale as
tref15day_nmme_mean / precip15day_nmme_mean.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from streamflow.config import RAW_DIR, WEEKLY_HIST_PATH
from streamflow.zonal import basin_cell_index, north_up_lat

logger = logging.getLogger(__name__)

NMME_DIR = RAW_DIR / "nmme"
OUT_MONTHLY = RAW_DIR / "nmme_basin_monthly.parquet"
BASE = "https://ftp.cpc.ncep.noaa.gov/NMME/realtime_anom/ENSMEAN"
USER_AGENT = "usgs-streamflow-monitor (local)"
NMME_RES = 1.0
LEADS = (0, 1, 2, 3)
LEAD_DAYS = {0: 15, 1: 45, 2: 75, 3: 105}
VARIABLES = {
    "tmp2m": "tref",
    "prate": "precip",
}


def _curl(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
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
        raise RuntimeError(f"NMME download failed ({result.returncode}): {url}")


def download_month(var: str, year: int, month: int) -> Path:
    stamp = f"{year}{month:02d}"
    dest = NMME_DIR / f"NMME.{var}.{stamp}.ENSMEAN.anom.nc"
    if dest.exists() and dest.stat().st_size > 100_000:
        return dest
    url = (
        f"{BASE}/{year}{month:02d}0800/NMME.{var}.{stamp}.ENSMEAN.anom.nc"
    )
    logger.info("download %s", url)
    _curl(url, dest)
    return dest


def _shift_lon(lon: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    order = np.argsort(lon)
    return lon[order], grid[..., order]


def _index(sta_ids: list[str], sample: Path):
    ds = xr.open_dataset(sample, decode_times=False)
    try:
        lon = np.asarray(ds["lon"].values)
        lat = np.asarray(ds["lat"].values)
        grid = np.asarray(ds["fcst"].values)
    finally:
        ds.close()
    lon, _ = _shift_lon(lon, grid)
    lat, _ = north_up_lat(lat)
    return basin_cell_index(lon, lat, res=NMME_RES, sta_ids=sta_ids)


def _extract(path: Path, var: str, ordered: list[str], cells: list[np.ndarray]) -> pd.DataFrame:
    ds = xr.open_dataset(path, decode_times=False)
    try:
        lon = np.asarray(ds["lon"].values)
        lat = np.asarray(ds["lat"].values)
        grid = np.asarray(ds["fcst"].values, dtype=np.float32)
        init = float(np.asarray(ds["initial_time"].values).ravel()[0])
    finally:
        ds.close()
    lon, grid = _shift_lon(lon, grid)
    lat, flipped = north_up_lat(lat)
    if flipped:
        grid = grid[:, ::-1, :]
    # initial_time is months since 1960-01
    year = 1960 + int(init) // 12
    month = int(init) % 12 + 1
    rows = []
    for lead in LEADS:
        if lead >= grid.shape[0]:
            continue
        means = np.full(len(ordered), np.nan, dtype=np.float32)
        flat = grid[lead].ravel()
        for j, idx in enumerate(cells):
            with np.errstate(all="ignore"):
                means[j] = np.nanmean(flat[idx])
        # CPC prate anomalies are millimeters per second.
        if var == "prate":
            means = means * 86400.0
        col = f"{VARIABLES[var]}{LEAD_DAYS[lead]}day_nmme_anom"
        frame = pd.DataFrame(
            {
                "StaID": ordered,
                "year": year,
                "month": month,
                col: means,
            }
        )
        rows.append(frame)
    out = rows[0]
    for extra in rows[1:]:
        out = out.merge(extra, on=["StaID", "year", "month"], how="outer")
    return out


def build_basin_monthly(
    start: date,
    end: date,
    *,
    hist_path: Path = WEEKLY_HIST_PATH,
    out_path: Path = OUT_MONTHLY,
) -> Path:
    hist = pd.read_parquet(hist_path, columns=["StaID"])
    sta_ids = sorted(hist["StaID"].astype(str).unique())
    sample = download_month("tmp2m", start.year, start.month)
    ordered, cells = _index(sta_ids, sample)

    pieces: list[pd.DataFrame] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        month_frame: pd.DataFrame | None = None
        for var in VARIABLES:
            try:
                path = download_month(var, year, month)
            except RuntimeError:
                logger.warning("skip NMME %s %s-%02d", var, year, month)
                continue
            one = _extract(path, var, ordered, cells)
            month_frame = (
                one
                if month_frame is None
                else month_frame.merge(one, on=["StaID", "year", "month"], how="outer")
            )
        if month_frame is not None:
            pieces.append(month_frame)
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

    if not pieces:
        raise RuntimeError("No NMME months downloaded")
    frame = pd.concat(pieces, ignore_index=True)
    if out_path.exists():
        old = pd.read_parquet(out_path)
        frame = pd.concat([old, frame], ignore_index=True)
    frame = (
        frame.sort_values(["StaID", "year", "month"])
        .drop_duplicates(["StaID", "year", "month"], keep="last")
        .reset_index(drop=True)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    logger.info(
        "wrote %s rows (%s stations) -> %s (%.1f MB)",
        len(frame),
        frame["StaID"].nunique(),
        out_path,
        out_path.stat().st_size / 1e6,
    )
    return out_path


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Download NMME ensemble-mean forecasts onto drainage areas."
    )
    parser.add_argument("--start", default="2024-09-01")
    parser.add_argument("--end", default=date.today().replace(day=1).isoformat())
    args = parser.parse_args(argv)
    build_basin_monthly(date.fromisoformat(args.start), date.fromisoformat(args.end))


if __name__ == "__main__":
    main()
