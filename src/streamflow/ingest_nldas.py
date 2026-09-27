"""NLDAS-2 Noah soil moisture onto DroughtCast basins.

NASA publishes hourly files only. Soil water in the 10–100 cm layers
changes slowly, so we take 12:00 UTC each day as that day's value.
Units stay kilograms per square meter (water in the layer), which is
what the 1980–2020 table actually used.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from streamflow.config import RAW_DIR, WEEKLY_HIST_PATH
from streamflow.earthdata import download as edl_download
from streamflow.earthdata import session as edl_session
from streamflow.storage import atomic_parquet
from streamflow.zonal import basin_cell_index, basin_means, north_up_lat

logger = logging.getLogger(__name__)

NLDAS_DIR = RAW_DIR / "nldas"
OUT_DAILY = RAW_DIR / "nldas_basin_daily.parquet"
HOUR = "1200"
NLDAS_RES = 0.125
BASE = "https://data.gesdisc.earthdata.nasa.gov/data/NLDAS/NLDAS_NOAH0125_H.2.0"

# netcdf name -> daily column (USGS weekly name drops the day grain)
LAYERS = {
    "SoilM_0_10cm": "soilm_0_10cm",
    "SoilM_10_40cm": "soilm_10_40cm",
    "SoilM_40_100cm": "soilm_40_100cm",
}


def nldas_url(day: date) -> str:
    doy = day.strftime("%j")
    name = f"NLDAS_NOAH0125_H.A{day:%Y%m%d}.{HOUR}.020.nc"
    return f"{BASE}/{day:%Y}/{doy}/{name}"


def nldas_path(day: date) -> Path:
    return NLDAS_DIR / f"{day:%Y}" / day.strftime("%j") / f"noah_{day:%Y%m%d}_{HOUR}.nc"


def download_day(day: date, sess) -> Path:
    dest = nldas_path(day)
    return edl_download(nldas_url(day), dest, sess=sess)


def _grid_xy(nc_path: Path) -> tuple[np.ndarray, np.ndarray, bool]:
    ds = xr.open_dataset(nc_path)
    try:
        lon = ds["lon"].values if "lon" in ds.coords else ds["longitude"].values
        lat = ds["lat"].values if "lat" in ds.coords else ds["latitude"].values
    finally:
        ds.close()
    lat, flipped = north_up_lat(lat)
    return lon, lat, flipped


def extract_day(
    nc_path: Path,
    day: date,
    ordered: list[str],
    cells: list[np.ndarray],
    flip_lat: bool,
) -> pd.DataFrame:
    ds = xr.open_dataset(nc_path)
    try:
        cols = {"StaID": ordered, "date": pd.Timestamp(day)}
        for nc_name, out_name in LAYERS.items():
            if nc_name not in ds:
                raise RuntimeError(f"{nc_path.name} missing {nc_name}: {list(ds.data_vars)}")
            arr = np.asarray(ds[nc_name].values, dtype=np.float32)
            arr = np.squeeze(arr)
            if flip_lat:
                arr = arr[::-1, :]
            cols[out_name] = basin_means(arr, ordered, cells)
    finally:
        ds.close()
    return pd.DataFrame(cols)


def build_basin_daily(
    start: date,
    end: date,
    *,
    out_path: Path = OUT_DAILY,
) -> Path:
    hist = pd.read_parquet(WEEKLY_HIST_PATH, columns=["StaID"])
    sta_ids = sorted(hist["StaID"].astype(str).unique())
    days: list[date] = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)

    first: Path | None = None
    missing: list[str] = []
    done = 0

    def _one(day: date) -> tuple[date, Path | None, str | None]:
        try:
            return day, download_day(day, edl_session()), None
        except Exception as exc:  # noqa: BLE001
            return day, None, str(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_one, day) for day in days]
        for fut in as_completed(futures):
            day, path, err = fut.result()
            done += 1
            if err:
                logger.warning("skip %s: %s", day, err)
                missing.append(str(day))
            else:
                first = path
            if done % 50 == 0 or done == 1 or done == len(days):
                logger.info("downloaded %s/%s (latest log %s)", done, len(days), day)
    if first is None:
        raise RuntimeError("No NLDAS days downloaded")

    lon, lat, flip_lat = _grid_xy(first)
    ordered, cells = basin_cell_index(lon, lat, res=NLDAS_RES, sta_ids=sta_ids)

    frames: list[pd.DataFrame] = []
    for day in days:
        path = nldas_path(day)
        if not path.exists():
            continue
        frames.append(extract_day(path, day, ordered, cells, flip_lat))
    frame = pd.concat(frames, ignore_index=True)
    if out_path.exists():
        old = pd.read_parquet(out_path)
        frame = pd.concat([old, frame], ignore_index=True)
    frame = (
        frame.sort_values(["StaID", "date"])
        .drop_duplicates(["StaID", "date"], keep="last")
        .reset_index(drop=True)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_parquet(frame, out_path)
    logger.info(
        "wrote %s rows (%s gages) %s to %s -> %s (%.1f MB); skipped %s days",
        len(frame),
        frame["StaID"].nunique(),
        pd.to_datetime(frame["date"]).min().date(),
        pd.to_datetime(frame["date"]).max().date(),
        out_path,
        out_path.stat().st_size / 1e6,
        len(missing),
    )
    return out_path


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Download NLDAS-2 soil moisture and average onto gage basins."
    )
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-09-21")
    args = parser.parse_args(argv)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    build_basin_daily(start, end)


if __name__ == "__main__":
    main()
