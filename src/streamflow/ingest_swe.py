"""University of Arizona 4 km snow water equivalent onto DroughtCast basins.

Same drainage-area mean as gridMET. Units stay millimeters of water.
Yearly water-year files cover completed years. The current water year
is one NetCDF per day. Files are deleted after extract.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from streamflow.config import RAW_DIR, WEEKLY_HIST_PATH
from streamflow.storage import atomic_parquet
from streamflow.zonal import basin_cell_index, north_up_lat

logger = logging.getLogger(__name__)

SWE_DIR = RAW_DIR / "ua_swe"
OUT_DAILY = RAW_DIR / "swe_basin_daily.parquet"
BASE = "https://climate.arizona.edu/data/UA_SWE"
USER_AGENT = "usgs-streamflow-monitor (local)"
SWE_RES = 1.0 / 24.0


def _curl(url: str, dest: Path, *, retries: int = 6, retry_all: bool = True) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl.exe",
        "-fL",
        "-C",
        "-",
        "-A",
        USER_AGENT,
        "--retry",
        str(retries),
        "--retry-delay",
        "4",
        "-o",
        str(dest),
        url,
    ]
    if retry_all:
        cmd.insert(-3, "--retry-all-errors")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"SWE download failed ({result.returncode}): {url}")


def download_water_year(wy: int) -> Path:
    dest = SWE_DIR / f"UA_SWE_Depth_WY{wy}.nc"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    url = f"{BASE}/WYData_4km/UA_SWE_Depth_WY{wy}.nc"
    logger.info("download %s", url)
    _curl(url, dest)
    return dest


def download_daily(day: date) -> Path | None:
    dest = SWE_DIR / "daily" / f"UA_SWE_{day:%Y%m%d}.nc"
    if dest.exists() and dest.stat().st_size > 10_000:
        return dest
    wy = day.year + 1 if day.month >= 10 else day.year
    for suffix in ("stable", "provisional"):
        name = f"UA_SWE_Depth_4km_v1_{day:%Y%m%d}_{suffix}.nc"
        url = f"{BASE}/DailyData_4km/WY{wy}/{name}"
        try:
            logger.info("download %s", url)
            # 404 means that suffix is not published; do not retry it.
            _curl(url, dest, retries=0, retry_all=False)
            if dest.exists() and dest.stat().st_size > 10_000:
                return dest
        except RuntimeError:
            if dest.exists():
                dest.unlink()
    logger.warning("no UA SWE file for %s", day)
    return None


def _index_from_nc(nc_path: Path, sta_ids: list[str]):
    ds = xr.open_dataset(nc_path)
    try:
        lon = np.asarray(ds["lon"].values)
        lat = np.asarray(ds["lat"].values)
    finally:
        ds.close()
    lat, flipped = north_up_lat(lat)
    ordered, cells = basin_cell_index(
        lon, lat, res=SWE_RES, sta_ids=sta_ids
    )
    return ordered, cells, flipped


def _extract(nc_path: Path, ordered: list[str], cells: list[np.ndarray], flipped: bool) -> pd.DataFrame:
    ds = xr.open_dataset(nc_path)
    try:
        swe = np.asarray(ds["SWE"].values, dtype=np.float32)
        if "time" in ds.coords:
            days = pd.to_datetime(ds["time"].values)
        else:
            days = pd.to_datetime(ds["day"].values) if "day" in ds.coords else None
    finally:
        ds.close()
    if swe.ndim == 2:
        swe = swe[np.newaxis, ...]
        if days is None:
            raise RuntimeError(f"{nc_path} has no time coordinate")
    if flipped:
        swe = swe[:, ::-1, :]
    n_days, n_lat, n_lon = swe.shape
    flat = swe.reshape(n_days, n_lat * n_lon)
    means = np.full((n_days, len(ordered)), np.nan, dtype=np.float32)
    for j, idx in enumerate(cells):
        block = flat[:, idx]
        with np.errstate(all="ignore"):
            means[:, j] = np.nanmean(block, axis=1)
    if days is None:
        raise RuntimeError(f"{nc_path} has no time coordinate")
    days = pd.DatetimeIndex(np.atleast_1d(pd.to_datetime(days)))
    if len(days) != n_days:
        raise RuntimeError(f"{nc_path}: {len(days)} times vs {n_days} SWE layers")
    return pd.DataFrame(
        {
            "StaID": np.repeat(np.array(ordered, dtype=object), n_days),
            "date": np.tile(days, len(ordered)),
            "swe_mean": means.T.ravel(),
        }
    )


def build_basin_daily(
    start: date,
    end: date,
    *,
    hist_path: Path = WEEKLY_HIST_PATH,
    out_path: Path = OUT_DAILY,
) -> Path:
    hist = pd.read_parquet(hist_path, columns=["StaID"])
    sta_ids = sorted(hist["StaID"].astype(str).unique())
    sample = download_water_year(2025)
    ordered, cells, flipped = _index_from_nc(sample, sta_ids)

    pieces: list[pd.DataFrame] = []
    have: set[pd.Timestamp] = set()
    if out_path.exists():
        old = pd.read_parquet(out_path, columns=["date"])
        have = set(pd.to_datetime(old["date"]).unique())
        logger.info("resume SWE: %s days already on disk", len(have))
    wys = set()
    cursor = start
    while cursor <= end:
        wy = cursor.year + 1 if cursor.month >= 10 else cursor.year
        wys.add(wy)
        cursor = date(cursor.year + 1, 1, 1)

    today = date.today()
    current_wy = today.year + 1 if today.month >= 10 else today.year
    for wy in sorted(wys):
        wy_start = date(wy - 1, 10, 1)
        wy_end = date(wy, 9, 30)
        overlap_start = max(start, wy_start)
        overlap_end = min(end, wy_end)
        if overlap_start > overlap_end:
            continue
        if wy < current_wy:
            path = download_water_year(wy)
            frame = _extract(path, ordered, cells, flipped)
            mask = (frame["date"] >= pd.Timestamp(overlap_start)) & (
                frame["date"] <= pd.Timestamp(overlap_end)
            )
            frame = frame.loc[mask]
            if have:
                frame = frame.loc[~frame["date"].isin(have)]
            if not frame.empty:
                pieces.append(frame)
            continue
        day = overlap_start
        while day <= overlap_end:
            if pd.Timestamp(day) in have:
                day += timedelta(days=1)
                continue
            path = download_daily(day)
            if path is not None:
                pieces.append(_extract(path, ordered, cells, flipped))
                path.unlink(missing_ok=True)
            day += timedelta(days=1)

    if not pieces and not out_path.exists():
        raise RuntimeError("No UA SWE rows extracted")
    if pieces:
        frame = pd.concat(pieces, ignore_index=True)
        if out_path.exists():
            frame = pd.concat(
                [pd.read_parquet(out_path), frame], ignore_index=True
            )
        frame = frame.sort_values(["StaID", "date"]).drop_duplicates(
            ["StaID", "date"], keep="last"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_parquet(frame, out_path)
    else:
        frame = pd.read_parquet(out_path)
    logger.info(
        "wrote %s rows (%s stations) %s to %s -> %s (%.1f MB)",
        len(frame),
        frame["StaID"].nunique(),
        pd.to_datetime(frame["date"]).min().date(),
        pd.to_datetime(frame["date"]).max().date(),
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
        description="Download UA SWE and average onto drainage areas."
    )
    parser.add_argument("--start", default="2024-09-01")
    parser.add_argument("--end", default=date.today().isoformat())
    args = parser.parse_args(argv)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    build_basin_daily(start, end)


if __name__ == "__main__":
    main()
