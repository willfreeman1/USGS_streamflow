"""Download live gridMET and average each day onto DroughtCast basins.

Yearly NetCDFs come from Climatology Lab. Temperatures are Kelvin in the
file; we store Celsius to match USGS. Precipitation and PET stay millimeters.

Watersheds nest (an upstream cell sits in many downstream polygons), so each
basin is averaged on its own cells. A single ID raster would steal cells.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
import xarray as xr
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

from streamflow.config import RAW_DIR, WEEKLY_HIST_PATH
from streamflow.storage import atomic_parquet

logger = logging.getLogger(__name__)

GRIDMET_DIR = RAW_DIR / "gridmet"
BASIN_SHP = RAW_DIR / "rdews_gages" / "rdews_gages.shp"
OUT_DAILY = RAW_DIR / "gridmet_basin_daily.parquet"

GRIDMET_BASE = "https://www.northwestknowledge.net/metdata/data"
USER_AGENT = "usgs-streamflow-monitor (local)"
GRID_RES = 1.0 / 24.0

# file stem -> (preferred netcdf variable, daily column, kelvin_to_c)
VARIABLES = {
    "tmmx": ("air_temperature", "tmmx", True),
    "tmmn": ("air_temperature", "tmmn", True),
    "pr": ("precipitation_amount", "pr", False),
    "pet": ("potential_evapotranspiration", "pet", False),
}


def download_year(variable: str, year: int, dest_dir: Path = GRIDMET_DIR) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{variable}_{year}.nc"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    url = f"{GRIDMET_BASE}/{variable}_{year}.nc"
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
        raise RuntimeError(f"gridMET download failed ({result.returncode}): {url}")
    return dest


def _data_var(ds: xr.Dataset, preferred: str) -> str:
    if preferred in ds.data_vars:
        return preferred
    skip = {"crs", "lat", "lon", "day", "time"}
    names = [name for name in ds.data_vars if name not in skip]
    if len(names) == 1:
        logger.info("using %s (wanted %s)", names[0], preferred)
        return names[0]
    raise RuntimeError(f"expected {preferred}, have {list(ds.data_vars)}")


def basin_cell_index(
    lon: np.ndarray,
    lat: np.ndarray,
    sta_ids: list[str],
    shp_path: Path = BASIN_SHP,
) -> tuple[list[str], list[np.ndarray]]:
    """Per-basin flat cell indices on the gridMET grid (WGS84)."""
    if not shp_path.exists():
        raise RuntimeError(f"Missing {shp_path}; wait for ingest-basins")
    wanted = set(sta_ids)
    transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    reader = shapefile.Reader(str(shp_path.with_suffix("")))
    field_names = [field[0] for field in reader.fields[1:]]
    sta_idx = field_names.index("StaID")
    west = float(lon[0]) - GRID_RES / 2.0
    north = float(lat[0]) + GRID_RES / 2.0
    transform = from_origin(west, north, GRID_RES, GRID_RES)
    out_shape = (int(lat.size), int(lon.size))

    ordered: list[str] = []
    cells: list[np.ndarray] = []
    empty: list[str] = []
    for rec, shp in zip(reader.iterRecords(), reader.iterShapes()):
        sta = str(rec[sta_idx])
        if sta not in wanted:
            continue
        geom = shp_transform(transformer.transform, shape(shp.__geo_interface__))
        mask = rasterize(
            [(geom, 1)],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        )
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            mask = rasterize(
                [(geom, 1)],
                out_shape=out_shape,
                transform=transform,
                fill=0,
                dtype=np.uint8,
                all_touched=True,
            )
            idx = np.flatnonzero(mask)
        if idx.size == 0:
            empty.append(sta)
            continue
        ordered.append(sta)
        cells.append(idx.astype(np.int32, copy=False))
        if len(ordered) % 400 == 0:
            logger.info("indexed %s / ~%s basins", len(ordered), len(wanted))
    if empty:
        logger.warning("%s basins missed the 4 km grid (first: %s)", len(empty), empty[:8])
    if not ordered:
        raise RuntimeError("No basin polygons matched DroughtCast StaIDs")
    logger.info(
        "indexed %s basins; median cells %s",
        len(ordered),
        int(np.median([c.size for c in cells])),
    )
    return ordered, cells


def extract_variable(
    nc_path: Path,
    nc_var: str,
    out_name: str,
    to_celsius: bool,
    ordered: list[str],
    cells: list[np.ndarray],
) -> pd.DataFrame:
    ds = xr.open_dataset(nc_path)
    try:
        data = ds[_data_var(ds, nc_var)]
        days = pd.to_datetime(ds["day"].values)
        arr = np.asarray(data.values)
    finally:
        ds.close()
    n_days, n_lat, n_lon = arr.shape
    flat = arr.reshape(n_days, n_lat * n_lon)
    n_sites = len(ordered)
    means = np.full((n_days, n_sites), np.nan, dtype=np.float32)
    for j, idx in enumerate(cells):
        block = flat[:, idx]
        with np.errstate(all="ignore"):
            means[:, j] = np.nanmean(block, axis=1)
        if (j + 1) % 800 == 0:
            logger.info("%s basin %s/%s", nc_path.name, j + 1, n_sites)
    if to_celsius:
        means = means - np.float32(273.15)
    frame = pd.DataFrame(
        {
            "StaID": np.repeat(np.array(ordered, dtype=object), n_days),
            "date": np.tile(days, n_sites),
            out_name: means.T.ravel(),
        }
    )
    logger.info("%s %s: %s rows", nc_path.name, out_name, len(frame))
    return frame


def build_basin_daily(
    years: list[int],
    *,
    shp_path: Path = BASIN_SHP,
    hist_path: Path = WEEKLY_HIST_PATH,
    out_path: Path = OUT_DAILY,
) -> Path:
    hist = pd.read_parquet(hist_path, columns=["StaID"])
    sta_ids = sorted(hist["StaID"].astype(str).unique())
    sample_path = GRIDMET_DIR / f"tmmx_{years[0]}.nc"
    if not sample_path.exists():
        download_year("tmmx", years[0])
    sample = xr.open_dataset(sample_path)
    lon = sample["lon"].values
    lat = sample["lat"].values
    sample.close()
    ordered, cells = basin_cell_index(lon, lat, sta_ids, shp_path=shp_path)

    pieces: list[pd.DataFrame] = []
    for year in years:
        year_frame: pd.DataFrame | None = None
        for stem, (nc_var, out_name, to_c) in VARIABLES.items():
            nc_path = download_year(stem, year)
            one = extract_variable(nc_path, nc_var, out_name, to_c, ordered, cells)
            year_frame = (
                one
                if year_frame is None
                else year_frame.merge(one, on=["StaID", "date"], how="outer")
            )
        pieces.append(year_frame)
    frame = pd.concat(pieces, ignore_index=True)
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
        "wrote %s rows (%s gages) %s to %s -> %s (%.1f MB)",
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
        description="Download gridMET and average onto gage basins."
    )
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Fetch NetCDFs; skip basin averaging.",
    )
    args = parser.parse_args(argv)
    years = list(range(args.start_year, args.end_year + 1))
    for year in years:
        for stem in VARIABLES:
            download_year(stem, year)
    if args.download_only:
        return
    build_basin_daily(years)


if __name__ == "__main__":
    main()
