"""SMAP L3 36 km soil moisture onto DroughtCast basins.

This is a satellite retrieval (cubic centimeters of water per cubic
centimeter of soil), not the NLDAS layer water. NASA’s 1980–2020 table
does not have it. We still want it live: it is an independent check on
whether the land is actually wet.

The grid is EASE-2, not a regular lat/lon raster, so each basin uses the
cell centers that fall inside it. A basin smaller than 36 km gets the
nearest cell. Morning (AM) is preferred; afternoon (PM) fills gaps.
Only retrievals NASA marks as recommended quality (flag 0 or 8) are kept.
Granules are deleted after extract so we do not store ~24 GB of HDF5.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
import shapefile
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform as shp_transform
from shapely.strtree import STRtree
from shapely.validation import make_valid

from streamflow.config import RAW_DIR, WEEKLY_HIST_PATH
from streamflow.earthdata import download as edl_download
from streamflow.earthdata import session as edl_session
from streamflow.zonal import BASIN_SHP

logger = logging.getLogger(__name__)

SMAP_DIR = RAW_DIR / "smap"
OUT_DAILY = RAW_DIR / "smap_basin_daily.parquet"
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
FILL = -9999.0


def cmr_h5_links(start: date, end: date) -> list[tuple[date, str]]:
    """One SPL3SMP v009 granule URL per day in [start, end]."""
    sess = requests.Session()
    sess.headers["User-Agent"] = "usgs-streamflow-monitor (local)"
    params = {
        "short_name": "SPL3SMP",
        "version": "009",
        "temporal": f"{start.isoformat()}T00:00:00Z,{end.isoformat()}T23:59:59Z",
        "page_size": 2000,
        "sort_key": "start_date",
    }
    headers: dict[str, str] = {}
    out: list[tuple[date, str]] = []
    while True:
        r = sess.get(CMR, params=params, headers=headers, timeout=90)
        r.raise_for_status()
        entries = r.json().get("feed", {}).get("entry", [])
        if not entries:
            break
        for entry in entries:
            href = ""
            for link in entry.get("links", []):
                cand = link.get("href", "")
                if cand.endswith(".h5") and "protected" in cand:
                    href = cand
                    break
            if not href:
                continue
            day = date.fromisoformat(entry["time_start"][:10])
            out.append((day, href))
        token = r.headers.get("CMR-Search-After")
        if not token:
            break
        headers["CMR-Search-After"] = token
    # last URL wins if a day has more than one revision
    by_day = {day: url for day, url in out}
    days = sorted(by_day)
    logger.info("CMR listed %s SMAP days %s to %s", len(days), days[0], days[-1])
    return [(day, by_day[day]) for day in days]


def _recommended(sm: np.ndarray, qual: np.ndarray) -> np.ndarray:
    sm = np.asarray(sm, dtype=np.float32)
    qual = np.asarray(qual)
    good = (sm > 0.0) & (sm < 1.0) & ((qual == 0) | (qual == 8))
    out = sm.copy()
    out[~good] = np.nan
    return out


def read_daily_grid(path: Path) -> np.ndarray:
    """AM preferred, PM fills holes. Shape matches the EASE 406×964 grid."""
    with h5py.File(path, "r") as handle:
        am = _recommended(
            handle["Soil_Moisture_Retrieval_Data_AM/soil_moisture"][:],
            handle["Soil_Moisture_Retrieval_Data_AM/retrieval_qual_flag"][:],
        )
        pm = _recommended(
            handle["Soil_Moisture_Retrieval_Data_PM/soil_moisture_pm"][:],
            handle["Soil_Moisture_Retrieval_Data_PM/retrieval_qual_flag_pm"][:],
        )
    return np.where(np.isfinite(am), am, pm).astype(np.float32)


def read_lonlat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        lon = np.asarray(
            handle["Soil_Moisture_Retrieval_Data_AM/longitude"][:], dtype=np.float64
        )
        lat = np.asarray(
            handle["Soil_Moisture_Retrieval_Data_AM/latitude"][:], dtype=np.float64
        )
    return lon, lat


def smap_cell_index(
    lon: np.ndarray,
    lat: np.ndarray,
    sta_ids: list[str],
    shp_path: Path = BASIN_SHP,
) -> tuple[list[str], list[np.ndarray]]:
    lon_f = lon.ravel()
    lat_f = lat.ravel()
    valid = (
        np.isfinite(lon_f)
        & np.isfinite(lat_f)
        & (lon_f > FILL + 1)
        & (np.abs(lon_f) <= 180)
        & (np.abs(lat_f) <= 90)
    )
    conus = valid & (lon_f >= -125) & (lon_f <= -66) & (lat_f >= 24.5) & (lat_f <= 53)
    idx = np.flatnonzero(conus)
    lon_c = lon_f[idx]
    lat_c = lat_f[idx]
    tree = STRtree([Point(float(x), float(y)) for x, y in zip(lon_c, lat_c)])
    wanted = set(sta_ids)
    transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    reader = shapefile.Reader(str(shp_path.with_suffix("")))
    field_names = [field[0] for field in reader.fields[1:]]
    sta_col = field_names.index("StaID")

    ordered: list[str] = []
    cells: list[np.ndarray] = []
    n_nn = 0
    for rec, shp in zip(reader.iterRecords(), reader.iterShapes()):
        sta = str(rec[sta_col])
        if sta not in wanted:
            continue
        geom = make_valid(
            shp_transform(transformer.transform, shape(shp.__geo_interface__))
        )
        hits = tree.query(geom, predicate="intersects")
        if len(hits) == 0:
            centroid = geom.centroid
            j = int(np.argmin((lon_c - centroid.x) ** 2 + (lat_c - centroid.y) ** 2))
            cell_ids = np.array([idx[j]], dtype=np.int32)
            n_nn += 1
        else:
            cell_ids = idx[np.asarray(hits, dtype=int)].astype(np.int32)
        ordered.append(sta)
        cells.append(cell_ids)
        if len(ordered) % 400 == 0:
            logger.info("indexed %s / ~%s basins", len(ordered), len(wanted))
    logger.info(
        "indexed %s basins; median cells %s; nearest-cell fallback %s",
        len(ordered),
        int(np.median([c.size for c in cells])),
        n_nn,
    )
    return ordered, cells


def extract_day(
    path: Path,
    day: date,
    ordered: list[str],
    cells: list[np.ndarray],
) -> pd.DataFrame:
    grid = read_daily_grid(path).ravel()
    means = np.full(len(ordered), np.nan, dtype=np.float32)
    n_ok = np.zeros(len(ordered), dtype=np.int16)
    for i, idx in enumerate(cells):
        block = grid[idx]
        good = block[np.isfinite(block)]
        if good.size:
            means[i] = float(good.mean())
            n_ok[i] = good.size
    return pd.DataFrame(
        {
            "StaID": ordered,
            "date": pd.Timestamp(day),
            "smap_sm": means,
            "smap_n_cells": n_ok,
        }
    )


def _process(
    day: date, url: str, ordered: list[str], cells: list[np.ndarray], keep: bool
) -> pd.DataFrame | None:
    dest = SMAP_DIR / f"SPL3SMP_{day:%Y%m%d}.h5"
    try:
        edl_download(url, dest, sess=edl_session())
        frame = extract_day(dest, day, ordered, cells)
    except Exception as exc:  # noqa: BLE001
        logger.warning("skip %s: %s", day, exc)
        return None
    finally:
        if dest.exists() and not keep:
            dest.unlink()
    return frame


def build_basin_daily(
    start: date,
    end: date,
    *,
    keep_granules: bool = False,
    out_path: Path = OUT_DAILY,
) -> Path:
    hist = pd.read_parquet(WEEKLY_HIST_PATH, columns=["StaID"])
    sta_ids = sorted(hist["StaID"].astype(str).unique())
    links = cmr_h5_links(start, end)
    if not links:
        raise RuntimeError("CMR returned no SPL3SMP granules")

    seed = SMAP_DIR / "grid_seed.h5"
    if not seed.exists():
        existing = next(SMAP_DIR.glob("*.h5"), None) if SMAP_DIR.exists() else None
        if existing is not None:
            seed = existing
        else:
            edl_download(links[0][1], seed, sess=edl_session())
    lon, lat = read_lonlat(seed)
    ordered, cells = smap_cell_index(lon, lat, sta_ids)

    frames: list[pd.DataFrame] = []
    done = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = [
            pool.submit(_process, day, url, ordered, cells, keep_granules)
            for day, url in links
        ]
        for fut in as_completed(futs):
            frame = fut.result()
            done += 1
            if frame is not None:
                frames.append(frame)
            if done % 50 == 0 or done == 1 or done == len(links):
                logger.info("extracted %s/%s SMAP days", done, len(links))
    if not frames:
        raise RuntimeError("No SMAP days extracted")
    table = pd.concat(frames, ignore_index=True)
    if out_path.exists():
        old = pd.read_parquet(out_path)
        table = pd.concat([old, table], ignore_index=True)
    table = (
        table.sort_values(["StaID", "date"])
        .drop_duplicates(["StaID", "date"], keep="last")
        .reset_index(drop=True)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False)
    logger.info(
        "wrote %s rows (%s gages) %s to %s -> %s (%.1f MB); %s days with any value",
        len(table),
        table["StaID"].nunique(),
        pd.to_datetime(table["date"]).min().date(),
        pd.to_datetime(table["date"]).max().date(),
        out_path,
        out_path.stat().st_size / 1e6,
        int(table.groupby("date")["smap_sm"].apply(lambda s: s.notna().any()).sum()),
    )
    return out_path


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Download SMAP L3 soil moisture and average onto gage basins."
    )
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-09-21")
    parser.add_argument(
        "--keep-granules",
        action="store_true",
        help="Keep the 33 MB HDF5 files (default: delete after extract).",
    )
    args = parser.parse_args(argv)
    build_basin_daily(
        date.fromisoformat(args.start),
        date.fromisoformat(args.end),
        keep_granules=args.keep_granules,
    )


if __name__ == "__main__":
    main()
