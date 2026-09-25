"""GEFS ensemble-mean 00Z forecasts onto DroughtCast basins.

NOAA puts the ensemble mean on AWS as GRIB2. We take the 0.5-degree
file, read 2-meter TMAX/TMIN and 6-hour precipitation with rasterio,
average each drainage area, then delete the GRIB.

Leads: 24, 48, ..., 240 hours. Seven-day columns use hour 168. Ten-day
columns use hour 240. The 1-to-7-day columns are the mean of hours
24 through 168. Precipitation is the 6-hour total at that lead, which
matches the 1980–2020 GEFS rain columns.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from threading import Lock, Semaphore

import numpy as np
import pandas as pd
import rasterio

from streamflow.config import RAW_DIR
from streamflow.zonal import basin_cell_index, basin_means, north_up_lat

logger = logging.getLogger(__name__)

GEFS_DIR = RAW_DIR / "gefs"
OUT_DAILY = RAW_DIR / "gefs_basin_daily.parquet"
STATIC_PATH = RAW_DIR / "sciencebase_p132nswy" / "conus_static_inputs_gages.csv"
USER_AGENT = "usgs-streamflow-monitor (local)"
GEFS_RES = 0.5
LEADS = (24, 48, 72, 96, 120, 144, 168, 192, 216, 240)
# Cap concurrent curls so we do not open 60 AWS connections at once.
_DOWNLOAD_SLOTS = Semaphore(24)
_GDAL_LOCK = Lock()
_FLUSH_LOCK = Lock()


def _curl(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl.exe",
        "-fsSL",
        "-C",
        "-",
        "-A",
        USER_AGENT,
        "--retry",
        "5",
        "--retry-delay",
        "3",
        "--retry-all-errors",
        "-o",
        str(dest),
        url,
    ]
    result = subprocess.run(cmd, check=False, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"GEFS download failed ({result.returncode}): {url} {err}")


def gefs_urls(day: date, lead: int) -> list[str]:
    stamp = f"{day:%Y%m%d}"
    name = f"geavg.t00z.pgrb2a.0p50.f{lead:03d}"
    return [
        f"https://noaa-gefs-pds.s3.amazonaws.com/gefs.{stamp}/00/atmos/pgrb2ap5/{name}",
        f"https://noaa-gefs-pds.s3.amazonaws.com/gefs.{stamp}/00/pgrb2ap5/{name}",
        f"https://noaa-gefs-pds.s3.amazonaws.com/gefs.{stamp}/00/pgrb2a/{name}",
    ]


def download_lead(day: date, lead: int) -> Path:
    dest = GEFS_DIR / f"geavg_{day:%Y%m%d}_f{lead:03d}.grib2"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    last_err: Exception | None = None
    with _DOWNLOAD_SLOTS:
        for url in gefs_urls(day, lead):
            try:
                _curl(url, dest)
                if dest.exists() and dest.stat().st_size > 1_000_000:
                    return dest
            except RuntimeError as exc:
                last_err = exc
                if dest.exists():
                    dest.unlink()
    raise RuntimeError(f"GEFS {day} f{lead:03d}: {last_err}")


def _band_index(src: rasterio.DatasetReader, element: str) -> int:
    for i in range(1, src.count + 1):
        tags = src.tags(i)
        if tags.get("GRIB_ELEMENT") == element and "2-HTGL" in str(
            tags.get("GRIB_SHORT_NAME", "")
        ):
            return i
    if element == "APCP06":
        for i in range(1, src.count + 1):
            if src.tags(i).get("GRIB_ELEMENT") == "APCP06":
                return i
    raise RuntimeError(f"no {element} band in {src.name}")


def _lon_lat(
    src: rasterio.DatasetReader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    transform = src.transform
    cols = np.arange(src.width)
    rows = np.arange(src.height)
    lon = transform.c + (cols + 0.5) * transform.a
    lat = transform.f + (rows + 0.5) * transform.e
    lon = lon.astype(np.float64)
    if float(np.nanmin(lon)) >= 0:
        lon = np.where(lon > 180.0, lon - 360.0, lon)
        order = np.argsort(lon)
        lon = lon[order]
        return lon, lat.astype(np.float64), order
    return lon, lat.astype(np.float64), None


def _read_fields(path: Path) -> dict[str, np.ndarray]:
    with _GDAL_LOCK:
        with rasterio.open(path) as src:
            tmax = src.read(_band_index(src, "TMAX"))
            tmin = src.read(_band_index(src, "TMIN"))
            apcp = src.read(_band_index(src, "APCP06"))
            lon, lat, lon_order = _lon_lat(src)
    if lon_order is not None:
        tmax = tmax[:, lon_order]
        tmin = tmin[:, lon_order]
        apcp = apcp[:, lon_order]
    return {"tmax": tmax, "tmin": tmin, "apcp": apcp, "lon": lon, "lat": lat}


def _sta_ids() -> list[str]:
    if STATIC_PATH.exists():
        static = pd.read_csv(STATIC_PATH, usecols=["StaID"])
        return sorted(static["StaID"].astype(str).str.zfill(8).unique())
    hist = pd.read_parquet(RAW_DIR.parent / "processed" / "weekly_features.parquet", columns=["StaID"])
    return sorted(hist["StaID"].astype(str).unique())


def _day_rows(
    day: date,
    ordered: list[str],
    cells: list[np.ndarray],
    flipped: bool,
) -> list[dict]:
    paths: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=len(LEADS)) as pool:
        futs = {pool.submit(download_lead, day, lead): lead for lead in LEADS}
        for fut in as_completed(futs):
            paths[futs[fut]] = fut.result()

    by_lead: dict[int, dict[str, np.ndarray]] = {}
    try:
        for lead in LEADS:
            raw = _read_fields(paths[lead])
            tmax = raw["tmax"][::-1] if flipped else raw["tmax"]
            tmin = raw["tmin"][::-1] if flipped else raw["tmin"]
            apcp = raw["apcp"][::-1] if flipped else raw["apcp"]
            by_lead[lead] = {
                "tmax": basin_means(tmax, ordered, cells),
                "tmin": basin_means(tmin, ordered, cells),
                "tp": basin_means(apcp, ordered, cells),
            }
    finally:
        for path in paths.values():
            path.unlink(missing_ok=True)

    tmax_1to7 = np.nanmean(
        np.stack([by_lead[h]["tmax"] for h in LEADS if h <= 168]), axis=0
    )
    tmin_1to7 = np.nanmean(
        np.stack([by_lead[h]["tmin"] for h in LEADS if h <= 168]), axis=0
    )
    tp_1to7 = np.nanmean(
        np.stack([by_lead[h]["tp"] for h in LEADS if h <= 168]), axis=0
    )
    tmax_1to10 = np.nanmean(np.stack([by_lead[h]["tmax"] for h in LEADS]), axis=0)
    tmin_1to10 = np.nanmean(np.stack([by_lead[h]["tmin"] for h in LEADS]), axis=0)
    tp_1to10 = np.nanmean(np.stack([by_lead[h]["tp"] for h in LEADS]), axis=0)
    return [
        {
            "StaID": sta,
            "date": pd.Timestamp(day),
            "tmax7day_gefs_mean": by_lead[168]["tmax"][i],
            "tmin7day_gefs_mean": by_lead[168]["tmin"][i],
            "tp7day_gefs_mean": by_lead[168]["tp"][i],
            "tmax10day_gefs_mean": by_lead[240]["tmax"][i],
            "tmin10day_gefs_mean": by_lead[240]["tmin"][i],
            "tp10day_gefs_mean": by_lead[240]["tp"][i],
            "tmax1to7day_gefs_mean": tmax_1to7[i],
            "tmin1to7day_gefs_mean": tmin_1to7[i],
            "tp1to7day_gefs_mean": tp_1to7[i],
            "tmax1to10day_gefs_mean": tmax_1to10[i],
            "tmin1to10day_gefs_mean": tmin_1to10[i],
            "tp1to10day_gefs_mean": tp_1to10[i],
        }
        for i, sta in enumerate(ordered)
    ]


def build_basin_daily(
    start: date,
    end: date,
    *,
    out_path: Path = OUT_DAILY,
    workers: int = 6,
) -> Path:
    sta_ids = _sta_ids()
    sample_day = start
    sample = download_lead(sample_day, 168)
    fields = _read_fields(sample)
    lat, flipped = north_up_lat(fields["lat"])
    ordered, cells = basin_cell_index(
        fields["lon"], lat, res=GEFS_RES, sta_ids=sta_ids
    )
    sample.unlink(missing_ok=True)

    done: set[pd.Timestamp] = set()
    if out_path.exists():
        existing = pd.read_parquet(out_path, columns=["date"])
        done = set(pd.to_datetime(existing["date"]).unique())
        logger.info("resume GEFS: %s days already on disk", len(done))

    todo: list[date] = []
    day = start
    while day <= end:
        if pd.Timestamp(day) not in done:
            todo.append(day)
        day += timedelta(days=1)
    logger.info("GEFS %s days to pull with %s day-workers", len(todo), workers)

    rows: list[dict] = []
    finished = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(_day_rows, day, ordered, cells, flipped): day for day in todo
        }
        for fut in as_completed(futs):
            day = futs[fut]
            try:
                rows.extend(fut.result())
                finished += 1
                logger.info("GEFS %s (%s / %s days)", day, finished, len(todo))
            except Exception as exc:  # noqa: BLE001
                logger.warning("skip GEFS %s: %s", day, exc)
            if len(rows) >= 3229 * 8:
                with _FLUSH_LOCK:
                    _flush_gefs(rows, out_path)
                    rows = []

    if rows:
        _flush_gefs(rows, out_path)
    if not out_path.exists():
        raise RuntimeError("No GEFS days extracted")
    frame = pd.read_parquet(out_path)
    logger.info(
        "GEFS done: %s rows (%s stations) %s to %s -> %s (%.1f MB)",
        len(frame),
        frame["StaID"].nunique(),
        frame["date"].min().date(),
        frame["date"].max().date(),
        out_path,
        out_path.stat().st_size / 1e6,
    )
    return out_path


def _flush_gefs(rows: list[dict], out_path: Path) -> None:
    frame = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        old = pd.read_parquet(out_path)
        frame = (
            pd.concat([old, frame], ignore_index=True)
            .sort_values(["StaID", "date"])
            .drop_duplicates(["StaID", "date"], keep="last")
        )
    frame.to_parquet(out_path, index=False)
    logger.info("flushed GEFS -> %s (%s rows)", out_path, len(frame))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Download GEFS ensemble-mean forecasts onto drainage areas."
    )
    parser.add_argument("--start", default="2024-09-16")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="How many calendar days to download at once (default 6).",
    )
    args = parser.parse_args(argv)
    build_basin_daily(
        date.fromisoformat(args.start),
        date.fromisoformat(args.end),
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
