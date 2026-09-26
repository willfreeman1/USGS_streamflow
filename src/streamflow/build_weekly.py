"""Gage × week table.

Historical path stacks ScienceBase feathers. Live path turns daily mean
discharge into the same Monday week and scores it against 1980–2020.
USGS published weekly_weibull_jd but left it empty; we fill it.
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as ft
import pyarrow.parquet as pq

from streamflow.config import (
    LIVE_WEEKLY_START,
    DAILY_FLOW_PATH,
    OPENET_POINT_MONTHLY,
    PROCESSED_DIR,
    RAW_DIR,
    RESERVOIR_BASIN_DAILY,
    SSEBOP_BASIN_MONTHLY,
    WEEKLY_HIST_PATH,
    WEEKLY_LIVE_PATH,
)
from streamflow.ingest_gridmet import OUT_DAILY as GRIDMET_DAILY
from streamflow.ingest_nldas import OUT_DAILY as NLDAS_DAILY
from streamflow.ingest_openet import attach_openet_to_gages
from streamflow.ingest_smap import OUT_DAILY as SMAP_DAILY
from streamflow.matched import (
    add_flow_scores,
    add_rolls,
    attach_climate,
    attach_gefs,
    attach_nmme,
    attach_static,
    attach_swe,
    attach_water_use_mean,
    write_hist_matched,
)

logger = logging.getLogger(__name__)

SB_DIR = RAW_DIR / "sciencebase_p1x5vh96"
ZIP_PATH = SB_DIR / "weekly_model_inputs.zip"
ADDED_DIR = SB_DIR / "added"
OUT_PATH = WEEKLY_HIST_PATH

MIN_DAYS_IN_WEEK = 4
MIN_CLIM_N = 10


def build_weekly_historical(
    *,
    zip_path: Path = ZIP_PATH,
    added_dir: Path = ADDED_DIR,
    out_path: Path = OUT_PATH,
    delete_zip: bool = False,
) -> Path:
    added = sorted(added_dir.rglob("*.feather"))
    if not added:
        raise RuntimeError(f"No added feathers under {added_dir}")
    if not zip_path.exists():
        raise RuntimeError(f"Missing {zip_path}; run ingest-historical")

    schema = _read_feather(added[0]).schema
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".parquet.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    n_files = 0
    n_rows = 0
    sites: set[str] = set()
    writer: pq.ParquetWriter | None = None
    try:
        for path in added:
            table = _align(_read_feather(path), schema)
            writer = _write(writer, tmp_path, table)
            n_files += 1
            n_rows += table.num_rows
            sites.update(_site_ids(table))
            logger.info("added %s (%s rows)", path.name, table.num_rows)

        with zipfile.ZipFile(zip_path) as archive:
            names = [n for n in archive.namelist() if n.endswith(".feather")]
            logger.info("zip feathers %s", len(names))
            for i, name in enumerate(names, start=1):
                table = _align(_read_feather(BytesIO(archive.read(name))), schema)
                writer = _write(writer, tmp_path, table)
                n_files += 1
                n_rows += table.num_rows
                sites.update(_site_ids(table))
                if i % 100 == 0 or i == len(names):
                    logger.info(
                        "zip %s/%s files rows=%s sites=%s",
                        i,
                        len(names),
                        n_rows,
                        len(sites),
                    )
    finally:
        if writer is not None:
            writer.close()

    if n_files != 3229 or len(sites) != 3229:
        raise RuntimeError(
            f"expected 3229 gages, wrote {n_files} files / {len(sites)} sites"
        )
    tmp_path.replace(out_path)
    logger.info(
        "wrote %s rows (%s sites, %s files) to %s (%.1f GB)",
        n_rows,
        len(sites),
        n_files,
        out_path,
        out_path.stat().st_size / 1e9,
    )
    if delete_zip:
        zip_path.unlink()
        logger.info("deleted %s", zip_path)
    return out_path


def build_weekly_from_daily(
    *,
    daily_path: Path = DAILY_FLOW_PATH,
    hist_path: Path = WEEKLY_HIST_PATH,
    out_path: Path = WEEKLY_LIVE_PATH,
) -> Path:
    """Monday weeks from live daily flow, scored on 1980–2020 climatology."""
    if not daily_path.exists():
        raise RuntimeError(f"Missing {daily_path}; run ingest-flow")
    if not hist_path.exists():
        raise RuntimeError(f"Missing {hist_path}; run build-weekly --historical")

    logger.info("loading historical Flow_cfs climatology from %s", hist_path)
    hist = pq.read_table(
        hist_path, columns=["StaID", "Date", "week_num", "Flow_cfs"]
    ).to_pandas()
    hist["StaID"] = hist["StaID"].astype("string")
    hist["week_num"] = hist["week_num"].astype("Int64")
    by_week, by_site = _climatology(hist)
    logger.info(
        "climatology: %s gages, %s week bins",
        len(by_site),
        len(by_week),
    )
    _log_hist_check(hist, by_week)

    daily = pd.read_parquet(
        daily_path,
        columns=["monitoring_location_id", "date", "discharge_cfs"],
    )
    daily["StaID"] = daily["monitoring_location_id"].map(_to_sta_id)
    daily["date"] = pd.to_datetime(daily["date"])
    weekly = _daily_to_monday_weeks(daily)
    overlap = weekly["StaID"].isin(by_site)
    logger.info(
        "live weeks %s rows / %s gages; %s rows on DroughtCast gages",
        len(weekly),
        weekly["StaID"].nunique(),
        int(overlap.sum()),
    )
    scored = weekly.loc[overlap].copy()
    scored["weekly_weibull_jd"] = [
        _weibull_new(by_week.get((sta, int(week))), flow)
        for sta, week, flow in zip(
            scored["StaID"], scored["week_num"], scored["Flow_cfs"]
        )
    ]
    scored["weekly_weibull_site"] = [
        _weibull_new(by_site.get(sta), flow)
        for sta, flow in zip(scored["StaID"], scored["Flow_cfs"])
    ]
    scored = _stamp_calendar(scored)
    scored = attach_static(scored)
    scored = _attach_gridmet(scored)
    scored = _attach_nldas(scored)
    scored = _attach_smap(scored)
    scored = _attach_reservoirs(scored)
    scored = _attach_ssebop(scored)
    scored = _attach_openet(scored)
    scored = attach_swe(scored)
    scored = attach_gefs(scored)
    scored = attach_nmme(scored)
    scored = attach_climate(scored)
    scored = attach_water_use_mean(scored)
    scored = add_flow_scores(scored, daily)
    scored = add_rolls(scored)
    start = pd.Timestamp(LIVE_WEEKLY_START)
    n_warm = int((scored["Date"] < start).sum())
    scored = scored.loc[scored["Date"] >= start].copy()
    logger.info(
        "dropped %s warmup weeks before %s; kept %s from the live start",
        n_warm,
        LIVE_WEEKLY_START,
        len(scored),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(out_path, index=False)
    logger.info(
        "wrote %s rows (%s gages) %s to %s to %s (%.1f MB)",
        len(scored),
        scored["StaID"].nunique(),
        scored["Date"].min().date(),
        scored["Date"].max().date(),
        out_path,
        out_path.stat().st_size / 1e6,
    )
    return out_path


def _climatology(
    hist: pd.DataFrame,
) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, np.ndarray]]:
    clean = hist.dropna(subset=["StaID", "week_num", "Flow_cfs"])
    by_week: dict[tuple[str, int], np.ndarray] = {}
    for (sta, week), grp in clean.groupby(["StaID", "week_num"], sort=False):
        by_week[(str(sta), int(week))] = grp["Flow_cfs"].to_numpy(dtype=float)
    by_site = {
        str(sta): grp["Flow_cfs"].to_numpy(dtype=float)
        for sta, grp in clean.groupby("StaID", sort=False)
    }
    return by_week, by_site


def _daily_to_monday_weeks(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.dropna(subset=["discharge_cfs"]).copy()
    # A few gages have two USGS daily series; one value per site-day.
    frame = (
        frame.groupby(["StaID", "date"], sort=False)["discharge_cfs"]
        .mean()
        .reset_index()
    )
    frame["week_dt"] = frame["date"] - pd.to_timedelta(
        frame["date"].dt.dayofweek, unit="D"
    )
    last = frame["date"].max()
    grouped = (
        frame.groupby(["StaID", "week_dt"], sort=False)
        .agg(
            Flow_cfs=("discharge_cfs", "mean"),
            n_days=("discharge_cfs", "count"),
        )
        .reset_index()
    )
    complete_end = grouped["week_dt"] + pd.Timedelta(days=6)
    keep = (grouped["n_days"] >= MIN_DAYS_IN_WEEK) & (complete_end <= last)
    weekly = grouped.loc[keep].copy()
    weekly["week_num"] = weekly["week_dt"].dt.isocalendar().week.astype(int)
    weekly["Date"] = weekly["week_dt"]
    return weekly


def _attach_gridmet(
    weekly: pd.DataFrame, daily_path: Path = GRIDMET_DAILY
) -> pd.DataFrame:
    """Monday-week weather. Names match USGS: weekly mean of daily basin values.

    The dictionary says pr/pet are 'total weekly' but the 1980–2020 numbers
    are daily-scale (national pr_mean mean 2.85 mm, pet_mean 3.23 mm).
    """
    if not daily_path.exists():
        logger.info("no %s; skip gridMET join", daily_path)
        return weekly
    daily = pd.read_parquet(daily_path)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["week_dt"] = daily["date"] - pd.to_timedelta(
        daily["date"].dt.dayofweek, unit="D"
    )
    wx = (
        daily.groupby(["StaID", "week_dt"], sort=False)
        .agg(
            tmmx_mean=("tmmx", "mean"),
            tmmn_mean=("tmmn", "mean"),
            pr_mean=("pr", "mean"),
            pet_mean=("pet", "mean"),
        )
        .reset_index()
    )
    out = weekly.merge(wx, on=["StaID", "week_dt"], how="left")
    logger.info(
        "gridMET join: %s / %s weeks have tmmx_mean",
        int(out["tmmx_mean"].notna().sum()),
        len(out),
    )
    return out


def _attach_nldas(
    weekly: pd.DataFrame, daily_path: Path = NLDAS_DAILY
) -> pd.DataFrame:
    """Monday-week mean of daily NLDAS soil moisture (kg per square meter)."""
    if not daily_path.exists():
        logger.info("no %s; skip NLDAS join", daily_path)
        return weekly
    daily = pd.read_parquet(daily_path)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["week_dt"] = daily["date"] - pd.to_timedelta(
        daily["date"].dt.dayofweek, unit="D"
    )
    sm = (
        daily.groupby(["StaID", "week_dt"], sort=False)
        .agg(
            soilm_0_10cm_mean=("soilm_0_10cm", "mean"),
            soilm_10_40cm_mean=("soilm_10_40cm", "mean"),
            soilm_40_100cm_mean=("soilm_40_100cm", "mean"),
        )
        .reset_index()
    )
    out = weekly.merge(sm, on=["StaID", "week_dt"], how="left")
    logger.info(
        "NLDAS join: %s / %s weeks have soilm_0_10cm_mean",
        int(out["soilm_0_10cm_mean"].notna().sum()),
        len(out),
    )
    return out


def _attach_smap(
    weekly: pd.DataFrame, daily_path: Path = SMAP_DAILY
) -> pd.DataFrame:
    """Monday-week mean of SMAP volumetric soil moisture (cm3/cm3)."""
    if not daily_path.exists():
        logger.info("no %s; skip SMAP join", daily_path)
        return weekly
    daily = pd.read_parquet(daily_path)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["week_dt"] = daily["date"] - pd.to_timedelta(
        daily["date"].dt.dayofweek, unit="D"
    )
    sm = (
        daily.groupby(["StaID", "week_dt"], sort=False)
        .agg(
            smap_sm_mean=("smap_sm", "mean"),
            smap_n_days=("smap_sm", "count"),
        )
        .reset_index()
    )
    out = weekly.merge(sm, on=["StaID", "week_dt"], how="left")
    logger.info(
        "SMAP join: %s / %s weeks have smap_sm_mean",
        int(out["smap_sm_mean"].notna().sum()),
        len(out),
    )
    return out


def _attach_reservoirs(
    weekly: pd.DataFrame, daily_path: Path = RESERVOIR_BASIN_DAILY
) -> pd.DataFrame:
    """Monday-week mean of daily watershed storage (acre-feet)."""
    if not daily_path.exists():
        logger.info("no %s; skip reservoir join", daily_path)
        return weekly
    daily = pd.read_parquet(daily_path)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["week_dt"] = daily["date"] - pd.to_timedelta(
        daily["date"].dt.dayofweek, unit="D"
    )
    res = (
        daily.groupby(["StaID", "week_dt"], sort=False)
        .agg(
            reservoirs_storage_total=("reservoirs_storage_total", "mean"),
            reservoirs_n_total=("reservoirs_n_total", "mean"),
        )
        .reset_index()
    )
    out = weekly.merge(res, on=["StaID", "week_dt"], how="left")
    logger.info(
        "reservoir join: %s / %s weeks have storage",
        int(out["reservoirs_storage_total"].notna().sum()),
        len(out),
    )
    return out


def _attach_ssebop(
    weekly: pd.DataFrame, monthly_path: Path = SSEBOP_BASIN_MONTHLY
) -> pd.DataFrame:
    """Monthly SSEBop AET on every week in that month. aet_mean is mm/day."""
    if not monthly_path.exists():
        logger.info("no %s; skip SSEBop join", monthly_path)
        return weekly
    monthly = pd.read_parquet(monthly_path)
    monthly["date"] = pd.to_datetime(monthly["date"])
    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month
    keep = monthly[["StaID", "year", "month", "aet_mean", "aet_mm_month"]]
    out = weekly.merge(keep, on=["StaID", "year", "month"], how="left")
    logger.info(
        "SSEBop join: %s / %s weeks have aet_mean",
        int(out["aet_mean"].notna().sum()),
        len(out),
    )
    return out


def _attach_openet(
    weekly: pd.DataFrame, point_path: Path = OPENET_POINT_MONTHLY
) -> pd.DataFrame:
    """Nearest OpenET Ensemble point, monthly millimeters."""
    if not point_path.exists():
        logger.info("no %s; skip OpenET join", point_path)
        return weekly
    monthly = pd.read_parquet(point_path)
    monthly["date"] = pd.to_datetime(monthly["date"])
    sta_ids = sorted(weekly["StaID"].astype(str).unique())
    mapped = attach_openet_to_gages(monthly, sta_ids)
    mapped["year"] = mapped["date"].dt.year
    mapped["month"] = mapped["date"].dt.month
    keep = mapped[["StaID", "year", "month", "openet_et_mm"]].drop_duplicates(
        ["StaID", "year", "month"]
    )
    out = weekly.merge(keep, on=["StaID", "year", "month"], how="left")
    logger.info(
        "OpenET join: %s / %s weeks have openet_et_mm",
        int(out["openet_et_mm"].notna().sum()),
        len(out),
    )
    return out


def _stamp_calendar(weekly: pd.DataFrame) -> pd.DataFrame:
    out = weekly.copy()
    out["year"] = out["Date"].dt.year
    out["month"] = out["Date"].dt.month
    out["jd"] = out["Date"].dt.dayofyear
    out["wy"] = np.where(out["month"] >= 10, out["year"] + 1, out["year"])
    out["cy"] = np.where(out["month"] >= 4, out["year"] + 1, out["year"])
    cols = [
        "Date",
        "week_dt",
        "week_num",
        "year",
        "month",
        "wy",
        "cy",
        "jd",
        "StaID",
        "Flow_cfs",
        "n_days",
        "weekly_weibull_jd",
        "weekly_weibull_site",
    ]
    return out[cols].sort_values(["StaID", "Date"]).reset_index(drop=True)


def _weibull_new(sample: np.ndarray | None, value: float) -> float:
    """Out-of-sample Weibull: (1 + #strictly less + 0.5*ties) / (n+1) * 100."""
    if sample is None or not np.isfinite(value):
        return np.nan
    sample = sample[np.isfinite(sample)]
    n = int(sample.size)
    if n < MIN_CLIM_N:
        return np.nan
    less = float(np.sum(sample < value))
    ties = float(np.sum(sample == value))
    rank = 1.0 + less + 0.5 * ties
    return rank / (n + 1.0) * 100.0


def _log_hist_check(
    hist: pd.DataFrame, by_week: dict[tuple[str, int], np.ndarray]
) -> None:
    """In-sample check: our week-of-year rank vs USGS weibull_jd (related, not equal)."""
    try:
        probe = pq.read_table(
            WEEKLY_HIST_PATH,
            columns=["StaID", "week_num", "Flow_cfs", "weibull_jd"],
        ).to_pandas()
    except Exception as exc:  # noqa: BLE001
        logger.warning("skip hist check: %s", exc)
        return
    probe = probe.dropna(subset=["Flow_cfs", "weibull_jd", "week_num"])
    sites = probe["StaID"].drop_duplicates().head(5)
    maes: list[float] = []
    for sta in sites:
        sub = probe.loc[probe["StaID"] == sta].copy()

        def _in_sample(row: pd.Series) -> float:
            sample = by_week.get((str(row["StaID"]), int(row["week_num"])))
            return _weibull_new(sample, float(row["Flow_cfs"]))

        sub["ours"] = sub.apply(_in_sample, axis=1)
        mae = (sub["ours"] - sub["weibull_jd"]).abs().mean()
        maes.append(float(mae))
    if maes:
        logger.info(
            "in-sample |weekly_weibull_jd - USGS weibull_jd| mean MAE on 5 gages: %.2f",
            float(np.mean(maes)),
        )


def _to_sta_id(raw: object) -> str:
    text = str(raw)
    if text.startswith("USGS-"):
        text = text[5:]
    if text.isdigit() and len(text) < 8:
        return text.zfill(8)
    return text


def _read_feather(source: Path | BytesIO) -> pa.Table:
    return ft.read_table(source)


def _align(table: pa.Table, schema: pa.Schema) -> pa.Table:
    for field in schema:
        if field.name not in table.schema.names:
            table = table.append_column(
                field.name, pa.nulls(table.num_rows, type=field.type)
            )
    return table.select(list(schema.names)).cast(schema, safe=False)


def _write(
    writer: pq.ParquetWriter | None, path: Path, table: pa.Table
) -> pq.ParquetWriter:
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def _site_ids(table: pa.Table) -> list[str]:
    col = table.column("StaID")
    return [str(x) for x in col.unique().to_pylist() if x is not None]


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Build the weekly gage×week table."
    )
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Restack ScienceBase feathers into weekly_features.parquet.",
    )
    parser.add_argument(
        "--delete-zip",
        action="store_true",
        help="With --historical, delete weekly_model_inputs.zip after success.",
    )
    parser.add_argument(
        "--from-daily",
        action="store_true",
        help="Build weekly_live.parquet from ingested daily discharge.",
    )
    parser.add_argument(
        "--matched-history",
        action="store_true",
        help="Write weekly_hist_matched.parquet (USGS columns we can rebuild).",
    )
    args = parser.parse_args(argv)
    if not args.historical and not args.from_daily and not args.matched_history:
        args.from_daily = True
    if args.historical:
        build_weekly_historical(delete_zip=args.delete_zip)
    if args.from_daily:
        build_weekly_from_daily()
    if args.matched_history:
        write_hist_matched()


if __name__ == "__main__":
    main()
