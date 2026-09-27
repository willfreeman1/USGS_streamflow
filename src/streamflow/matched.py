"""Columns we can keep on both sides of 2020, and how to attach them live."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from streamflow.config import (
    PROCESSED_DIR,
    RAW_DIR,
    WEEKLY_HIST_PATH,
)
from streamflow.storage import atomic_parquet

logger = logging.getLogger(__name__)

MIN_CLIM_N = 10

STATIC_PATH = RAW_DIR / "sciencebase_p132nswy" / "conus_static_inputs_gages.csv"
CLIMATE_PATH = RAW_DIR / "climate_monthly.parquet"
SWE_DAILY = RAW_DIR / "swe_basin_daily.parquet"
NMME_MONTHLY = RAW_DIR / "nmme_basin_monthly.parquet"
GEFS_DAILY = RAW_DIR / "gefs_basin_daily.parquet"
WEEKLY_HIST_MATCHED = PROCESSED_DIR / "weekly_hist_matched.parquet"

STATIC_COLS = [
    "S-SqKm",
    "S-DI_EROM",
    "S-DI_PMC",
    "S-TOT_AET",
    "S-TOT_ARTIFICIAL",
    "S-TOT_BFI",
    "S-TOT_CANALDITCH",
    "S-TOT_CLAYAVE",
    "S-TOT_CWD",
    "S-TOT_ELEV_MAX",
    "S-TOT_ELEV_MIN",
    "S-TOT_EWT",
    "S-TOT_FRESHWATER_WD",
    "S-TOT_FSTFZ6190",
    "S-TOT_HGD",
    "S-TOT_LSTFZ6190",
    "S-TOT_MAXP6190",
    "S-TOT_MAXWD6190",
    "S-TOT_MIRAD_2012",
    "S-TOT_NLCD19_DEVELOPED",
    "S-TOT_NLCD19_FOREST",
    "S-TOT_NLCD19_WETLAND",
    "S-TOT_PERMAVE",
    "S-TOT_PIPELINE",
    "S-TOT_PRSNOW",
    "S-TOT_RECHG",
    "S-TOT_RH",
    "S-TOT_ROCKDEP",
    "S-TOT_SANDAVE",
    "S-TOT_SILTAVE",
    "S-TOT_STREAM_LENGTH",
    "S-TOT_STREAM_SLOPE",
    "S-TOT_TOTAL_ROAD_DENS",
    "S-TOT_TWI",
    "S-TOT_WB5100_ANN",
    "S-TOT_WDANN",
    "S-TOT_WTDEP",
]

CFS_TO_MM = 2.446575

HIST_KEEP = [
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
    "Flow_mm",
    "mean_value_7d",
    "weibull_jd_30d_wndw_7d",
    "weibull_site_7d",
    "weibull_jd_30d_wndw_7d_roll30d_Mean",
    "weibull_jd_30d_wndw_7d_roll90d_Mean",
    "weibull_jd_30d_wndw_7d_roll365d_Mean",
    "Flow_cfs_roll30d_Mean",
    "Flow_cfs_roll90d_Mean",
    "Flow_cfs_roll365d_Mean",
    "Flow_mm_roll30d_Mean",
    "Flow_mm_roll90d_Mean",
    "Flow_mm_roll365d_Mean",
    "tmmx_mean",
    "tmmn_mean",
    "pr_mean",
    "pet_mean",
    "soilm_0_10cm_mean",
    "soilm_10_40cm_mean",
    "soilm_40_100cm_mean",
    "swe_mean",
    "tmmx_mean_roll30d_Mean",
    "tmmx_mean_roll90d_Mean",
    "tmmx_mean_roll365d_Mean",
    "tmmn_mean_roll30d_Mean",
    "tmmn_mean_roll90d_Mean",
    "tmmn_mean_roll365d_Mean",
    "pr_mean_roll30d_Sum",
    "pr_mean_roll90d_Sum",
    "pr_mean_roll365d_Sum",
    "pet_mean_roll30d_Sum",
    "pet_mean_roll90d_Sum",
    "pet_mean_roll365d_Sum",
    "soilm_0_10cm_mean_roll30d_Mean",
    "soilm_0_10cm_mean_roll90d_Mean",
    "soilm_0_10cm_mean_roll365d_Mean",
    "soilm_10_40cm_mean_roll30d_Mean",
    "soilm_10_40cm_mean_roll90d_Mean",
    "soilm_10_40cm_mean_roll365d_Mean",
    "soilm_40_100cm_mean_roll30d_Mean",
    "soilm_40_100cm_mean_roll90d_Mean",
    "soilm_40_100cm_mean_roll365d_Mean",
    "swe_mean_roll30d_Mean",
    "swe_mean_roll90d_Mean",
    "swe_mean_roll365d_Mean",
    "tmax7day_gefs_mean",
    "tmin7day_gefs_mean",
    "tp7day_gefs_mean",
    "tmax10day_gefs_mean",
    "tmin10day_gefs_mean",
    "tp10day_gefs_mean",
    "tmax1to7day_gefs_mean",
    "tmin1to7day_gefs_mean",
    "tp1to7day_gefs_mean",
    "tmax1to10day_gefs_mean",
    "tmin1to10day_gefs_mean",
    "tp1to10day_gefs_mean",
    "tref15day_nmme_mean",
    "precip15day_nmme_mean",
    "tref45day_nmme_mean",
    "precip45day_nmme_mean",
    "tref75day_nmme_mean",
    "precip75day_nmme_mean",
    "tref105day_nmme_mean",
    "precip105day_nmme_mean",
    "AMO",
    "ENSO",
    "PDO",
    "PNA",
    "sunspots",
    "irrigation_Mean20Yr_sum",
    "publicsupply_Mean20Yr_sum",
]


def attach_static(weekly: pd.DataFrame, path: Path = STATIC_PATH) -> pd.DataFrame:
    if not path.exists():
        logger.info("no %s; skip static join", path)
        return weekly
    static = pd.read_csv(path)
    static["StaID"] = static["StaID"].astype(str).str.zfill(8)
    keep = ["StaID"] + [c for c in STATIC_COLS if c in static.columns]
    out = weekly.merge(static[keep], on="StaID", how="left")
    if "S-SqKm" in out.columns:
        area = pd.to_numeric(out["S-SqKm"], errors="coerce")
        out["Flow_mm"] = np.where(area > 0, out["Flow_cfs"] * CFS_TO_MM / area, np.nan)
    logger.info("static join: %s / %s weeks have S-SqKm", int(out["S-SqKm"].notna().sum()), len(out))
    return out


def attach_climate(weekly: pd.DataFrame, path: Path = CLIMATE_PATH) -> pd.DataFrame:
    if not path.exists():
        logger.info("no %s; skip climate join", path)
        return weekly
    clim = pd.read_parquet(path)
    out = weekly.merge(clim, on=["year", "month"], how="left")
    logger.info("climate join: %s / %s weeks have ENSO", int(out["ENSO"].notna().sum()), len(out))
    return out


def attach_swe(weekly: pd.DataFrame, path: Path = SWE_DAILY) -> pd.DataFrame:
    if not path.exists():
        logger.info("no %s; skip SWE join", path)
        return weekly
    daily = pd.read_parquet(path)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["week_dt"] = daily["date"] - pd.to_timedelta(daily["date"].dt.dayofweek, unit="D")
    wx = daily.groupby(["StaID", "week_dt"], sort=False)["swe_mean"].mean().reset_index()
    out = weekly.merge(wx, on=["StaID", "week_dt"], how="left")
    logger.info("SWE join: %s / %s weeks have swe_mean", int(out["swe_mean"].notna().sum()), len(out))
    return out


def attach_gefs(weekly: pd.DataFrame, path: Path = GEFS_DAILY) -> pd.DataFrame:
    if not path.exists():
        logger.info("no %s; skip GEFS join", path)
        return weekly
    daily = pd.read_parquet(path)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["week_dt"] = daily["date"] - pd.to_timedelta(daily["date"].dt.dayofweek, unit="D")
    cols = [c for c in daily.columns if c.endswith("_gefs_mean")]
    wx = daily.groupby(["StaID", "week_dt"], sort=False)[cols].mean().reset_index()
    out = weekly.merge(wx, on=["StaID", "week_dt"], how="left")
    logger.info(
        "GEFS join: %s / %s weeks have tmax7day_gefs_mean",
        int(out["tmax7day_gefs_mean"].notna().sum()) if "tmax7day_gefs_mean" in out else 0,
        len(out),
    )
    return out


def attach_nmme(
    weekly: pd.DataFrame,
    path: Path = NMME_MONTHLY,
    hist_path: Path = WEEKLY_HIST_PATH,
) -> pd.DataFrame:
    if not path.exists():
        logger.info("no %s; skip NMME join", path)
        return weekly
    monthly = pd.read_parquet(path)
    out = weekly.merge(monthly, on=["StaID", "year", "month"], how="left")
    pairs = [
        ("tref15day_nmme_anom", "tref15day_nmme_mean"),
        ("tref45day_nmme_anom", "tref45day_nmme_mean"),
        ("tref75day_nmme_anom", "tref75day_nmme_mean"),
        ("tref105day_nmme_anom", "tref105day_nmme_mean"),
        ("precip15day_nmme_anom", "precip15day_nmme_mean"),
        ("precip45day_nmme_anom", "precip45day_nmme_mean"),
        ("precip75day_nmme_anom", "precip75day_nmme_mean"),
        ("precip105day_nmme_anom", "precip105day_nmme_mean"),
    ]
    have_anom = [a for a, _ in pairs if a in out.columns]
    if have_anom and hist_path.exists():
        hist_cols = ["StaID", "month"] + [h for _, h in pairs if h in pq.read_schema(hist_path).names]
        hist = pq.read_table(hist_path, columns=hist_cols).to_pandas()
        clim = hist.groupby(["StaID", "month"], sort=False).mean(numeric_only=True).reset_index()
        clim = clim.rename(columns={h: f"{h}_clim" for _, h in pairs if h in clim.columns})
        out["month"] = out["month"].astype("int64")
        clim["month"] = clim["month"].astype("int64")
        out = out.merge(clim, on=["StaID", "month"], how="left")
        for anom, hist_name in pairs:
            clim_name = f"{hist_name}_clim"
            if anom in out.columns and clim_name in out.columns:
                out[hist_name] = out[anom] + out[clim_name]
        out = out.drop(columns=[c for c in out.columns if c.endswith("_clim") or c.endswith("_anom")])
    logger.info(
        "NMME join: %s / %s weeks have tref15day_nmme_mean",
        int(out["tref15day_nmme_mean"].notna().sum()) if "tref15day_nmme_mean" in out else 0,
        len(out),
    )
    return out


def attach_water_use_mean(weekly: pd.DataFrame, hist_path: Path = WEEKLY_HIST_PATH) -> pd.DataFrame:
    if not hist_path.exists():
        return weekly
    cols = ["StaID", "irrigation_Mean20Yr_sum", "publicsupply_Mean20Yr_sum"]
    schema = set(pq.read_schema(hist_path).names)
    cols = [c for c in cols if c in schema]
    if len(cols) < 2:
        return weekly
    hist = pq.read_table(hist_path, columns=cols).to_pandas()
    means = hist.groupby("StaID", sort=False).mean(numeric_only=True).reset_index()
    out = weekly.merge(means, on="StaID", how="left")
    logger.info("20-year water-use means attached")
    return out


def add_flow_scores(
    weekly: pd.DataFrame,
    daily: pd.DataFrame,
    hist_path: Path = WEEKLY_HIST_PATH,
) -> pd.DataFrame:
    """Live 7-day mean flow and the two USGS percentile scores."""
    hist = pq.read_table(
        hist_path,
        columns=["StaID", "jd", "mean_value_7d"],
    ).to_pandas()
    hist["StaID"] = hist["StaID"].astype(str)
    hist = hist.dropna(subset=["mean_value_7d", "jd"])

    flow = daily.dropna(subset=["discharge_cfs"]).copy()
    flow["StaID"] = flow["StaID"].astype(str)
    if flow.duplicated(["StaID", "date"]).any():
        raise RuntimeError(
            "Daily discharge still has multiple series per station-day."
        )
    flow = flow[["StaID", "date", "discharge_cfs"]].sort_values(["StaID", "date"])
    flow["mean_value_7d"] = flow.groupby("StaID", sort=False)["discharge_cfs"].transform(
        lambda s: s.rolling(7, min_periods=4).mean()
    )
    flow["week_dt"] = flow["date"] - pd.to_timedelta(flow["date"].dt.dayofweek, unit="D")
    week7 = (
        flow.groupby(["StaID", "week_dt"], sort=False)["mean_value_7d"]
        .mean()
        .reset_index()
    )
    out = weekly.merge(week7, on=["StaID", "week_dt"], how="left")

    by_site = {
        sta: grp["mean_value_7d"].to_numpy(dtype=float)
        for sta, grp in hist.groupby("StaID", sort=False)
    }
    needed_days: dict[str, set[int]] = {}
    for sta, jd in zip(out["StaID"], out["jd"]):
        if pd.notna(jd):
            needed_days.setdefault(str(sta), set()).add(int(jd))
    by_jd: dict[tuple[str, int], np.ndarray] = {}
    for sta, grp in hist.groupby("StaID", sort=False):
        days = needed_days.get(str(sta))
        if not days:
            continue
        jd = grp["jd"].to_numpy(dtype=int)
        vals = grp["mean_value_7d"].to_numpy(dtype=float)
        for day in days:
            dist = np.minimum(np.abs(jd - day), 366 - np.abs(jd - day))
            sample = vals[dist <= 15]
            if sample.size:
                by_jd[(str(sta), day)] = sample

    out["weibull_jd_30d_wndw_7d"] = [
        _weibull_new(by_jd.get((str(sta), int(jd) if pd.notna(jd) else -1)), val)
        for sta, jd, val in zip(out["StaID"], out["jd"], out["mean_value_7d"])
    ]
    out["weibull_site_7d"] = [
        _weibull_new(by_site.get(str(sta)), val)
        for sta, val in zip(out["StaID"], out["mean_value_7d"])
    ]
    logger.info(
        "flow scores: %s / %s weeks have weibull_jd_30d_wndw_7d",
        int(out["weibull_jd_30d_wndw_7d"].notna().sum()),
        len(out),
    )
    return out


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


def add_rolls(weekly: pd.DataFrame) -> pd.DataFrame:
    """30/90/365-day rolls from the weekly series (5, 13, 52 weeks)."""
    out = weekly.sort_values(["StaID", "Date"]).copy()
    windows = [("roll30d", 5), ("roll90d", 13), ("roll365d", 52)]
    mean_cols = [
        c
        for c in (
            "Flow_cfs",
            "Flow_mm",
            "mean_value_7d",
            "weibull_jd_30d_wndw_7d",
            "tmmx_mean",
            "tmmn_mean",
            "soilm_0_10cm_mean",
            "soilm_10_40cm_mean",
            "soilm_40_100cm_mean",
            "swe_mean",
        )
        if c in out.columns
    ]
    sum_cols = [c for c in ("pr_mean", "pet_mean") if c in out.columns]
    grouped = out.groupby("StaID", sort=False)
    for label, span in windows:
        minp = max(2, span // 2)
        for col in mean_cols:
            name = f"{col}_{label}_Mean"
            out[name] = grouped[col].transform(
                lambda s, span=span, minp=minp: s.rolling(span, min_periods=minp).mean()
            )
        for col in sum_cols:
            name = f"{col}_{label}_Sum"
            # Weekly pr/pet are daily-rate means. USGS's 30/90/365-day sums
            # add daily millimeters, so each week counts as seven days.
            out[name] = grouped[col].transform(
                lambda s, span=span, minp=minp: s.rolling(span, min_periods=minp).sum()
                * 7.0
            )
    logger.info("added rolling columns")
    return out


def write_hist_matched(
    hist_path: Path = WEEKLY_HIST_PATH,
    out_path: Path = WEEKLY_HIST_MATCHED,
) -> Path:
    schema = set(pq.read_schema(hist_path).names)
    cols = [c for c in HIST_KEEP if c in schema]
    table = pq.read_table(hist_path, columns=cols)
    frame = table.to_pandas()
    frame = attach_static(frame)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_parquet(frame, out_path)
    logger.info(
        "wrote matched history %s rows %s cols -> %s (%.1f GB)",
        len(frame),
        frame.shape[1],
        out_path,
        out_path.stat().st_size / 1e9,
    )
    return out_path
