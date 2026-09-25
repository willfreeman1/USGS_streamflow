"""Four-week LightGBM and persistence on the matched weekly table.

Target: weibull_jd_30d_wndw_7d four weeks later (0–100 drought percentile).
Persistence: that same column today. Station identifier is not a feature;
static drainage-area columns carry site differences.
"""

from __future__ import annotations

from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd

from streamflow.matched import HIST_KEEP, STATIC_COLS

LEAD_WEEKS = 4
LEAD_DAYS = 7 * LEAD_WEEKS
TARGET_COL = "weibull_jd_30d_wndw_7d"
DROUGHT_PERCENTILE = 10.0
# Written from the 2000–2012 table only. Do not retune on later years.
LOCKED_DROUGHT_CUTOFF = 0.70
# Promote if trailing-52-week recall gap (model − already-dry) is below this.
LOCKED_RECALL_GAP_MIN = 0.10
# Watch only: current-flow mix vs the 1999 training mix. Do not promote on this.
LOCKED_FLOW_SHIFT_WATCH = 0.20
# How many consecutive labeled Mondays make an accuracy window.
LOCKED_TRAIL_WEEKS = 52

# Years used to compare retrain rules.
DESIGN_FIT_END = date(1999, 12, 31)
DESIGN_WALK_START = date(2000, 1, 1)
DESIGN_WALK_END = date(2012, 12, 31)
HOLD_START = date(2013, 1, 1)
HOLD_END = date(2020, 3, 30)
GATE_END = date(2026, 12, 31)

ID_COLS = {"Date", "week_dt", "year", "wy", "cy", "StaID"}
# Calendar pieces we keep as features (not the year itself).
FEATURE_ALLOW = (set(HIST_KEEP) | set(STATIC_COLS)) - ID_COLS


def feature_columns(frame: pd.DataFrame) -> list[str]:
    cols = []
    for name in frame.columns:
        if name in FEATURE_ALLOW and pd.api.types.is_numeric_dtype(frame[name]):
            cols.append(name)
    return cols


def add_lead_target(frame: pd.DataFrame, *, lead_weeks: int = LEAD_WEEKS) -> pd.DataFrame:
    """Attach the percentile that actually occurs `lead_weeks` later (28 days)."""
    out = frame.sort_values(["StaID", "Date"]).copy()
    future = out[["StaID", "Date", TARGET_COL]].rename(
        columns={"Date": "target_date", TARGET_COL: "target"}
    )
    out["target_date"] = out["Date"] + pd.Timedelta(days=7 * lead_weeks)
    out = out.merge(future, on=["StaID", "target_date"], how="inner")
    return out


def _safe_names(cols: list[str]) -> list[str]:
    return [c.replace("-", "_") for c in cols]


def fit_lgbm(
    train: pd.DataFrame,
    feature_cols: list[str],
    *,
    num_boost_round: int = 150,
) -> lgb.Booster:
    x = train[feature_cols]
    y = train["target"].to_numpy(dtype=np.float32)
    dataset = lgb.Dataset(
        x,
        label=y,
        feature_name=_safe_names(feature_cols),
        free_raw_data=False,
    )
    params = {
        "objective": "regression",
        "metric": "l1",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "num_threads": 16,
    }
    return lgb.train(params, dataset, num_boost_round=num_boost_round)


def predict_lgbm(model: lgb.Booster, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return model.predict(frame[feature_cols])


def persistence_predict(frame: pd.DataFrame) -> np.ndarray:
    return frame[TARGET_COL].to_numpy(dtype=np.float64)


def add_drought_label(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["drought"] = (out["target"] <= DROUGHT_PERCENTILE).astype(np.int8)
    return out


def persistence_drought(frame: pd.DataFrame) -> np.ndarray:
    """This week's percentile at or below 10, as a 0/1 forecast of week +4."""
    return (frame[TARGET_COL] <= DROUGHT_PERCENTILE).to_numpy(dtype=np.int8)


def fit_lgbm_drought(
    train: pd.DataFrame,
    feature_cols: list[str],
    *,
    num_boost_round: int = 150,
) -> lgb.Booster:
    y = train["drought"].to_numpy(dtype=np.float32)
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    weight = (n_neg / n_pos) if n_pos else 1.0
    dataset = lgb.Dataset(
        train[feature_cols],
        label=y,
        feature_name=_safe_names(feature_cols),
        free_raw_data=False,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "scale_pos_weight": weight,
        "verbosity": -1,
        "num_threads": 16,
    }
    return lgb.train(params, dataset, num_boost_round=num_boost_round)


def predict_drought_proba(
    model: lgb.Booster, frame: pd.DataFrame, feature_cols: list[str]
) -> np.ndarray:
    return model.predict(frame[feature_cols])


def drought_metrics(y_true: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    """y_hat is 0/1. Recall = share of real drought weeks we flagged."""
    y = np.asarray(y_true).astype(bool)
    h = np.asarray(y_hat).astype(bool)
    n_dry = int(y.sum())
    n_wet = int((~y).sum())
    tp = int((y & h).sum())
    fp = int((~y & h).sum())
    return {
        "n": int(y.size),
        "n_drought": n_dry,
        "recall": (tp / n_dry) if n_dry else np.nan,
        "false_alarm_rate": (fp / n_wet) if n_wet else np.nan,
        "precision": (tp / (tp + fp)) if (tp + fp) else np.nan,
    }


def metrics(y_true: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_hat)
    y = y_true[mask]
    p = y_hat[mask]
    if y.size == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "drought_hit": np.nan}
    mae = float(np.mean(np.abs(y - p)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    dry_true = y <= DROUGHT_PERCENTILE
    dry_hat = p <= DROUGHT_PERCENTILE
    if dry_true.sum() == 0:
        hit = np.nan
    else:
        hit = float((dry_true & dry_hat).sum() / dry_true.sum())
    return {"n": int(y.size), "mae": mae, "rmse": rmse, "drought_hit": hit}
