"""Yearly recall gap and input shift on 2000–2012. Does not score live years."""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from streamflow.config import DROUGHT_MONITOR_YEAR_PATH, WEEKLY_HIST_MATCHED
from streamflow.model import (
    DESIGN_FIT_END,
    DESIGN_WALK_END,
    DESIGN_WALK_START,
    LOCKED_DROUGHT_CUTOFF,
    TARGET_COL,
    add_drought_label,
    add_lead_target,
    drought_metrics,
    feature_columns,
    fit_lgbm_drought,
    persistence_drought,
    predict_drought_proba,
)

logger = logging.getLogger(__name__)

# Inputs we can also rebuild live. Used only as a watch, not a promote rule.
DRIFT_COLS = (
    TARGET_COL,
    "pr_mean",
    "tmmx_mean",
    "soilm_0_10cm_mean",
)


def population_shift(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """How much one year's values moved vs the 1999 training mix.

    This is the population-stability index: 0 means the same mix of low/high
    values; larger means the year looks different. Empty or constant data
    returns NaN.
    """
    ref = np.asarray(ref, dtype=np.float64)
    cur = np.asarray(cur, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size < 50 or cur.size < 50:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 3:
        return float("nan")
    expected = np.clip(np.histogram(ref, bins=edges)[0] / ref.size, 1e-6, None)
    actual = np.clip(np.histogram(cur, bins=edges)[0] / cur.size, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def run_monitor(hist_path=WEEKLY_HIST_MATCHED) -> pd.DataFrame:
    if not hist_path.exists():
        raise RuntimeError(f"Missing {hist_path}; run build-weekly --matched-history")

    logger.info("loading %s", hist_path)
    raw = pd.read_parquet(hist_path)
    raw["Date"] = pd.to_datetime(raw["Date"])
    frame = add_drought_label(add_lead_target(raw))
    frame = frame.dropna(subset=["target", "drought"])
    feats = feature_columns(frame)
    train = frame.loc[frame["target_date"] <= pd.Timestamp(DESIGN_FIT_END)]
    walk = frame.loc[
        frame["target_date"].between(
            pd.Timestamp(DESIGN_WALK_START), pd.Timestamp(DESIGN_WALK_END)
        )
    ]
    logger.info(
        "train %s rows through %s; walk %s rows; cutoff %.2f",
        len(train),
        DESIGN_FIT_END,
        len(walk),
        LOCKED_DROUGHT_CUTOFF,
    )
    model = fit_lgbm_drought(train, feats)
    proba = predict_drought_proba(model, walk, feats)
    y = walk["drought"].to_numpy()
    pers = persistence_drought(walk)
    hat = proba >= LOCKED_DROUGHT_CUTOFF

    ref = {col: train[col].to_numpy() for col in DRIFT_COLS if col in train.columns}
    rows = []
    for year in range(DESIGN_WALK_START.year, DESIGN_WALK_END.year + 1):
        mask = walk["target_date"].dt.year == year
        yy = y[mask.to_numpy()]
        pm = drought_metrics(yy, pers[mask.to_numpy()])
        mm = drought_metrics(yy, hat[mask.to_numpy()])
        year_frame = walk.loc[mask]
        row = {
            "year": year,
            "n": int(mask.sum()),
            "n_drought": int(yy.sum()),
            "drought_share": float(yy.mean()) if yy.size else float("nan"),
            "persistence_recall": pm["recall"],
            "locked_recall": mm["recall"],
            "recall_gap": mm["recall"] - pm["recall"],
            "locked_false_alarm_rate": mm["false_alarm_rate"],
            "persistence_false_alarm_rate": pm["false_alarm_rate"],
        }
        for col, values in ref.items():
            row[f"shift_{col}"] = population_shift(values, year_frame[col].to_numpy())
        rows.append(row)
        logger.info(
            "%s locked_recall=%.3f persist=%.3f gap=%.3f far=%.3f shift_flow=%.3f",
            year,
            mm["recall"],
            pm["recall"],
            mm["recall"] - pm["recall"],
            mm["false_alarm_rate"],
            row.get(f"shift_{TARGET_COL}", float("nan")),
        )

    yearly = pd.DataFrame(rows)
    DROUGHT_MONITOR_YEAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    yearly.to_parquet(DROUGHT_MONITOR_YEAR_PATH, index=False)
    logger.info("wrote %s", DROUGHT_MONITOR_YEAR_PATH)
    return yearly


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Measure 2000–2012 recall gap and input shift at the locked 0.70 line."
    )
    parser.parse_args(argv)
    yearly = run_monitor()
    print(yearly.to_string(index=False))
    print(
        "gap min=%.3f median=%.3f; flow-shift max=%.3f"
        % (
            yearly["recall_gap"].min(),
            yearly["recall_gap"].median(),
            yearly[f"shift_{TARGET_COL}"].max(),
        )
    )


if __name__ == "__main__":
    main()
