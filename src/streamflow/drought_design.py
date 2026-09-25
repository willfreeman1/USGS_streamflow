"""Four-week drought yes/no on 2000–2012. Does not score 2013–2020."""

from __future__ import annotations

import argparse
import logging
from datetime import date

import numpy as np
import pandas as pd

from streamflow.config import (
    DROUGHT_CUTOFF_PATH,
    DROUGHT_RETRAIN_CUTOFF_PATH,
    DROUGHT_RETRAIN_YEAR_PATH,
    DROUGHT_YEAR_PATH,
    WEEKLY_HIST_MATCHED,
)
from streamflow.model import (
    DESIGN_FIT_END,
    DESIGN_WALK_END,
    DESIGN_WALK_START,
    HOLD_START,
    add_drought_label,
    add_lead_target,
    drought_metrics,
    feature_columns,
    fit_lgbm_drought,
    persistence_drought,
    predict_drought_proba,
)

logger = logging.getLogger(__name__)

CUTOFFS = tuple(round(x, 2) for x in np.linspace(0.10, 0.80, 15))
YEAR_CUTOFFS = (0.20, 0.35, 0.50)


def _train_through(frame: pd.DataFrame, end: date) -> pd.DataFrame:
    return frame.loc[frame["target_date"] <= pd.Timestamp(end)]


def _cutoff_row(label: str, y: np.ndarray, hat: np.ndarray) -> dict:
    m = drought_metrics(y, hat)
    return {
        "cutoff": label,
        "recall": m["recall"],
        "false_alarm_rate": m["false_alarm_rate"],
        "precision": m["precision"],
        "n_drought": m["n_drought"],
        "n": m["n"],
    }


def run_design(hist_path=WEEKLY_HIST_MATCHED) -> tuple[pd.DataFrame, ...]:
    if not hist_path.exists():
        raise RuntimeError(f"Missing {hist_path}; run build-weekly --matched-history")

    logger.info("loading %s", hist_path)
    raw = pd.read_parquet(hist_path)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.loc[raw["Date"] < pd.Timestamp(HOLD_START)]
    frame = add_drought_label(add_lead_target(raw))
    frame = frame.dropna(subset=["target", "drought"])
    feats = feature_columns(frame)
    years = list(range(DESIGN_WALK_START.year, DESIGN_WALK_END.year + 1))
    tests = {year: frame.loc[frame["target_date"].dt.year == year] for year in years}
    walk = pd.concat((tests[year] for year in years), ignore_index=True)

    initial = _train_through(frame, DESIGN_FIT_END)
    logger.info(
        "train %s rows through %s (drought share %.3f); walk %s rows",
        len(initial),
        DESIGN_FIT_END,
        float(initial["drought"].mean()),
        len(walk),
    )
    models: dict[int, object] = {1999: fit_lgbm_drought(initial, feats)}

    frozen_proba = predict_drought_proba(models[1999], walk, feats)
    y_walk = walk["drought"].to_numpy()
    pers_walk = persistence_drought(walk)

    cutoff_rows = [_cutoff_row("this_week_already_dry", y_walk, pers_walk)]
    for cut in CUTOFFS:
        row = _cutoff_row(f"{cut:.2f}", y_walk, frozen_proba >= cut)
        cutoff_rows.append(row)
        logger.info(
            "frozen cutoff %s recall=%.3f false_alarm=%.3f precision=%.3f",
            cut,
            row["recall"],
            row["false_alarm_rate"],
            row["precision"],
        )

    frozen_year_rows = []
    retrain_year_rows = []
    fresh_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    frozen_chunks: list[np.ndarray] = []
    pers_chunks: list[np.ndarray] = []

    for year in years:
        test = tests[year]
        y = test["drought"].to_numpy()
        pers = persistence_drought(test)
        frozen = predict_drought_proba(models[1999], test, feats)
        train_end_year = year - 1
        if train_end_year not in models:
            train = _train_through(frame, date(train_end_year, 12, 31))
            logger.info(
                "fitting drought model through %s (%s rows)",
                train_end_year,
                len(train),
            )
            models[train_end_year] = fit_lgbm_drought(train, feats)
        fresh = predict_drought_proba(models[train_end_year], test, feats)

        y_chunks.append(y)
        frozen_chunks.append(frozen)
        fresh_chunks.append(fresh)
        pers_chunks.append(pers)

        pm = drought_metrics(y, pers)
        frozen_row: dict = {
            "year": year,
            "n": int(len(test)),
            "n_drought": int(y.sum()),
            "persistence_recall": pm["recall"],
            "persistence_false_alarm_rate": pm["false_alarm_rate"],
        }
        retrain_row: dict = {
            "year": year,
            "n": int(len(test)),
            "n_drought": int(y.sum()),
            "persistence_recall": pm["recall"],
            "persistence_false_alarm_rate": pm["false_alarm_rate"],
        }
        for cut in YEAR_CUTOFFS:
            key = f"p{int(cut * 100):02d}"
            fm = drought_metrics(y, frozen >= cut)
            rm = drought_metrics(y, fresh >= cut)
            frozen_row[f"{key}_recall"] = fm["recall"]
            frozen_row[f"{key}_false_alarm_rate"] = fm["false_alarm_rate"]
            retrain_row[f"frozen_{key}_recall"] = fm["recall"]
            retrain_row[f"frozen_{key}_false_alarm_rate"] = fm["false_alarm_rate"]
            retrain_row[f"fresh_{key}_recall"] = rm["recall"]
            retrain_row[f"fresh_{key}_false_alarm_rate"] = rm["false_alarm_rate"]
        frozen_year_rows.append(frozen_row)
        retrain_year_rows.append(retrain_row)
        logger.info(
            "%s drought_weeks=%s persist_recall=%.3f frozen_p50=%.3f fresh_p50=%.3f",
            year,
            int(y.sum()),
            pm["recall"],
            drought_metrics(y, frozen >= 0.50)["recall"],
            drought_metrics(y, fresh >= 0.50)["recall"],
        )

    y_all = np.concatenate(y_chunks)
    frozen_all = np.concatenate(frozen_chunks)
    fresh_all = np.concatenate(fresh_chunks)
    pers_all = np.concatenate(pers_chunks)

    retrain_cutoff_rows = [
        {
            "policy": "this_week_already_dry",
            **_cutoff_row("this_week_already_dry", y_all, pers_all),
        }
    ]
    for policy, proba in (
        ("never_retrain", frozen_all),
        ("retrain_every_year", fresh_all),
    ):
        for cut in CUTOFFS:
            row = _cutoff_row(f"{cut:.2f}", y_all, proba >= cut)
            retrain_cutoff_rows.append({"policy": policy, **row})
            logger.info(
                "%s cutoff %s recall=%.3f false_alarm=%.3f",
                policy,
                cut,
                row["recall"],
                row["false_alarm_rate"],
            )

    cutoffs = pd.DataFrame(cutoff_rows)
    yearly = pd.DataFrame(frozen_year_rows)
    retrain_years = pd.DataFrame(retrain_year_rows)
    retrain_cutoffs = pd.DataFrame(retrain_cutoff_rows)
    DROUGHT_YEAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    cutoffs.to_parquet(DROUGHT_CUTOFF_PATH, index=False)
    yearly.to_parquet(DROUGHT_YEAR_PATH, index=False)
    retrain_years.to_parquet(DROUGHT_RETRAIN_YEAR_PATH, index=False)
    retrain_cutoffs.to_parquet(DROUGHT_RETRAIN_CUTOFF_PATH, index=False)
    logger.info(
        "wrote %s, %s, %s, %s",
        DROUGHT_CUTOFF_PATH,
        DROUGHT_YEAR_PATH,
        DROUGHT_RETRAIN_YEAR_PATH,
        DROUGHT_RETRAIN_CUTOFF_PATH,
    )
    return yearly, cutoffs, retrain_years, retrain_cutoffs


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Score a four-week drought yes/no model on 2000–2012."
    )
    parser.parse_args(argv)
    yearly, cutoffs, retrain_years, retrain_cutoffs = run_design()
    print(cutoffs.to_string(index=False))
    print(yearly.to_string(index=False))
    print(retrain_cutoffs.to_string(index=False))
    print(retrain_years.to_string(index=False))


if __name__ == "__main__":
    main()
