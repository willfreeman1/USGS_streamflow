"""Walk retrain rules through 2000–2012 only.

2013–2020 and the live 2024–2026 file are not scored here.
A decision to install a new model in year Y may use only years before Y.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

import numpy as np
import pandas as pd

from streamflow.config import RETRAIN_POLICY_PATH, RETRAIN_YEAR_PATH, WEEKLY_HIST_MATCHED
from streamflow.model import (
    DESIGN_FIT_END,
    DESIGN_WALK_END,
    DESIGN_WALK_START,
    HOLD_START,
    add_lead_target,
    feature_columns,
    fit_lgbm,
    metrics,
    persistence_predict,
    predict_lgbm,
)

logger = logging.getLogger(__name__)


def _train_through(frame: pd.DataFrame, end: date) -> pd.DataFrame:
    return frame.loc[frame["target_date"] <= pd.Timestamp(end)]


def run_design(hist_path=WEEKLY_HIST_MATCHED) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not hist_path.exists():
        raise RuntimeError(f"Missing {hist_path}; run build-weekly --matched-history")

    logger.info("loading %s", hist_path)
    raw = pd.read_parquet(hist_path)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.loc[raw["Date"] < pd.Timestamp(HOLD_START)]
    frame = add_lead_target(raw)
    frame = frame.dropna(subset=["target", "weibull_jd_30d_wndw_7d"])
    feats = feature_columns(frame)
    logger.info(
        "design rows %s; target dates %s to %s; %s features",
        len(frame),
        frame["target_date"].min().date(),
        frame["target_date"].max().date(),
        len(feats),
    )

    years = list(range(DESIGN_WALK_START.year, DESIGN_WALK_END.year + 1))
    models: dict[int, object] = {}
    tests: dict[int, pd.DataFrame] = {}
    for year in years:
        tests[year] = frame.loc[frame["target_date"].dt.year == year]

    initial = _train_through(frame, DESIGN_FIT_END)
    logger.info("fitting model through %s (%s rows)", DESIGN_FIT_END, len(initial))
    models[1999] = fit_lgbm(initial, feats)

    year_rows: list[dict] = []
    for year in years:
        test = tests[year]
        y = test["target"].to_numpy()
        pers = metrics(y, persistence_predict(test))
        froze = metrics(y, predict_lgbm(models[1999], test, feats))
        train_end_year = year - 1
        if train_end_year not in models:
            train = _train_through(frame, date(train_end_year, 12, 31))
            logger.info(
                "fitting model through %s (%s rows)", train_end_year, len(train)
            )
            models[train_end_year] = fit_lgbm(train, feats)
        fresh = metrics(y, predict_lgbm(models[train_end_year], test, feats))
        year_rows.append(
            {
                "year": year,
                "n": pers["n"],
                "persistence_mae": pers["mae"],
                "persistence_rmse": pers["rmse"],
                "persistence_drought_hit": pers["drought_hit"],
                "frozen_mae": froze["mae"],
                "frozen_rmse": froze["rmse"],
                "frozen_drought_hit": froze["drought_hit"],
                "fresh_mae": fresh["mae"],
                "fresh_rmse": fresh["rmse"],
                "fresh_drought_hit": fresh["drought_hit"],
                "fresh_beats_frozen": bool(fresh["mae"] < froze["mae"]),
                "fresh_beats_persistence": bool(fresh["mae"] < pers["mae"]),
            }
        )
        logger.info(
            "%s n=%s persistence=%.3f frozen=%.3f fresh=%.3f",
            year,
            pers["n"],
            pers["mae"],
            froze["mae"],
            fresh["mae"],
        )

    yearly = pd.DataFrame(year_rows)
    policies = _policies(yearly, models, tests, feats)
    RETRAIN_YEAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    yearly.to_parquet(RETRAIN_YEAR_PATH, index=False)
    policies.to_parquet(RETRAIN_POLICY_PATH, index=False)
    logger.info("wrote %s and %s", RETRAIN_YEAR_PATH, RETRAIN_POLICY_PATH)
    return yearly, policies


def _score_model(model, test: pd.DataFrame, feats: list[str]) -> float:
    return metrics(test["target"].to_numpy(), predict_lgbm(model, test, feats))["mae"]


def _policies(
    yearly: pd.DataFrame,
    models: dict[int, object],
    tests: dict[int, pd.DataFrame],
    feats: list[str],
) -> pd.DataFrame:
    years = yearly["year"].tolist()

    def summarize(name: str, maes: list[float], retrains: int) -> dict:
        return {
            "policy": name,
            "mean_mae": float(np.mean(maes)),
            "retrains": retrains,
            "years": len(maes),
        }

    rows = [
        summarize("persistence", yearly["persistence_mae"].tolist(), 0),
        summarize("never_retrain", yearly["frozen_mae"].tolist(), 0),
        summarize("retrain_every_year", yearly["fresh_mae"].tolist(), len(years) - 1),
    ]

    def calendar(step: int) -> dict:
        installed_end = 1999
        maes = []
        retrains = 0
        for year in years:
            if year > 2000 and (year - 2000) % step == 0:
                installed_end = year - 1
                retrains += 1
            maes.append(_score_model(models[installed_end], tests[year], feats))
        return summarize(f"retrain_every_{step}_years", maes, retrains)

    rows.append(calendar(3))

    def promote(min_rel: float, name: str) -> dict:
        """Promote using only the previous year's errors.

        Probe at the start of year Y: model trained through Y-2, scored on Y-1.
        If that probe beat the model that was actually installed in Y-1 by
        `min_rel`, install the model trained through Y-1 for year Y.
        """
        installed_end = 1999
        maes = []
        retrains = 0
        installed_mae_by_year: dict[int, float] = {}
        for year in years:
            if year >= 2002:
                probe_mae = float(
                    yearly.loc[yearly["year"] == year - 1, "fresh_mae"].iloc[0]
                )
                prior_installed = installed_mae_by_year[year - 1]
                if probe_mae < prior_installed * (1.0 - min_rel):
                    installed_end = year - 1
                    retrains += 1
            mae = _score_model(models[installed_end], tests[year], feats)
            installed_mae_by_year[year] = mae
            maes.append(mae)
        return summarize(name, maes, retrains)

    rows.append(promote(0.0, "promote_if_better_on_prior_year"))
    rows.append(promote(0.05, "promote_if_5pct_better_on_prior_year"))
    rows.append(promote(0.10, "promote_if_10pct_better_on_prior_year"))
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Walk 2000–2012 retrain rules. Does not score 2013–2020."
    )
    parser.parse_args(argv)
    yearly, policies = run_design()
    print(yearly.to_string(index=False))
    print(policies.to_string(index=False))


if __name__ == "__main__":
    main()
