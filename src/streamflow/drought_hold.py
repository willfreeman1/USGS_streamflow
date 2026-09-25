"""Score the locked 0.70 drought call on 2013–2020. Does not retune."""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from streamflow.config import (
    DROUGHT_HOLD_PATH,
    DROUGHT_HOLD_YEAR_PATH,
    WEEKLY_HIST_MATCHED,
)
from streamflow.model import (
    DESIGN_FIT_END,
    HOLD_END,
    HOLD_START,
    LOCKED_DROUGHT_CUTOFF,
    add_drought_label,
    add_lead_target,
    drought_metrics,
    feature_columns,
    fit_lgbm_drought,
    persistence_drought,
    predict_drought_proba,
)

logger = logging.getLogger(__name__)


def run_hold(hist_path=WEEKLY_HIST_MATCHED) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not hist_path.exists():
        raise RuntimeError(f"Missing {hist_path}; run build-weekly --matched-history")

    logger.info("loading %s", hist_path)
    raw = pd.read_parquet(hist_path)
    raw["Date"] = pd.to_datetime(raw["Date"])
    frame = add_drought_label(add_lead_target(raw))
    frame = frame.dropna(subset=["target", "drought"])
    feats = feature_columns(frame)
    train = frame.loc[frame["target_date"] <= pd.Timestamp(DESIGN_FIT_END)]
    hold = frame.loc[
        frame["target_date"].between(
            pd.Timestamp(HOLD_START), pd.Timestamp(HOLD_END)
        )
    ]
    logger.info(
        "train %s rows through %s; hold %s rows %s to %s; cutoff %.2f",
        len(train),
        DESIGN_FIT_END,
        len(hold),
        HOLD_START,
        HOLD_END,
        LOCKED_DROUGHT_CUTOFF,
    )
    model = fit_lgbm_drought(train, feats)
    proba = predict_drought_proba(model, hold, feats)
    y = hold["drought"].to_numpy()
    pers = persistence_drought(hold)
    hat = proba >= LOCKED_DROUGHT_CUTOFF

    pers_m = drought_metrics(y, pers)
    model_m = drought_metrics(y, hat)
    summary = pd.DataFrame(
        [
            {
                "window": "2013-2020",
                "rule": "this_week_already_dry",
                "cutoff": None,
                **pers_m,
            },
            {
                "window": "2013-2020",
                "rule": "locked_1999_model",
                "cutoff": LOCKED_DROUGHT_CUTOFF,
                **model_m,
            },
        ]
    )
    logger.info(
        "hold already_dry recall=%.3f false_alarm=%.3f; locked_%.2f recall=%.3f false_alarm=%.3f precision=%.3f",
        pers_m["recall"],
        pers_m["false_alarm_rate"],
        LOCKED_DROUGHT_CUTOFF,
        model_m["recall"],
        model_m["false_alarm_rate"],
        model_m["precision"],
    )

    year_rows = []
    for year in range(HOLD_START.year, HOLD_END.year + 1):
        mask = hold["target_date"].dt.year == year
        if int(mask.sum()) == 0:
            continue
        yy = y[mask.to_numpy()]
        pm = drought_metrics(yy, pers[mask.to_numpy()])
        mm = drought_metrics(yy, hat[mask.to_numpy()])
        year_rows.append(
            {
                "year": year,
                "n": int(mask.sum()),
                "n_drought": int(yy.sum()),
                "persistence_recall": pm["recall"],
                "persistence_false_alarm_rate": pm["false_alarm_rate"],
                "locked_recall": mm["recall"],
                "locked_false_alarm_rate": mm["false_alarm_rate"],
                "locked_precision": mm["precision"],
            }
        )
        logger.info(
            "%s drought_weeks=%s persist_recall=%.3f locked_recall=%.3f locked_far=%.3f",
            year,
            int(yy.sum()),
            pm["recall"],
            mm["recall"],
            mm["false_alarm_rate"],
        )

    yearly = pd.DataFrame(year_rows)
    DROUGHT_HOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(DROUGHT_HOLD_PATH, index=False)
    yearly.to_parquet(DROUGHT_HOLD_YEAR_PATH, index=False)
    logger.info("wrote %s and %s", DROUGHT_HOLD_PATH, DROUGHT_HOLD_YEAR_PATH)
    return summary, yearly


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Score the locked 0.70 drought call on 2013–2020. Does not retune."
    )
    parser.parse_args(argv)
    summary, yearly = run_hold()
    print(summary.to_string(index=False))
    print(yearly.to_string(index=False))


if __name__ == "__main__":
    main()
