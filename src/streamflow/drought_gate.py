"""Walk the locked promote rule on trailing 52-week windows. Does not retune."""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from streamflow.config import (
    DROUGHT_GATE_TRAIL_PATH,
    DROUGHT_GATE_YEAR_PATH,
    WEEKLY_HIST_MATCHED,
    WEEKLY_LIVE_PATH,
)
from streamflow.drought_monitor import population_shift
from streamflow.storage import atomic_parquet
from streamflow.model import (
    DESIGN_FIT_END,
    DESIGN_WALK_END,
    DESIGN_WALK_START,
    HOLD_END,
    HOLD_START,
    LOCKED_DROUGHT_CUTOFF,
    LOCKED_FLOW_SHIFT_WATCH,
    LOCKED_RECALL_GAP_MIN,
    LOCKED_TRAIL_WEEKS,
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

# A hole larger than this breaks a trailing window (2020 vs 2024).
MAX_WEEK_GAP_DAYS = 10


def _prepare(path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    raw["Date"] = pd.to_datetime(raw["Date"])
    frame = add_drought_label(add_lead_target(raw))
    return frame.dropna(subset=["target", "drought"])


def _stitch_prepare(hist_path, live_path) -> pd.DataFrame:
    """One labeled table so March 2020 hist weeks can use April 2020 live outcomes."""
    hist = pd.read_parquet(hist_path)
    live = pd.read_parquet(live_path)
    hist["Date"] = pd.to_datetime(hist["Date"])
    live["Date"] = pd.to_datetime(live["Date"])
    cols = [c for c in hist.columns if c in live.columns]
    combo = pd.concat([hist[cols], live[cols]], ignore_index=True)
    combo = (
        combo.sort_values(["StaID", "Date"])
        .drop_duplicates(["StaID", "Date"], keep="last")
        .reset_index(drop=True)
    )
    frame = add_drought_label(add_lead_target(combo))
    return frame.dropna(subset=["target", "drought"])


def _train_through(frame: pd.DataFrame, end) -> pd.DataFrame:
    return frame.loc[frame["target_date"] <= pd.Timestamp(end)]


def _align(frame: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    missing = [c for c in feats if c not in frame.columns]
    if not missing:
        return frame
    out = frame.copy()
    for col in missing:
        out[col] = np.nan
    return out


def _consecutive_runs(dates: list[pd.Timestamp]) -> list[list[pd.Timestamp]]:
    if not dates:
        return []
    ordered = sorted(dates)
    runs: list[list[pd.Timestamp]] = [[ordered[0]]]
    for day in ordered[1:]:
        if (day - runs[-1][-1]).days <= MAX_WEEK_GAP_DAYS:
            runs[-1].append(day)
        else:
            runs.append([day])
    return runs


def _window_metrics(
    frame: pd.DataFrame,
    hat: np.ndarray,
    pers: np.ndarray,
    dates: list[pd.Timestamp],
    ref_flow: np.ndarray,
) -> dict:
    mask = frame["target_date"].isin(dates)
    y = frame.loc[mask, "drought"].to_numpy()
    mm = drought_metrics(y, hat[mask.to_numpy()])
    pm = drought_metrics(y, pers[mask.to_numpy()])
    gap = mm["recall"] - pm["recall"]
    shift = population_shift(ref_flow, frame.loc[mask, TARGET_COL].to_numpy())
    return {
        "n": mm["n"],
        "n_weeks": len(dates),
        "n_drought": mm["n_drought"],
        "locked_recall": mm["recall"],
        "persistence_recall": pm["recall"],
        "recall_gap": gap,
        "locked_false_alarm_rate": mm["false_alarm_rate"],
        "flow_shift": shift,
        "watch": bool(shift == shift and shift > LOCKED_FLOW_SHIFT_WATCH),
    }


def _walk_segment(
    frame: pd.DataFrame,
    *,
    feats: list[str],
    model,
    installed_end,
    ref_flow: np.ndarray,
    training: pd.DataFrame,
    allow_promote: bool,
    source: str,
    min_promote_date=None,
) -> tuple[list[dict], object, object]:
    if frame.empty:
        return [], model, installed_end
    frame = frame.sort_values("target_date").reset_index(drop=True)
    aligned = _align(frame, feats)
    proba = predict_drought_proba(model, aligned, feats)
    hat = proba >= LOCKED_DROUGHT_CUTOFF
    pers = persistence_drought(frame)
    rows: list[dict] = []
    for run in _consecutive_runs(frame["target_date"].drop_duplicates().tolist()):
        if len(run) < LOCKED_TRAIL_WEEKS:
            continue
        next_promote_i = LOCKED_TRAIL_WEEKS - 1
        for end_i in range(LOCKED_TRAIL_WEEKS - 1, len(run)):
            window = run[end_i + 1 - LOCKED_TRAIL_WEEKS : end_i + 1]
            stats = _window_metrics(frame, hat, pers, window, ref_flow)
            end = pd.Timestamp(window[-1])
            promote = bool(
                allow_promote
                and end_i >= next_promote_i
                and (min_promote_date is None or end >= pd.Timestamp(min_promote_date))
                and stats["recall_gap"] < LOCKED_RECALL_GAP_MIN
            )
            rows.append(
                {
                    "end_date": end,
                    "source": (
                        "hist"
                        if source == "gate" and end <= pd.Timestamp(HOLD_END)
                        else "live"
                        if source == "gate"
                        else source
                    ),
                    "installed_through": installed_end,
                    "promote": promote,
                    **stats,
                }
            )
            if promote:
                train = _train_through(training, end)
                logger.info(
                    "promote at %s gap=%.3f; fitting %s rows through that week",
                    end.date(),
                    stats["recall_gap"],
                    len(train),
                )
                model = fit_lgbm_drought(train, feats)
                installed_end = end.date()
                from streamflow.registry import save_model

                save_model(model, installed_end, last_promote=installed_end)
                next_promote_i = end_i + LOCKED_TRAIL_WEEKS
                later = frame["target_date"] > end
                if later.any():
                    later_aligned = _align(frame.loc[later], feats)
                    later_p = predict_drought_proba(model, later_aligned, feats)
                    hat[later.to_numpy()] = later_p >= LOCKED_DROUGHT_CUTOFF
    return rows, model, installed_end


def _write_trail(trail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest = trail.loc[trail["source"] != "design_2000_2012"]
    if not latest.empty:
        last = latest.iloc[-1]
        logger.info(
            "latest trail ending %s installed=%s gap=%.3f far=%.3f watch=%s promote=%s",
            pd.Timestamp(last["end_date"]).date(),
            last["installed_through"],
            last["recall_gap"],
            last["locked_false_alarm_rate"],
            last["watch"],
            last["promote"],
        )

    DROUGHT_GATE_TRAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_parquet(trail, DROUGHT_GATE_TRAIL_PATH)
    year_rows = []
    scored = trail.loc[trail["source"].isin(("hist", "live"))]
    if not scored.empty:
        scored = scored.copy()
        scored["year"] = pd.to_datetime(scored["end_date"]).dt.year
        for year, grp in scored.groupby("year"):
            row = grp.iloc[-1].to_dict()
            row["year"] = int(str(year))
            year_rows.append(row)
    yearly = pd.DataFrame(year_rows)
    atomic_parquet(yearly, DROUGHT_GATE_YEAR_PATH)
    logger.info("wrote %s and %s", DROUGHT_GATE_TRAIL_PATH, DROUGHT_GATE_YEAR_PATH)
    return trail, yearly


def run_live_only(
    hist_path=WEEKLY_HIST_MATCHED,
    live_path=WEEKLY_LIVE_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility alias for the canonical stitched replay."""
    logger.info("--live-only now runs the canonical stitched replay")
    return run_gate(hist_path, live_path)


def run_gate(
    hist_path=WEEKLY_HIST_MATCHED,
    live_path=WEEKLY_LIVE_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not hist_path.exists():
        raise RuntimeError(f"Missing {hist_path}; run build-weekly --matched-history")

    logger.info("loading and stitching %s with %s", hist_path, live_path)
    combo = (
        _stitch_prepare(hist_path, live_path)
        if live_path.exists()
        else _prepare(hist_path)
    )
    feats = feature_columns(combo)
    initial = _train_through(combo, DESIGN_FIT_END)
    logger.info(
        "train %s rows through %s; trail %s weeks; promote if gap < %.2f",
        len(initial),
        DESIGN_FIT_END,
        LOCKED_TRAIL_WEEKS,
        LOCKED_RECALL_GAP_MIN,
    )
    model = fit_lgbm_drought(initial, feats)
    ref_flow = initial[TARGET_COL].to_numpy()

    design = combo.loc[
        combo["target_date"].between(
            pd.Timestamp(DESIGN_WALK_START), pd.Timestamp(DESIGN_WALK_END)
        )
    ]
    design_rows, _, _ = _walk_segment(
        design,
        feats=feats,
        model=model,
        installed_end=DESIGN_FIT_END,
        ref_flow=ref_flow,
        training=combo,
        allow_promote=False,
        source="design_2000_2012",
    )
    if design_rows:
        gaps = [r["recall_gap"] for r in design_rows]
        logger.info(
            "design trailing-52 gap min=%.3f median=%.3f n_windows=%s",
            float(np.min(gaps)),
            float(np.median(gaps)),
            len(gaps),
        )

    installed_end = DESIGN_FIT_END
    gate = combo.loc[combo["target_date"] >= pd.Timestamp(HOLD_START)]
    gate_rows, model, installed_end = _walk_segment(
        gate,
        feats=feats,
        model=model,
        installed_end=installed_end,
        ref_flow=ref_flow,
        training=combo,
        allow_promote=True,
        source="gate",
    )

    trail = pd.DataFrame(design_rows + gate_rows)
    return _write_trail(trail)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Walk the locked 0.10 gap on trailing 52-week windows."
    )
    parser.add_argument(
        "--live-only",
        action="store_true",
        help="Compatibility alias for the canonical full stitched replay.",
    )
    args = parser.parse_args(argv)
    trail, yearly = run_live_only() if args.live_only else run_gate()
    design = trail.loc[trail["source"] == "design_2000_2012"]
    if not design.empty:
        print(
            "design trailing-52 gap min=%.3f median=%.3f"
            % (design["recall_gap"].min(), design["recall_gap"].median())
        )
    live_or_hist = trail.loc[trail["source"].isin(("hist", "live"))]
    if not live_or_hist.empty:
        last = live_or_hist.iloc[-1]
        print("latest window")
        print(pd.DataFrame([last]).to_string(index=False))
        print(
            "promotes: %s"
            % live_or_hist.loc[live_or_hist["promote"], "end_date"].tolist()
        )
    if not yearly.empty:
        print(yearly[["year", "end_date", "recall_gap", "promote", "installed_through"]].to_string(index=False))


if __name__ == "__main__":
    main()
