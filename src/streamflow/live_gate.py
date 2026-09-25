"""Score the latest 52 known weeks and apply the locked replace rule."""

from __future__ import annotations

import logging
import pandas as pd

from streamflow.config import BOOTSTRAP_INSTALLED, WEEKLY_HIST_MATCHED, WEEKLY_LIVE_PATH
from streamflow.drought_gate import (
    _align,
    _consecutive_runs,
    _prepare,
    _train_through,
    _window_metrics,
)
from streamflow.model import (
    DESIGN_FIT_END,
    LOCKED_DROUGHT_CUTOFF,
    LOCKED_RECALL_GAP_MIN,
    LOCKED_TRAIL_WEEKS,
    TARGET_COL,
    feature_columns,
    fit_lgbm_drought,
    persistence_drought,
    predict_drought_proba,
)
from streamflow.registry import load_model, load_registry, save_model

logger = logging.getLogger(__name__)


def weeks_since(start, end) -> int:
    a = pd.Timestamp(start)
    b = pd.Timestamp(end)
    return int((b - a).days // 7)


def promote_allowed(
    *,
    n_weeks: int,
    recall_gap: float,
    last_promote,
    window_end,
) -> bool:
    if n_weeks < LOCKED_TRAIL_WEEKS:
        return False
    if last_promote is not None and weeks_since(last_promote, window_end) < LOCKED_TRAIL_WEEKS:
        return False
    return bool(recall_gap < LOCKED_RECALL_GAP_MIN)


def bootstrap_model(
    hist: pd.DataFrame,
    feats: list[str],
    live: pd.DataFrame | None = None,
    through=BOOTSTRAP_INSTALLED,
):
    train = _train_through(hist, through)
    if live is not None:
        extra = _train_through(live, through)
        if not extra.empty:
            train = pd.concat([train, extra], ignore_index=True)
    logger.info("bootstrap trees through %s on %s rows", through, len(train))
    model = fit_lgbm_drought(train, feats)
    save_model(model, through, last_promote=through)
    return model


def ensure_model(hist: pd.DataFrame, feats: list[str], live: pd.DataFrame | None):
    state = load_registry()
    if state is not None:
        return load_model(), state
    model = bootstrap_model(hist, feats, live)
    state = load_registry()
    if state is None:
        raise RuntimeError("bootstrap did not write models/registry.json")
    return model, state


def score_current_window(
    hist_path=WEEKLY_HIST_MATCHED,
    live_path=WEEKLY_LIVE_PATH,
    *,
    allow_promote: bool = True,
) -> dict:
    hist = _prepare(hist_path)
    if not live_path.exists():
        raise RuntimeError(f"Missing {live_path}; run build-weekly --from-daily")
    live = _prepare(live_path)
    feats = feature_columns(hist)
    model, state = ensure_model(hist, feats, live)
    if state is None:
        raise RuntimeError("Model registry is empty after load.")
    installed = state["installed_through"]
    last_promote = state.get("last_promote") or installed
    ref_flow = _train_through(hist, DESIGN_FIT_END)[TARGET_COL].to_numpy()

    aligned = _align(live, feats)
    proba = predict_drought_proba(model, aligned, feats)
    hat = proba >= LOCKED_DROUGHT_CUTOFF
    pers = persistence_drought(live)
    runs = _consecutive_runs(live["target_date"].drop_duplicates().tolist())
    if not runs:
        return {
            "kind": "incomplete",
            "summary": "No labeled live weeks yet.",
            "end_date": None,
            "installed_through": installed,
            "promote": False,
            "warnings": [],
        }
    run = runs[-1]
    end = pd.Timestamp(run[-1])
    if len(run) < LOCKED_TRAIL_WEEKS:
        window = run
        stats = _window_metrics(live, hat, pers, window, ref_flow)
        return {
            "kind": "incomplete",
            "summary": (
                f"Only {len(run)} known weeks in a row. "
                "The replace rule waits for 52."
            ),
            "end_date": end.date(),
            "installed_through": installed,
            "promote": False,
            "warnings": [],
            **stats,
        }

    window = run[-LOCKED_TRAIL_WEEKS:]
    stats = _window_metrics(live, hat, pers, window, ref_flow)
    promote = bool(
        allow_promote
        and promote_allowed(
            n_weeks=stats["n_weeks"],
            recall_gap=stats["recall_gap"],
            last_promote=last_promote,
            window_end=end,
        )
    )
    if promote:
        train = pd.concat(
            [_train_through(hist, end), _train_through(live, end)],
            ignore_index=True,
        )
        logger.info("promote at %s gap=%.3f; fitting %s rows", end.date(), stats["recall_gap"], len(train))
        model = fit_lgbm_drought(train, feats)
        save_model(model, end.date(), last_promote=end.date())
        installed = end.date()
        kind = "replace"
        summary = (
            f"Lead was {100 * stats['recall_gap']:.1f} points, below 10. "
            "Retrained the model on all known weeks through this window."
        )
    else:
        kind = "keep"
        summary = (
            f"Lead is {100 * stats['recall_gap']:.1f} points, at or above 10. "
            "Keep the current model."
        )
    return {
        "kind": kind,
        "summary": summary,
        "end_date": end.date(),
        "installed_through": installed,
        "last_promote": last_promote if not promote else end.date(),
        "promote": promote,
        "source": "live",
        "warnings": [],
        **stats,
    }
