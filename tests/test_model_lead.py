from datetime import datetime

import pandas as pd

from streamflow.model import (
    LOCKED_DROUGHT_CUTOFF,
    LOCKED_FLOW_SHIFT_WATCH,
    LOCKED_RECALL_GAP_MIN,
    LOCKED_TRAIL_WEEKS,
    TARGET_COL,
    add_lead_target,
)


def test_drought_cutoff_is_locked_at_070():
    assert LOCKED_DROUGHT_CUTOFF == 0.70
    assert LOCKED_RECALL_GAP_MIN == 0.10
    assert LOCKED_FLOW_SHIFT_WATCH == 0.20
    assert LOCKED_TRAIL_WEEKS == 52


def test_lead_keeps_only_28_day_gaps():
    dates = pd.date_range("2000-01-03", periods=8, freq="7D")
    frame = pd.DataFrame(
        {
            "StaID": ["00000001"] * 8,
            "Date": dates,
            TARGET_COL: range(8),
        }
    )
    out = add_lead_target(frame)
    assert len(out) == 4
    assert (out["target_date"] - out["Date"]).dt.days.eq(28).all()
    assert out.iloc[0]["target"] == 4


def test_lead_drops_a_skipped_week():
    dates = [
        datetime(2000, 1, 3),
        datetime(2000, 1, 10),
        datetime(2000, 1, 24),
        datetime(2000, 1, 31),
        datetime(2000, 2, 7),
        datetime(2000, 2, 14),
    ]
    frame = pd.DataFrame(
        {"StaID": ["00000001"] * 6, "Date": dates, TARGET_COL: range(6)}
    )
    out = add_lead_target(frame)
    # 3 Jan -> 31 Jan is 28 days; 10 Jan -> 7 Feb is 28 days; 24 Jan -> 14 Feb is 21 days
    assert set(out["Date"].dt.date.astype(str)) == {"2000-01-03", "2000-01-10"}
