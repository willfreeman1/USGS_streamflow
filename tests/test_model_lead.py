from datetime import datetime

import pandas as pd

from streamflow.drought_gate import _stitch_prepare
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


def test_stitch_prefers_live_overlap_and_crosses_boundary(tmp_path):
    hist_dates = pd.date_range("2020-03-02", periods=5, freq="7D")
    live_dates = pd.date_range("2020-03-30", periods=5, freq="7D")
    hist = pd.DataFrame(
        {
            "StaID": ["00000001"] * 5,
            "Date": hist_dates,
            TARGET_COL: [1.0, 2.0, 3.0, 4.0, 999.0],
        }
    )
    live = pd.DataFrame(
        {
            "StaID": ["00000001"] * 5,
            "Date": live_dates,
            TARGET_COL: [5.0, 6.0, 7.0, 8.0, 9.0],
        }
    )
    hist_path = tmp_path / "hist.parquet"
    live_path = tmp_path / "live.parquet"
    hist.to_parquet(hist_path, index=False)
    live.to_parquet(live_path, index=False)

    stitched = _stitch_prepare(hist_path, live_path)
    boundary = stitched.loc[stitched["Date"] == pd.Timestamp("2020-03-30")].iloc[0]
    assert boundary[TARGET_COL] == 5.0
    assert boundary["target_date"] == pd.Timestamp("2020-04-27")
    assert boundary["target"] == 9.0
