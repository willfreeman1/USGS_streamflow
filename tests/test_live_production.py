from datetime import date

from streamflow.live_gate import (
    REQUIRED_RECENT_COLUMNS,
    promote_allowed,
    validate_recent_inputs,
    weeks_since,
)
from streamflow.notify import format_weekly_email
from streamflow.dashboard import write_dashboard
from streamflow import weekly_job
from streamflow.registry import load_model
from streamflow.storage import atomic_parquet
import lightgbm as lgb
import pandas as pd
import pytest


def test_promote_needs_52_weeks_and_thin_lead():
    end = date(2026, 9, 14)
    assert promote_allowed(
        n_weeks=51,
        recall_gap=0.05,
        last_promote=date(2019, 9, 23),
        window_end=end,
    ) is False
    assert promote_allowed(
        n_weeks=52,
        recall_gap=0.16,
        last_promote=date(2019, 9, 23),
        window_end=end,
    ) is False
    assert promote_allowed(
        n_weeks=52,
        recall_gap=0.05,
        last_promote=date(2019, 9, 23),
        window_end=end,
    ) is True
    assert promote_allowed(
        n_weeks=52,
        recall_gap=0.05,
        last_promote=date(2026, 8, 17),
        window_end=end,
    ) is False


def test_weeks_since_counts_mondays():
    assert weeks_since(date(2019, 9, 23), date(2020, 9, 21)) == 52


def test_email_names_keep_and_replace():
    keep = {
        "kind": "keep",
        "end_date": date(2026, 9, 14),
        "installed_through": "2019-09-23",
        "n_weeks": 52,
        "locked_recall": 0.675,
        "persistence_recall": 0.519,
        "recall_gap": 0.156,
        "locked_false_alarm_rate": 0.176,
        "flow_shift": 0.204,
        "summary": "Keep the current model.",
    }
    subject, body = format_weekly_email(keep, [])
    assert "keep" in subject.lower()
    assert "15.6" in body
    assert "10 points" in body

    replace = dict(keep, kind="replace")
    subject, _ = format_weekly_email(replace, ["NLDAS: unpublished"])
    assert "retrain" in subject.lower()


def test_dashboard_writes_tiny_files(tmp_path):
    trail = pd.DataFrame(
        [
            {
                "end_date": "2026-09-14",
                "source": "live",
                "installed_through": "2019-09-23",
                "promote": False,
                "n": 100,
                "n_weeks": 52,
                "n_drought": 20,
                "locked_recall": 0.68,
                "persistence_recall": 0.52,
                "recall_gap": 0.16,
                "locked_false_alarm_rate": 0.18,
                "flow_shift": 0.20,
                "watch": True,
            }
        ]
    )
    dest = write_dashboard(trail, docs_dir=tmp_path)
    assert dest.exists()
    assert (tmp_path / "data" / "latest.json").exists()
    assert (tmp_path / "data" / "trail.csv").exists()
    html = dest.read_text(encoding="utf-8")
    assert "0.16" in html or "16" in html
    assert "Keep" in html


def test_publish_failure_is_reported(monkeypatch):
    monkeypatch.setattr(weekly_job, "write_dashboard", lambda: None)

    def fail_push():
        raise RuntimeError("push rejected")

    monkeypatch.setattr(weekly_job, "push_docs", fail_push)
    decision = weekly_job.run_weekly(
        ingest=False,
        score=False,
        mail=False,
        push=True,
    )
    assert decision["publish_failed"] is True
    assert any("push rejected" in warning for warning in decision["warnings"])


def test_required_ingest_failure_blocks_scoring(monkeypatch):
    scored = False

    def should_not_score():
        nonlocal scored
        scored = True

    monkeypatch.setattr(weekly_job, "refresh_sources", lambda warnings: False)
    monkeypatch.setattr(weekly_job, "score_current_window", should_not_score)
    monkeypatch.setattr(weekly_job, "write_dashboard", lambda: None)

    decision = weekly_job.run_weekly(
        ingest=True,
        score=True,
        mail=False,
        push=False,
    )
    assert scored is False
    assert decision["kind"] == "failed"
    assert "did not score" in decision["summary"]


def test_saved_model_feature_schema_is_checked(tmp_path):
    x = pd.DataFrame({"feature_a": [0.0, 1.0] * 10})
    dataset = lgb.Dataset(x, label=[0, 1] * 10)
    model = lgb.train(
        {"objective": "binary", "verbosity": -1, "min_data_in_leaf": 1},
        dataset,
        num_boost_round=1,
    )
    path = tmp_path / "model.txt"
    model.save_model(str(path))
    load_model(path, expected_features=["feature_a"])
    with pytest.raises(RuntimeError, match="feature schema"):
        load_model(path, expected_features=["feature_b"])


def test_recent_inputs_must_be_fresh_and_complete():
    dates = [pd.Timestamp("2026-09-07"), pd.Timestamp("2026-09-14")]
    frame = pd.DataFrame(
        {
            "target_date": dates,
            **{column: [1.0, 1.0] for column in REQUIRED_RECENT_COLUMNS},
        }
    )
    validate_recent_inputs(frame, dates, as_of=date(2026, 9, 27))

    stale = date(2026, 10, 20)
    with pytest.raises(RuntimeError, match="Latest labeled week"):
        validate_recent_inputs(frame, dates, as_of=stale)

    frame.loc[:, "ENSO"] = float("nan")
    with pytest.raises(RuntimeError, match="ENSO=0.0%"):
        validate_recent_inputs(frame, dates, as_of=date(2026, 9, 27))


def test_atomic_parquet_replaces_file_without_temp_leftover(tmp_path):
    path = tmp_path / "state.parquet"
    atomic_parquet(pd.DataFrame({"value": [1]}), path)
    atomic_parquet(pd.DataFrame({"value": [2]}), path)
    assert pd.read_parquet(path)["value"].tolist() == [2]
    assert not path.with_suffix(".parquet.tmp").exists()
