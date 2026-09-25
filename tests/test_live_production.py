from datetime import date

from streamflow.live_gate import promote_allowed, weeks_since
from streamflow.notify import format_weekly_email
from streamflow.dashboard import write_dashboard
import pandas as pd


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
