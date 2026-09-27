"""Monday job: pull new days, rebuild live weeks, apply the locked gate, email."""

from __future__ import annotations

import argparse
import logging
import subprocess
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from streamflow.config import (
    DAILY_FLOW_PATH,
    LIVE_DECISIONS_PATH,
    ROOT,
)
from streamflow.dashboard import write_dashboard
from streamflow.ingest_climate import build_monthly as ingest_climate
from streamflow.ingest_flow import ingest_daily_discharge
from streamflow.ingest_gefs import OUT_DAILY as GEFS_DAILY
from streamflow.ingest_gefs import build_basin_daily as ingest_gefs
from streamflow.ingest_gridmet import OUT_DAILY as GRIDMET_DAILY
from streamflow.ingest_gridmet import VARIABLES as GRIDMET_VARS
from streamflow.ingest_gridmet import build_basin_daily as ingest_gridmet
from streamflow.ingest_gridmet import download_year as download_gridmet
from streamflow.ingest_nldas import OUT_DAILY as NLDAS_DAILY
from streamflow.ingest_nldas import build_basin_daily as ingest_nldas
from streamflow.ingest_nmme import build_basin_monthly as ingest_nmme
from streamflow.ingest_swe import OUT_DAILY as SWE_DAILY
from streamflow.ingest_swe import build_basin_daily as ingest_swe
from streamflow.live_gate import score_current_window
from streamflow.notify import format_weekly_email, send_email
from streamflow.storage import atomic_parquet

logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 10


def parquet_max_date(path: Path, column: str = "date") -> date | None:
    if not path.exists():
        return None
    series = pd.read_parquet(path, columns=[column])[column]
    if series.empty:
        return None
    return pd.to_datetime(series).max().date()


def incremental_start(path: Path, fallback_days: int = 21, column: str = "date") -> date:
    last = parquet_max_date(path, column)
    today = date.today()
    if last is None:
        return today - timedelta(days=fallback_days)
    return last - timedelta(days=LOOKBACK_DAYS)


def _run_step(name: str, fn, warnings: list[str]) -> bool:
    logger.info("step %s", name)
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        msg = f"{name}: {exc}"
        logger.warning(msg)
        warnings.append(msg)
        return False
    return True


def refresh_sources(warnings: list[str], today: date | None = None) -> bool:
    today = today or date.today()
    start_flow = incremental_start(DAILY_FLOW_PATH)
    ok = _run_step(
        "usgs flow",
        lambda: ingest_daily_discharge(start_flow, today),
        warnings,
    )

    years = [today.year]
    if today.month == 1:
        years.insert(0, today.year - 1)
    def _gridmet() -> None:
        for year in years:
            for stem in GRIDMET_VARS:
                download_gridmet(stem, year)
        ingest_gridmet(years)

    ok = _run_step("gridMET", _gridmet, warnings) and ok
    ok = _run_step(
        "NLDAS",
        lambda: ingest_nldas(incremental_start(NLDAS_DAILY), today),
        warnings,
    ) and ok
    ok = _run_step(
        "snow",
        lambda: ingest_swe(incremental_start(SWE_DAILY), today),
        warnings,
    ) and ok
    ok = _run_step(
        "GEFS",
        lambda: ingest_gefs(incremental_start(GEFS_DAILY), today),
        warnings,
    ) and ok
    month_start = date(today.year, today.month, 1)
    if today.month == 1:
        nmme_start = date(today.year - 1, 12, 1)
    else:
        nmme_start = date(today.year, today.month - 1, 1)
    ok = _run_step(
        "NMME", lambda: ingest_nmme(nmme_start, month_start), warnings
    ) and ok
    ok = _run_step("climate indexes", ingest_climate, warnings) and ok
    # GRIDMET_DAILY is only referenced so a missing file is obvious in logs
    logger.info("gridMET daily table %s exists=%s", GRIDMET_DAILY, GRIDMET_DAILY.exists())
    return ok


def rebuild_weekly(warnings: list[str]) -> bool:
    from streamflow.build_weekly import build_weekly_from_daily

    return _run_step("build weekly live table", build_weekly_from_daily, warnings)


def append_decision(decision: dict) -> None:
    row = {key: decision.get(key) for key in (
        "end_date",
        "source",
        "installed_through",
        "promote",
        "n",
        "n_weeks",
        "n_drought",
        "locked_recall",
        "persistence_recall",
        "recall_gap",
        "locked_false_alarm_rate",
        "flow_shift",
        "watch",
    )}
    row["source"] = row["source"] or "live"
    if row["end_date"] is None:
        return
    frame = pd.DataFrame([row])
    frame["end_date"] = pd.to_datetime(frame["end_date"])
    LIVE_DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LIVE_DECISIONS_PATH.exists():
        old = pd.read_parquet(LIVE_DECISIONS_PATH)
        frame = pd.concat([old, frame], ignore_index=True)
        frame = frame.sort_values("end_date").drop_duplicates("end_date", keep="last")
    atomic_parquet(frame, LIVE_DECISIONS_PATH)


def push_docs() -> None:
    if not (ROOT / ".git").exists():
        logger.info("no git repository; skip push")
        return
    subprocess.run(["git", "add", "docs"], cwd=ROOT, check=False)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", "docs"],
        cwd=ROOT,
    )
    if staged.returncode == 0:
        logger.info("docs unchanged; skip commit")
        return
    subprocess.run(
        ["git", "commit", "-m", "Update drought monitor page", "--", "docs"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(["git", "push"], cwd=ROOT, check=True)


def run_weekly(
    *,
    ingest: bool = True,
    score: bool = True,
    mail: bool = True,
    push: bool = False,
) -> dict:
    warnings: list[str] = []
    inputs_ready = True
    if ingest:
        inputs_ready = refresh_sources(warnings)
        if inputs_ready:
            inputs_ready = rebuild_weekly(warnings)
        else:
            warnings.append(
                "build weekly live table: skipped because a required ingest failed"
            )
    decision: dict
    if score and inputs_ready:
        try:
            decision = score_current_window()
        except Exception as exc:  # noqa: BLE001
            decision = {
                "kind": "failed",
                "summary": str(exc),
                "end_date": None,
                "installed_through": None,
                "promote": False,
            }
            warnings.append(str(exc))
        else:
            append_decision(decision)
    elif score:
        decision = {
            "kind": "failed",
            "summary": (
                "Required live inputs did not refresh. "
                "The job did not score or make a keep/retrain decision."
            ),
            "end_date": None,
            "installed_through": None,
            "promote": False,
        }
    else:
        decision = {
            "kind": "ran",
            "summary": "Published the existing gate history. Did not score a new window.",
            "end_date": None,
            "installed_through": None,
            "promote": False,
        }
    dashboard_failed = False
    try:
        write_dashboard()
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"dashboard: {exc}")
        dashboard_failed = True
    if push:
        if decision["kind"] == "failed":
            warnings.append("publish docs: skipped because the weekly job failed")
        elif dashboard_failed:
            warnings.append("publish docs: skipped because the dashboard update failed")
            decision["publish_failed"] = True
        else:
            try:
                push_docs()
            except Exception as exc:  # noqa: BLE001
                logger.warning("docs publish failed: %s", exc)
                warnings.append(f"publish docs: {exc}")
                decision["publish_failed"] = True
    decision["warnings"] = warnings
    subject, body = format_weekly_email(decision, warnings)
    logger.info("%s\n%s", subject, body)
    if mail:
        try:
            send_email(subject, body)
        except Exception as exc:  # noqa: BLE001
            logger.warning("email failed: %s", exc)
            warnings.append(f"email: {exc}")
    return decision


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Monday job: refresh live weeks, apply the locked gate, email."
    )
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Skip downloads and the weekly rebuild. Score the file already on disk.",
    )
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Rewrite docs/ from saved gate history. Do not download or score.",
    )
    parser.add_argument(
        "--no-email",
        action="store_true",
        help="Write the note to the log only.",
    )
    parser.add_argument(
        "--push-docs",
        action="store_true",
        help="Commit and push docs/ if this folder is a git repository.",
    )
    args = parser.parse_args(argv)
    if args.dashboard_only:
        dest = write_dashboard()
        print(dest)
        return
    decision = run_weekly(
        ingest=not args.no_ingest,
        score=True,
        mail=not args.no_email,
        push=args.push_docs,
    )
    print(decision.get("summary", decision.get("kind")))
    if decision.get("publish_failed") or decision.get("kind") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
