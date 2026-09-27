"""Fail when the committed monitor has not advanced recently."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


def dashboard_age(path: Path, *, today: date | None = None) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    end_date = date.fromisoformat(payload["end_date"])
    return ((today or date.today()) - end_date).days


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("docs/data/latest.json"),
    )
    parser.add_argument("--max-age-days", type=int, default=21)
    args = parser.parse_args()
    age = dashboard_age(args.path)
    print(f"{args.path} is {age} days behind today")
    if age < 0 or age > args.max_age_days:
        raise SystemExit(
            f"Dashboard is stale: expected age 0–{args.max_age_days} days, got {age}"
        )


if __name__ == "__main__":
    main()
