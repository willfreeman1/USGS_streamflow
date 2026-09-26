"""Static GitHub Pages report. Tiny files only — not the weekly parquet."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from streamflow.config import DOCS_DIR, DROUGHT_GATE_TRAIL_PATH, LIVE_DECISIONS_PATH
from streamflow.model import LOCKED_FLOW_SHIFT_WATCH as _WATCH
from streamflow.model import LOCKED_RECALL_GAP_MIN as _GAP

logger = logging.getLogger(__name__)


def load_trail(
    trail_path: Path = DROUGHT_GATE_TRAIL_PATH,
    live_path: Path = LIVE_DECISIONS_PATH,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    if trail_path.exists():
        pieces.append(pd.read_parquet(trail_path))
    if live_path.exists():
        pieces.append(pd.read_parquet(live_path))
    if not pieces:
        raise RuntimeError("No gate history to publish. Run score-drought-gate first.")
    trail = pd.concat(pieces, ignore_index=True)
    trail["end_date"] = pd.to_datetime(trail["end_date"])
    trail = trail.sort_values("end_date").drop_duplicates(
        ["end_date", "source"], keep="last"
    )
    return trail.reset_index(drop=True)


def latest_row(trail: pd.DataFrame) -> dict:
    scored = trail.loc[trail["source"].isin(("hist", "live"))]
    if scored.empty:
        scored = trail
    last = scored.iloc[-1]
    return {key: last[key] for key in last.index}


def write_dashboard(
    trail: pd.DataFrame | None = None,
    *,
    docs_dir: Path = DOCS_DIR,
) -> Path:
    if trail is None:
        trail = load_trail()
    last = latest_row(trail)
    scored = trail.loc[trail["source"].isin(("hist", "live"))].copy()
    docs_dir.mkdir(parents=True, exist_ok=True)
    data_dir = docs_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    export = scored[
        [
            "end_date",
            "source",
            "installed_through",
            "promote",
            "locked_recall",
            "persistence_recall",
            "recall_gap",
            "locked_false_alarm_rate",
            "flow_shift",
            "watch",
        ]
    ].copy()
    export["end_date"] = pd.to_datetime(export["end_date"]).dt.strftime("%Y-%m-%d")
    export["installed_through"] = (
        pd.to_datetime(export["installed_through"], errors="coerce")
        .dt.strftime("%Y-%m-%d")
        .fillna(export["installed_through"].astype(str).str[:10])
    )
    export.to_csv(data_dir / "trail.csv", index=False)

    latest = {
        "end_date": str(pd.Timestamp(last["end_date"]).date()),
        "installed_through": str(last["installed_through"])[:10],
        "promote": bool(last["promote"]),
        "locked_recall": float(last["locked_recall"]),
        "persistence_recall": float(last["persistence_recall"]),
        "recall_gap": float(last["recall_gap"]),
        "locked_false_alarm_rate": float(last["locked_false_alarm_rate"]),
        "flow_shift": float(last["flow_shift"]) if pd.notna(last["flow_shift"]) else None,
        "watch": bool(last["watch"]),
        "n": int(last["n"]) if last.get("n") is not None and pd.notna(last["n"]) else None,
        "n_drought": int(last["n_drought"])
        if last.get("n_drought") is not None and pd.notna(last["n_drought"])
        else None,
        "gap_line": _GAP,
        "shift_watch": _WATCH,
    }
    (data_dir / "latest.json").write_text(
        json.dumps(latest, indent=2) + "\n", encoding="utf-8"
    )
    html = _render_html(export, latest)
    dest = docs_dir / "index.html"
    dest.write_text(html, encoding="utf-8")
    write_svgs(export, docs_dir / "charts")
    logger.info("wrote %s and %s", dest, data_dir)
    return dest


def write_svgs(export: pd.DataFrame, charts_dir: Path) -> None:
    """PNG-free charts GitHub can show in the README."""
    charts_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(export["end_date"]).tolist()
    lead = (100 * export["recall_gap"].astype(float)).tolist()
    rec = (100 * export["locked_recall"].astype(float)).tolist()
    pers = (100 * export["persistence_recall"].astype(float)).tolist()
    far = (100 * export["locked_false_alarm_rate"].astype(float)).tolist()
    (charts_dir / "lead.svg").write_text(
        _line_svg(
            dates,
            [lead],
            labels=["Lead over “already dry”"],
            colors=["#1f4b82"],
            y_title="percentage points",
            hline=10,
            hline_label="Retrain line (10)",
        ),
        encoding="utf-8",
    )
    (charts_dir / "recall.svg").write_text(
        _line_svg(
            dates,
            [rec, pers],
            labels=["Model", "Already dry this week"],
            colors=["#1f4b82", "#888888"],
            y_title="percent of real drought weeks",
            y_min=0,
            y_max=100,
        ),
        encoding="utf-8",
    )
    (charts_dir / "false_positives.svg").write_text(
        _line_svg(
            dates,
            [far],
            labels=["Wrong drought calls on non-drought weeks"],
            colors=["#8a4b1f"],
            y_title="percent (false positives)",
            y_min=0,
        ),
        encoding="utf-8",
    )


def _line_svg(
    dates: list,
    series: list[list[float]],
    *,
    labels: list[str],
    colors: list[str],
    y_title: str,
    y_min: float | None = None,
    y_max: float | None = None,
    hline: float | None = None,
    hline_label: str = "",
    width: int = 860,
    height: int = 320,
) -> str:
    left, right, top, bottom = 58, 24, 28, 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    vals = [v for s in series for v in s if v == v]
    if hline is not None:
        vals.append(hline)
    lo = min(vals) if y_min is None else y_min
    hi = max(vals) if y_max is None else y_max
    if hi <= lo:
        hi = lo + 1
    pad = 0.06 * (hi - lo)
    if y_min is None:
        lo -= pad
    if y_max is None:
        hi += pad
    n = len(dates)
    stamps = [pd.Timestamp(d) for d in dates]
    t0 = stamps[0] if stamps else pd.Timestamp("2013-01-01")
    t1 = stamps[-1] if stamps else t0
    time_span = max((t1 - t0).total_seconds(), 1.0)

    def x_at(i: int) -> float:
        return left + plot_w * (stamps[i] - t0).total_seconds() / time_span

    def y_at(v: float) -> float:
        return top + plot_h * (1 - (v - lo) / (hi - lo))

    ticks = 4
    grid = []
    for i in range(ticks + 1):
        val = lo + (hi - lo) * i / ticks
        y = y_at(val)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            f'stroke="#ddd" /><text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#444">{val:.0f}</text>'
        )
    date_idx = sorted({0, n // 2, n - 1}) if n else []
    date_labels = []
    for i in date_idx:
        d = pd.Timestamp(dates[i]).strftime("%Y-%m-%d")
        date_labels.append(
            f'<text x="{x_at(i):.1f}" y="{height - 14}" text-anchor="middle" '
            f'font-size="11" fill="#444">{d}</text>'
        )
    paths = []
    gap_days = pd.Timedelta(days=21)
    for si, (ys, color, label) in enumerate(zip(series, colors, labels)):
        chunks: list[list[str]] = [[]]
        prev = None
        for i, v in enumerate(ys):
            if v != v:
                continue
            if prev is not None and stamps[i] - prev > gap_days:
                chunks.append([])
            chunks[-1].append(f"{x_at(i):.1f},{y_at(v):.1f}")
            prev = stamps[i]
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            pts = " ".join(chunk)
            paths.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="1.8" points="{pts}" />'
            )
        paths.append(
            f'<text x="{left + 8 + 200 * si}" y="18" font-size="12" fill="{color}">{label}</text>'
        )
    extra = ""
    if hline is not None:
        y = y_at(hline)
        extra = (
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            f'stroke="#a33" stroke-dasharray="6 4" />'
            f'<text x="{left + plot_w - 4}" y="{y - 6:.1f}" text-anchor="end" '
            f'font-size="11" fill="#a33">{hline_label}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        f'<rect width="{width}" height="{height}" fill="white"/>'
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" '
        f'fill="none" stroke="#bbb"/>'
        + "".join(grid)
        + extra
        + "".join(paths)
        + "".join(date_labels)
        + f'<text x="16" y="{top + plot_h / 2:.0f}" transform="rotate(-90 16,{top + plot_h / 2:.0f})" '
        f'font-size="11" fill="#444">{y_title}</text>'
        "</svg>\n"
    )


def _render_html(export: pd.DataFrame, latest: dict) -> str:
    points = json.loads(export.to_json(orient="records"))
    gap_pts = 100 * latest["recall_gap"]
    keep = not latest["promote"]
    status = "Keep the current model" if keep else "The model was retrained"
    watch = (
        "The mix of high and low flows is unlike 1980–1999. That is a reason to check "
        "how the live file is built, not to replace the model."
        if latest["watch"]
        else "The mix of high and low flows is close enough to 1980–1999 that we are not flagged to check inputs."
    )
    payload = json.dumps({"latest": latest, "points": points})
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Streamflow drought monitor</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <style>
    body {{ font-family: Georgia, "Times New Roman", serif; max-width: 52rem; margin: 2rem auto; padding: 0 1.2rem; color: #1a1a1a; line-height: 1.45; }}
    h1 {{ font-size: 1.7rem; margin-bottom: 0.3rem; }}
    .lede {{ color: #333; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr)); gap: 0.8rem; margin: 1.4rem 0; }}
    .card {{ border: 1px solid #ccc; padding: 0.8rem 0.9rem; }}
    .card strong {{ display: block; font-size: 1.25rem; }}
    .note {{ font-size: 0.95rem; color: #333; }}
    canvas {{ max-height: 22rem; margin: 1.2rem 0 2rem; }}
    footer {{ font-size: 0.9rem; color: #555; margin-top: 2rem; }}
  </style>
</head>
<body>
  <h1>Streamflow drought monitor</h1>
  <p class="lede">Four-week yes-or-no drought call for about 3,200 river stations in the lower 48. The interesting part is not beating the U.S. Geological Survey. It is a rule, written before later years were scored, that is allowed to say “keep this model” or “retrain.”</p>

  <div class="cards">
    <div class="card"><span>Latest 52-week window</span><strong>{latest["end_date"]}</strong></div>
    <div class="card"><span>Decision</span><strong>{"Keep" if keep else "Replace"}</strong><span class="note">{status}</span></div>
    <div class="card"><span>Model trained through</span><strong>{latest["installed_through"]}</strong></div>
    <div class="card"><span>Lead over “already dry”</span><strong>{gap_pts:.1f} points</strong><span class="note">Replace if this falls below 10</span></div>
    <div class="card"><span>Droughts caught</span><strong>{100 * latest["locked_recall"]:.0f}%</strong><span class="note">Already dry: {100 * latest["persistence_recall"]:.0f}%</span></div>
    <div class="card"><span>Ordinary weeks flagged</span><strong>{100 * latest["locked_false_alarm_rate"]:.0f}%</strong></div>
  </div>

  <p>{watch}</p>
  <p>Drought here means flow at or below the 10th percentile for that station and time of year, four weeks ahead. We call drought when the model’s 0–1 score is at least 0.70. That is a decision line, not “a 70% chance.” The simple check we have to beat is: if it is already dry this week, call drought four weeks out.</p>

  <h2>Lead over the simple check</h2>
  <p>Each point is the last 52 weeks we already know the answer for. The dashed line is 10 percentage points. Crossing below it is when we retrain, then wait 52 more known weeks.</p>
  <canvas id="gap"></canvas>

  <h2>Share of real droughts caught</h2>
  <canvas id="recall"></canvas>

  <h2>Share of ordinary weeks flagged</h2>
  <canvas id="far"></canvas>

  <footer>
    Locked numbers: call drought at 0.70; replace if the 52-week lead falls below 10 points; watch the flow mix at 0.20.
    The full story is the repository README.
    This page is a few kilobytes of numbers. The river and weather files stay on the desktop that runs the Monday job.
  </footer>
  <script>
    const DATA = {payload};
    const labels = DATA.points.map(p => p.end_date);
    const gap = DATA.points.map(p => 100 * p.recall_gap);
    const rec = DATA.points.map(p => 100 * p.locked_recall);
    const pers = DATA.points.map(p => 100 * p.persistence_recall);
    const far = DATA.points.map(p => 100 * p.locked_false_alarm_rate);
    const line = {{ borderWidth: 1.5, pointRadius: 0, tension: 0.15 }};
    new Chart(document.getElementById("gap"), {{
      type: "line",
      data: {{
        labels,
        datasets: [
          {{ label: "Lead (percentage points)", data: gap, borderColor: "#1f4b82", ...line }},
          {{ label: "Replace line (10)", data: labels.map(() => 10), borderColor: "#a33", borderDash: [6, 4], ...line }}
        ]
      }},
      options: {{ plugins: {{ legend: {{ position: "bottom" }} }}, scales: {{ y: {{ title: {{ display: true, text: "percentage points" }} }} }} }}
    }});
    new Chart(document.getElementById("recall"), {{
      type: "line",
      data: {{
        labels,
        datasets: [
          {{ label: "Model", data: rec, borderColor: "#1f4b82", ...line }},
          {{ label: "Already dry this week", data: pers, borderColor: "#888", ...line }}
        ]
      }},
      options: {{ plugins: {{ legend: {{ position: "bottom" }} }}, scales: {{ y: {{ min: 0, max: 100, title: {{ display: true, text: "percent of real drought weeks" }} }} }} }}
    }});
    new Chart(document.getElementById("far"), {{
      type: "line",
      data: {{
        labels,
        datasets: [
          {{ label: "Ordinary weeks flagged", data: far, borderColor: "#8a4b1f", ...line }}
        ]
      }},
      options: {{ plugins: {{ legend: {{ position: "bottom" }} }}, scales: {{ y: {{ min: 0, title: {{ display: true, text: "percent" }} }} }} }}
    }});
  </script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dest = write_dashboard()
    print(dest)


if __name__ == "__main__":
    main()
