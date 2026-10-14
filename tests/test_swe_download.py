"""Snow files: use daily (including early) when the yearly file is not out yet."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from streamflow import ingest_swe


def test_download_daily_accepts_early_file(monkeypatch, tmp_path):
    tried: list[str] = []

    def fake_curl(url: str, dest: Path, **kwargs) -> None:
        tried.append(url)
        if url.endswith("_early.nc"):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x" * 20_000)
            return
        raise RuntimeError(f"SWE download failed (22): {url}")

    monkeypatch.setattr(ingest_swe, "_curl", fake_curl)
    monkeypatch.setattr(ingest_swe, "SWE_DIR", tmp_path)
    path = ingest_swe.download_daily(date(2026, 10, 12))
    assert path is not None
    assert path.exists()
    assert any(url.endswith("20261012_early.nc") for url in tried)
    assert any("/WY2027/" in url for url in tried)
    assert any(url.endswith("20261012_stable.nc") for url in tried)


def test_build_uses_daily_when_yearly_file_is_missing(monkeypatch, tmp_path):
    hist = tmp_path / "hist.parquet"
    pd.DataFrame({"StaID": ["01234567"]}).to_parquet(hist)
    out = tmp_path / "swe.parquet"
    pd.DataFrame(
        {
            "StaID": ["01234567"],
            "date": [pd.Timestamp("2026-08-31")],
            "swe_mean": [1.0],
        }
    ).to_parquet(out)

    yearly: list[int] = []
    daily: list[date] = []

    def fake_yearly(wy: int) -> Path:
        yearly.append(wy)
        if wy == 2025:
            path = tmp_path / "wy2025.nc"
            path.write_bytes(b"x")
            return path
        raise RuntimeError(f"SWE download failed (22): WY{wy}")

    def fake_daily(day: date) -> Path | None:
        daily.append(day)
        return None

    monkeypatch.setattr(ingest_swe, "download_water_year", fake_yearly)
    monkeypatch.setattr(ingest_swe, "download_daily", fake_daily)
    monkeypatch.setattr(
        ingest_swe,
        "_index_from_nc",
        lambda *args, **kwargs: (["01234567"], [np.array([0])], False),
    )

    ingest_swe.build_basin_daily(
        date(2026, 9, 1),
        date(2026, 9, 3),
        hist_path=hist,
        out_path=out,
        today=date(2026, 10, 13),
    )
    assert 2026 in yearly
    assert daily == [date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3)]


def test_build_includes_october_when_window_starts_in_august(monkeypatch, tmp_path):
    hist = tmp_path / "hist.parquet"
    pd.DataFrame({"StaID": ["01234567"]}).to_parquet(hist)
    out = tmp_path / "swe.parquet"
    pd.DataFrame(
        {
            "StaID": ["01234567"],
            "date": [pd.Timestamp("2026-08-31")],
            "swe_mean": [1.0],
        }
    ).to_parquet(out)

    daily: list[date] = []

    def fake_yearly(wy: int) -> Path:
        if wy == 2025:
            path = tmp_path / "wy2025.nc"
            path.write_bytes(b"x")
            return path
        raise RuntimeError(f"SWE download failed (22): WY{wy}")

    def fake_daily(day: date) -> Path | None:
        daily.append(day)
        return None

    monkeypatch.setattr(ingest_swe, "download_water_year", fake_yearly)
    monkeypatch.setattr(ingest_swe, "download_daily", fake_daily)
    monkeypatch.setattr(
        ingest_swe,
        "_index_from_nc",
        lambda *args, **kwargs: (["01234567"], [np.array([0])], False),
    )

    ingest_swe.build_basin_daily(
        date(2026, 8, 21),
        date(2026, 10, 13),
        hist_path=hist,
        out_path=out,
        today=date(2026, 10, 13),
    )
    assert date(2026, 10, 1) in daily
    assert date(2026, 10, 12) in daily
