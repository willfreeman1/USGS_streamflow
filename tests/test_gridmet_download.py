"""Current-year gridMET files grow; Monday must download them again."""

from datetime import date
from pathlib import Path

from streamflow import ingest_gridmet


class _Result:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


def test_download_year_skips_a_finished_year(monkeypatch, tmp_path):
    dest = tmp_path / "tmmx_2025.nc"
    dest.write_bytes(b"x" * 2_000_000)
    called: list[int] = []

    def fake_run(*args, **kwargs):
        called.append(1)
        return _Result(0)

    monkeypatch.setattr(ingest_gridmet.subprocess, "run", fake_run)
    path = ingest_gridmet.download_year(
        "tmmx", 2025, dest_dir=tmp_path, today=date(2026, 10, 13)
    )
    assert path == dest
    assert called == []


def test_download_year_refreshes_the_open_year(monkeypatch, tmp_path):
    dest = tmp_path / "tmmx_2026.nc"
    dest.write_bytes(b"old" * 400_000)

    def fake_run(cmd, check=False):
        part = Path(cmd[cmd.index("-o") + 1])
        part.write_bytes(b"new" * 400_000)
        return _Result(0)

    monkeypatch.setattr(ingest_gridmet.subprocess, "run", fake_run)
    path = ingest_gridmet.download_year(
        "tmmx", 2026, dest_dir=tmp_path, today=date(2026, 10, 13)
    )
    assert path == dest
    assert path.read_bytes().startswith(b"new")
