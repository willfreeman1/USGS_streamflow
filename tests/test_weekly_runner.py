"""Prove the Monday PowerShell wrapper survives ordinary status messages."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "Run-WeeklyJob.ps1"

pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="The Monday wrapper is Windows PowerShell",
)

HELPER = """\
import logging
import sys

logging.basicConfig(level=logging.INFO)
logging.info("step usgs flow")
print("args=" + " ".join(sys.argv[1:]), flush=True)
raise SystemExit(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].lstrip("-").isdigit() else 0)
"""


def invoke_runner(
    *,
    job: Path | str,
    log_directory: Path,
    arguments: list[str] | None = None,
    working_directory: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    # Match Task Scheduler: powershell.exe -File Run-WeeklyJob.ps1 ...
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(RUNNER),
        "-Job",
        str(job),
        "-LogDirectory",
        str(log_directory),
    ]
    if working_directory is not None:
        command += ["-WorkingDirectory", str(working_directory)]
    if arguments is None:
        pass
    elif not arguments:
        command += ["-ArgumentList", ""]
    else:
        command += ["-ArgumentList", ",".join(arguments)]
    return subprocess.run(command, capture_output=True, text=True, check=False)


def latest_log(log_directory: Path) -> str:
    logs = sorted(log_directory.glob("weekly-*.log"))
    assert logs, f"wrapper wrote no log in {log_directory}"
    return logs[-1].read_text(encoding="utf-8", errors="replace")


def test_wrapper_survives_status_on_error_stream(tmp_path):
    helper = tmp_path / "status_job.py"
    helper.write_text(HELPER, encoding="utf-8")
    result = invoke_runner(
        job=sys.executable,
        arguments=[str(helper)],
        log_directory=tmp_path / "logs",
        working_directory=tmp_path,
    )
    text = latest_log(tmp_path / "logs")
    assert result.returncode == 0, text + result.stderr
    assert "step usgs flow" in text
    assert "Runner failed" not in text
    assert "Finished" in text
    assert "exit code 0" in text


def _cmd_helper(tmp_path: Path) -> Path:
    helper = tmp_path / "status_job.cmd"
    helper.write_text(
        "\r\n".join(
            [
                "@echo off",
                "echo step usgs flow 1>&2",
                "echo args=%*",
                "if \"%~1\"==\"\" exit /b 0",
                "echo %~1| findstr /r \"^[0-9][0-9]*$\" >nul",
                "if errorlevel 1 exit /b 0",
                "exit /b %~1",
                "",
            ]
        ),
        encoding="ascii",
    )
    return helper


def test_wrapper_default_passes_push_docs(tmp_path):
    result = invoke_runner(
        job=_cmd_helper(tmp_path),
        arguments=None,
        log_directory=tmp_path / "logs",
        working_directory=tmp_path,
    )
    text = latest_log(tmp_path / "logs")
    assert result.returncode == 0, text + result.stderr
    assert "--push-docs" in text
    assert "step usgs flow" in text


def test_wrapper_keeps_nonzero_exit_code(tmp_path):
    result = invoke_runner(
        job=_cmd_helper(tmp_path),
        arguments=["3"],
        log_directory=tmp_path / "logs",
        working_directory=tmp_path,
    )
    text = latest_log(tmp_path / "logs")
    assert result.returncode == 3, text + result.stderr
    assert "step usgs flow" in text
    assert "exit code 3" in text


def test_wrapper_leftover_lock_file_does_not_block(tmp_path):
    helper = tmp_path / "status_job.py"
    helper.write_text(HELPER, encoding="utf-8")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "weekly-job.lock").write_bytes(b"")
    result = invoke_runner(
        job=sys.executable,
        arguments=[str(helper)],
        log_directory=log_dir,
        working_directory=tmp_path,
    )
    text = latest_log(log_dir)
    assert result.returncode == 0, text + result.stderr


def test_wrapper_reports_busy_lock(tmp_path):
    helper = tmp_path / "status_job.py"
    helper.write_text(HELPER, encoding="utf-8")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    lock_path = log_dir / "weekly-job.lock"
    handle = lock_path.open("a+b")
    try:
        result = invoke_runner(
            job=sys.executable,
            arguments=[str(helper)],
            log_directory=log_dir,
            working_directory=tmp_path,
        )
    finally:
        handle.close()
    text = latest_log(log_dir)
    assert result.returncode == 75, text + result.stderr
    assert "already running" in text
