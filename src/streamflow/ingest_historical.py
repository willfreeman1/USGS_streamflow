"""Download USGS DroughtCast weekly model inputs (ScienceBase P1X5VH96).

The weekly table is 3,219 feather files in weekly_model_inputs.zip plus 10
more in added_08212025.zip. Do not unzip the big archive on a tight disk;
stack feathers from the zip later in build_weekly.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

import requests

from streamflow.config import RAW_DIR

logger = logging.getLogger(__name__)

ITEM_ID = "68191a56d4be0208bc3e02b9"
CATALOG_GET = f"https://www.sciencebase.gov/catalog/file/get/{ITEM_ID}"
USER_AGENT = "usgs-streamflow-monitor (local)"

OUT_DIR = RAW_DIR / "sciencebase_p1x5vh96"

# name= works. manager/download and ?f=__s3__ do not (HTML app / 404).
FILES = (
    "Drought_model_input_data_dictionary_05222025.csv",
    "Version_history_streamflow_drought_model_inputs.txt",
    "added_08212025.zip",
    "weekly_model_inputs.zip",
)


def catalog_url(name: str, item_id: str = ITEM_ID) -> str:
    return f"https://www.sciencebase.gov/catalog/file/get/{item_id}?name={name}"


def download_file(
    name: str,
    dest_dir: Path = OUT_DIR,
    *,
    item_id: str = ITEM_ID,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    url = catalog_url(name, item_id=item_id)
    logger.info("download %s -> %s", name, dest)
    _curl_resume(url, dest)
    if dest.stat().st_size < 1000:
        raise RuntimeError(f"{name} is too small; ScienceBase returned a stub")
    return dest


def _curl_resume(url: str, dest: Path) -> None:
    cmd = [
        "curl.exe",
        "-fL",
        "-C",
        "-",
        "-A",
        USER_AGENT,
        "--retry",
        "8",
        "--retry-delay",
        "5",
        "--retry-all-errors",
        "-o",
        str(dest),
        url,
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"curl failed ({result.returncode}) for {url}")


def download_via_requests(name: str, dest_dir: Path = OUT_DIR) -> Path:
    """Fallback if curl is missing. Does not resume."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    url = catalog_url(name)
    with requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in resp.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return dest


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Download ScienceBase weekly drought model inputs."
    )
    parser.add_argument(
        "--skip-weekly-zip",
        action="store_true",
        help="Skip the 8.22 GB weekly_model_inputs.zip",
    )
    args = parser.parse_args(argv)
    for name in FILES:
        if args.skip_weekly_zip and name == "weekly_model_inputs.zip":
            logger.info("skip %s", name)
            continue
        download_file(name)


if __name__ == "__main__":
    main()
