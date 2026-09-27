"""Monthly climate pattern series used in the USGS drought table.

ENSO here is Niño 3.4 sea-surface temperature in Celsius (about 24–29),
not the standardized ONI anomaly. The others are the usual monthly
indexes. One row per month; we repeat the month onto every week later.
"""

from __future__ import annotations

import argparse
import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from streamflow.config import RAW_DIR
from streamflow.storage import atomic_parquet

logger = logging.getLogger(__name__)

OUT_PATH = RAW_DIR / "climate_monthly.parquet"
USER_AGENT = "usgs-streamflow-monitor (local)"

URLS = {
    "enso": "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices",
    "pna": (
        "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/"
        "norm.pna.monthly.b5001.current.ascii"
    ),
    "pdo": "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat",
    "amo": "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.amo.dat",
    "sunspots": "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv",
}


def _get(url: str) -> str:
    r = requests.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.text


def _enso() -> pd.DataFrame:
    text = _get(URLS["enso"])
    df = pd.read_csv(StringIO(text), sep=r"\s+")
    df = df.rename(columns={"YR": "year", "MON": "month", "NINO3.4": "ENSO"})
    return df[["year", "month", "ENSO"]]


def _pna() -> pd.DataFrame:
    text = _get(URLS["pna"])
    df = pd.read_csv(
        StringIO(text),
        sep=r"\s+",
        header=None,
        names=["year", "month", "PNA"],
    )
    return df


def _pdo() -> pd.DataFrame:
    text = _get(URLS["pdo"])
    lines = [ln for ln in text.splitlines() if ln.strip() and ln.strip()[0].isdigit()]
    rows = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 13:
            continue
        year = int(float(parts[0]))
        for month, val in enumerate(parts[1:13], start=1):
            try:
                num = float(val)
            except ValueError:
                continue
            if abs(num) > 90:
                continue
            rows.append({"year": year, "month": month, "PDO": num})
    return pd.DataFrame(rows)


def _amo() -> pd.DataFrame:
    """ERSST v5 North Atlantic (0–60N) SST anomaly.

    PSL's Kaplan AMO file ends in January 2023. This file is current.
    It is not the same construction as the 1980–2020 USGS AMO column.
    """
    text = _get(URLS["amo"])
    rows = []
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) < 3 or not parts[0][0].isdigit():
            continue
        try:
            year = int(parts[0])
            month = int(parts[1])
            num = float(parts[2])
        except ValueError:
            continue
        if abs(num) > 90:
            continue
        rows.append({"year": year, "month": month, "AMO": num})
    return pd.DataFrame(rows)


def _sunspots() -> pd.DataFrame:
    text = _get(URLS["sunspots"])
    df = pd.read_csv(
        StringIO(text),
        sep=";",
        header=None,
        names=["year", "month", "frac", "sunspots", "sd", "n", "flag"],
    )
    df["sunspots"] = pd.to_numeric(df["sunspots"], errors="coerce")
    return df[["year", "month", "sunspots"]]


def build_monthly(out_path: Path = OUT_PATH) -> Path:
    pieces = [_enso(), _pna(), _pdo(), _amo(), _sunspots()]
    frame = pieces[0]
    for extra in pieces[1:]:
        frame = frame.merge(extra, on=["year", "month"], how="outer")
    frame = frame.sort_values(["year", "month"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_parquet(frame, out_path)
    last = frame.dropna(how="all", subset=["ENSO", "PNA", "PDO", "AMO", "sunspots"]).iloc[-1]
    logger.info(
        "wrote %s months %s-%02d to %s-%02d -> %s",
        len(frame),
        int(frame["year"].min()),
        int(frame.iloc[0]["month"]),
        int(last["year"]),
        int(last["month"]),
        out_path,
    )
    return out_path


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Download monthly climate pattern series."
    )
    parser.parse_args(argv)
    build_monthly()


if __name__ == "__main__":
    main()
