"""Download USGS RDEWS gage-basin polygons (Staub / Wieczorek, P98IG8LO).

Every DroughtCast StaID is in rdews_gages (9,097 basins). Needed to average
gridMET / NLDAS / SMAP onto each watershed. Geometry file is ~1.5 GB.
"""

from __future__ import annotations

import argparse
import logging

from streamflow.config import RAW_DIR
from streamflow.ingest_historical import download_file

logger = logging.getLogger(__name__)

ITEM_ID = "64510406d34eefd5da80a2c7"
OUT_DIR = RAW_DIR / "rdews_gages"
FILES = (
    "rdews_gages.prj",
    "rdews_gages.shx",
    "rdews_gages.dbf",
    "rdews_gages.cpg",
    "rdews_gages.shp",
)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Download CONUS USGS gage-basin polygons."
    )
    parser.add_argument(
        "--skip-shp",
        action="store_true",
        help="Skip the 1.5 GB .shp (attributes only).",
    )
    args = parser.parse_args(argv)
    for name in FILES:
        if args.skip_shp and name.endswith(".shp"):
            logger.info("skip %s", name)
            continue
        download_file(name, dest_dir=OUT_DIR, item_id=ITEM_ID)


if __name__ == "__main__":
    main()
