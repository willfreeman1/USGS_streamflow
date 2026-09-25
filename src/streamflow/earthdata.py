"""Earthdata Login downloads. Auth only on the login host."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

import requests

from streamflow.config import earthdata_credentials

logger = logging.getLogger(__name__)

USER_AGENT = "usgs-streamflow-monitor (local)"
_MAX_HOPS = 12


def session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def download(url: str, dest: Path, sess: requests.Session | None = None) -> Path:
    """Follow redirects; send username/password only to urs.earthdata.nasa.gov."""
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    user, password = earthdata_credentials()
    s = sess or session()
    current = url
    for hop in range(_MAX_HOPS):
        host = urlparse(current).hostname or ""
        auth = (user, password) if "urs.earthdata.nasa.gov" in host else None
        r = s.get(current, auth=auth, allow_redirects=False, stream=True, timeout=180)
        if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("Location"):
            loc = r.headers["Location"]
            if loc.startswith("/"):
                loc = f"{urlparse(current).scheme}://{host}{loc}"
            r.close()
            current = loc
            continue
        if r.status_code != 200:
            r.close()
            raise RuntimeError(f"Earthdata HTTP {r.status_code} for {dest.name}")
        with tmp.open("wb") as handle:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    handle.write(chunk)
        r.close()
        if tmp.stat().st_size < 100_000:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Earthdata file too small: {dest.name}")
        tmp.replace(dest)
        return dest
    raise RuntimeError(f"too many Earthdata redirects for {dest.name}")
