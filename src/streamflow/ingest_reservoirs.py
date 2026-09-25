"""Live reservoir storage onto DroughtCast basins.

Historical USGS columns came from ResOpsUS and stop in 2020. Live storage
is four public feeds: USGS lake storage (parameter 00054), Reclamation
RISE, Army Corps CWMS, and Texas Water Data. Each lake is a point. A
point can sit in many nested watersheds; we add that lake's acre-feet
to every basin that contains it.

The 1980–2020 table only attached a small local lake to Lees Ferry
(~100 acre-feet). Live totals include Lake Powell (millions). Same
column name, different join. Do not treat them as one series.
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from streamflow.config import RAW_DIR, USGS_DAILY_URL, USGS_TS_META_URL, WEEKLY_HIST_PATH
from streamflow.ingest_flow import _month_windows
from streamflow.usgs_client import UsgsClient
from streamflow.zonal import points_in_basins

logger = logging.getLogger(__name__)

OUT_SITES = RAW_DIR / "reservoir_sites.parquet"
OUT_DAILY = RAW_DIR / "reservoir_daily.parquet"
OUT_BASIN = RAW_DIR / "reservoir_basin_daily.parquet"
USGS_RAW = RAW_DIR / "usgs_reservoir_daily.parquet"
RISE_RAW = RAW_DIR / "rise_reservoir_daily.parquet"
USACE_RAW = RAW_DIR / "usace_reservoir_daily.parquet"
TWDB_RAW = RAW_DIR / "twdb_reservoir_daily.parquet"

STORAGE_PARAMETER = "00054"
STAT_RANK = {"00003": 0, "00011": 1, "32400": 2}
SOURCE_RANK = {"usgs": 0, "rise": 1, "usace": 2, "twdb": 3}
DEDUP_DEG = 0.02
M3_PER_AF = 1233.48183754752
USER_AGENT = "usgs-streamflow-monitor (local)"
RISE_ACCEPT = {"accept": "application/vnd.api+json", "User-Agent": USER_AGENT}
USACE_ACCEPT = {
    "accept": "application/json;version=2",
    "User-Agent": USER_AGENT,
}


def ingest_reservoirs(
    start: date,
    end: date,
    *,
    skip: set[str] | None = None,
) -> Path:
    skip = skip or set()
    if "usgs" not in skip:
        ingest_usgs(start, end)
    if "rise" not in skip:
        ingest_rise(start, end)
    if "usace" not in skip:
        ingest_usace(start, end)
    if "twdb" not in skip:
        try:
            ingest_twdb(start, end)
        except Exception as exc:  # noqa: BLE001
            logger.warning("TWDB failed; continuing with other sources: %s", exc)
    return assign_basins()


def ingest_usgs(start: date, end: date, *, client: UsgsClient | None = None) -> Path:
    """CONUS daily lake storage, parameter 00054, acre-feet."""
    client = client or UsgsClient()
    sites = _usgs_sites(client)
    logger.info("USGS 00054 sites with coordinates: %s", len(sites))
    frames: list[pd.DataFrame] = []
    for window_start, window_end in _month_windows(start, end):
        logger.info("USGS storage %s to %s", window_start, window_end)
        rows: list[dict] = []
        params = {
            "parameter_code": STORAGE_PARAMETER,
            "datetime": f"{window_start.isoformat()}/{window_end.isoformat()}",
            "bbox": "-125,24.5,-66,50",
        }
        for feat in client.iter_features(USGS_DAILY_URL, params):
            props = feat.get("properties") or {}
            site = props.get("monitoring_location_id")
            when = props.get("time")
            if site is None or when is None:
                continue
            rows.append(
                {
                    "source_id": site,
                    "date": when[:10],
                    "storage_af": _to_float(props.get("value")),
                    "statistic_id": props.get("statistic_id"),
                }
            )
        if not rows:
            logger.warning("no USGS storage rows %s to %s", window_start, window_end)
            continue
        frame = pd.DataFrame(rows)
        frame["stat_rank"] = frame["statistic_id"].map(
            lambda s: STAT_RANK.get(str(s), 9)
        )
        frame = (
            frame.sort_values(["source_id", "date", "stat_rank"])
            .drop_duplicates(["source_id", "date"], keep="first")
            .drop(columns=["statistic_id", "stat_rank"])
        )
        frames.append(frame)
    if not frames:
        raise RuntimeError("No USGS 00054 daily rows")
    daily = pd.concat(frames, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["source"] = "usgs"
    daily = daily.merge(sites, on="source_id", how="left")
    daily = (
        daily.sort_values(["source_id", "date"])
        .drop_duplicates(["source_id", "date"], keep="last")
        .reset_index(drop=True)
    )
    return _write_source(daily, USGS_RAW)


def _usgs_sites(client: UsgsClient) -> pd.DataFrame:
    rows: list[dict] = []
    for feat in client.iter_features(
        USGS_TS_META_URL, {"parameter_code": STORAGE_PARAMETER}
    ):
        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or [None, None]
        site = props.get("monitoring_location_id")
        if site is None or coords[0] is None:
            continue
        rows.append(
            {
                "source_id": site,
                "name": props.get("parameter_name") or site,
                "lon": float(coords[0]),
                "lat": float(coords[1]),
            }
        )
    sites = pd.DataFrame(rows).drop_duplicates("source_id")
    return sites


def ingest_rise(start: date, end: date) -> Path:
    """Reclamation RISE daily lake storage (parameter 3, acre-feet)."""
    sess = requests.Session()
    sess.headers.update(RISE_ACCEPT)
    items = _rise_storage_items(sess)
    logger.info("RISE storage catalog items: %s", len(items))
    loc_ids = sorted({it["location_id"] for it in items if it.get("location_id")})
    locations = _rise_locations(sess, loc_ids)
    logger.info("RISE locations with coordinates: %s", len(locations))

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {
            pool.submit(_rise_results, sess, it["item_id"], start, end): it
            for it in items
        }
        done = 0
        for fut in as_completed(futs):
            item = futs[fut]
            done += 1
            loc = locations.get(item.get("location_id") or "")
            if loc is None:
                continue
            try:
                values = fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("RISE item %s failed: %s", item["item_id"], exc)
                continue
            for when, value in values:
                rows.append(
                    {
                        "source_id": f"rise-{item['item_id']}",
                        "date": when,
                        "storage_af": value,
                        "name": loc["name"],
                        "lon": loc["lon"],
                        "lat": loc["lat"],
                    }
                )
            if done % 25 == 0 or done == len(items):
                logger.info("RISE series %s/%s", done, len(items))
    if not rows:
        raise RuntimeError("No RISE storage rows")
    daily = pd.DataFrame(rows)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["source"] = "rise"
    return _write_source(daily, RISE_RAW)


def _rise_storage_items(sess: requests.Session) -> list[dict]:
    items: list[dict] = []
    records: dict[str, str] = {}
    page = 1
    while True:
        payload = _rise_get(
            sess,
            "https://data.usbr.gov/rise/api/catalog-item",
            {"itemsPerPage": 100, "page": page},
        )
        batch = payload.get("data") or []
        if not batch:
            break
        for raw in batch:
            attrs = raw.get("attributes") or {}
            if attrs.get("parameterId") != 3:
                continue
            title = str(attrs.get("itemTitle") or "")
            if "Delta" in title:
                continue
            rel = ((raw.get("relationships") or {}).get("catalogRecord") or {}).get(
                "data"
            ) or {}
            rec_id = str(rel.get("id") or "").rsplit("/", 1)[-1]
            item_id = str(attrs.get("_id") or raw.get("id", "")).split("/")[-1]
            items.append({"item_id": item_id, "record_id": rec_id, "title": title})
            if rec_id:
                records[rec_id] = rec_id
        meta = payload.get("meta") or {}
        if page >= int(meta.get("totalItems", 0) / 100) + 1:
            break
        page += 1
    loc_by_record = {}
    for rec_id in records:
        try:
            rec = _rise_get(
                sess, f"https://data.usbr.gov/rise/api/catalog-record/{rec_id}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("RISE record %s: %s", rec_id, exc)
            continue
        loc = ((rec.get("data") or {}).get("relationships") or {}).get("location") or {}
        loc_id = str((loc.get("data") or {}).get("id") or "").rsplit("/", 1)[-1]
        loc_by_record[rec_id] = loc_id
    for item in items:
        item["location_id"] = loc_by_record.get(item["record_id"])
    return items


def _rise_locations(sess: requests.Session, loc_ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for loc_id in loc_ids:
        if not loc_id:
            continue
        try:
            payload = _rise_get(
                sess, f"https://data.usbr.gov/rise/api/location/{loc_id}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("RISE location %s: %s", loc_id, exc)
            continue
        attrs = (payload.get("data") or {}).get("attributes") or {}
        coords = (attrs.get("locationCoordinates") or {}).get("coordinates") or []
        if len(coords) < 2:
            continue
        out[str(loc_id)] = {
            "name": attrs.get("locationName") or loc_id,
            "lon": float(coords[0]),
            "lat": float(coords[1]),
        }
    return out


def _rise_results(
    sess: requests.Session, item_id: str, start: date, end: date
) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    page = 1
    while True:
        payload = _rise_get(
            sess,
            "https://data.usbr.gov/rise/api/result",
            {
                "itemId": item_id,
                "dateTime[after]": start.isoformat(),
                "dateTime[before]": (end + timedelta(days=1)).isoformat(),
                "itemsPerPage": 500,
                "page": page,
            },
        )
        batch = payload.get("data") or []
        if not batch:
            break
        for raw in batch:
            attrs = raw.get("attributes") or {}
            when = str(attrs.get("dateTime") or "")[:10]
            value = _to_float(attrs.get("result"))
            if when and value is not None:
                rows.append((when, value))
        meta = payload.get("meta") or {}
        total = int(meta.get("totalItems") or 0)
        if page * 500 >= total or not batch:
            break
        page += 1
    return rows


def _rise_get(
    sess: requests.Session, url: str, params: dict | None = None
) -> dict:
    last: Exception | None = None
    for attempt in range(6):
        try:
            resp = sess.get(url, params=params, timeout=90)
            if resp.status_code in {429, 500, 502, 503, 504}:
                time.sleep(min(30, 2**attempt))
                last = RuntimeError(f"RISE {resp.status_code}")
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last = exc
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"RISE failed {url}: {last}")


def ingest_usace(start: date, end: date) -> Path:
    """Army Corps CWMS reservoir storage. Prefer daily Stor; convert m3."""
    sess = requests.Session()
    sess.headers.update(USACE_ACCEPT)
    offices = _usace_districts(sess)
    logger.info("USACE districts: %s", len(offices))
    series: list[dict] = []
    coords: dict[tuple[str, str], tuple[float, float]] = {}
    for office in offices:
        found = _usace_stor_series(sess, office)
        if not found:
            continue
        series.extend(found)
        for key, xy in _usace_locations(sess, office, {s["loc"] for s in found}).items():
            coords[key] = xy
        logger.info("USACE %s: %s Stor series", office, len(found))
    logger.info("USACE Stor series total %s", len(series))

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {
            pool.submit(_usace_values, sess, item, start, end): item for item in series
        }
        done = 0
        for fut in as_completed(futs):
            item = futs[fut]
            done += 1
            xy = coords.get((item["office"], item["loc"]))
            if xy is None:
                continue
            try:
                values = fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("USACE %s %s: %s", item["office"], item["name"], exc)
                continue
            for when, value in values:
                rows.append(
                    {
                        "source_id": f"usace-{item['office']}-{item['loc']}",
                        "date": when,
                        "storage_af": value,
                        "name": item["loc"],
                        "lon": xy[0],
                        "lat": xy[1],
                    }
                )
            if done % 40 == 0 or done == len(series):
                logger.info("USACE values %s/%s", done, len(series))
    if not rows:
        raise RuntimeError("No USACE storage rows")
    daily = pd.DataFrame(rows)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["source"] = "usace"
    return _write_source(daily, USACE_RAW)


def _usace_districts(sess: requests.Session) -> list[str]:
    resp = sess.get("https://cwms-data.usace.army.mil/cwms-data/offices", timeout=90)
    resp.raise_for_status()
    return [o["name"] for o in resp.json() if o.get("type") == "DIS"]


def _usace_stor_series(sess: requests.Session, office: str) -> list[dict]:
    picked: dict[str, dict] = {}
    page = None
    for _ in range(40):
        params = {"office": office, "like": "*.Stor.Inst.*", "page-size": 500}
        if page:
            params["page"] = page
        try:
            resp = sess.get(
                "https://cwms-data.usace.army.mil/cwms-data/catalog/TIMESERIES",
                params=params,
                timeout=90,
            )
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("USACE catalog %s: %s", office, exc)
            return list(picked.values())
        for entry in payload.get("entries") or []:
            name = str(entry.get("name") or "")
            parts = name.split(".")
            if len(parts) < 6:
                continue
            loc, param, _kind, interval = parts[0], parts[1], parts[2], parts[3]
            version = parts[-1]
            if param != "Stor":
                continue
            rank = (
                (0 if interval == "1Day" else 1 if interval == "1Hour" else 2),
                (0 if "Rev" in version or "REV" in version else 1),
            )
            cur = picked.get(loc)
            if cur is None or rank < cur["rank"]:
                picked[loc] = {
                    "office": office,
                    "loc": loc,
                    "name": name,
                    "interval": interval,
                    "rank": rank,
                }
        page = payload.get("next-page")
        if not page:
            break
    return list(picked.values())


def _usace_locations(
    sess: requests.Session, office: str, names: set[str]
) -> dict[tuple[str, str], tuple[float, float]]:
    try:
        resp = sess.get(
            "https://cwms-data.usace.army.mil/cwms-data/locations",
            params={"office": office, "page-size": 5000},
            timeout=120,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("USACE locations %s: %s", office, exc)
        return {}
    locs = payload if isinstance(payload, list) else payload.get("locations") or []
    if isinstance(locs, dict):
        locs = locs.get("location") or []
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for loc in locs:
        name = loc.get("name")
        lon = loc.get("longitude")
        lat = loc.get("latitude")
        if name in names and lon is not None and lat is not None:
            out[(office, str(name))] = (float(lon), float(lat))
    return out


def _usace_values(
    sess: requests.Session, item: dict, start: date, end: date
) -> list[tuple[str, float]]:
    resp = sess.get(
        "https://cwms-data.usace.army.mil/cwms-data/timeseries",
        params={
            "name": item["name"],
            "office": item["office"],
            "begin": f"{start.isoformat()}T00:00:00Z",
            "end": f"{end.isoformat()}T23:59:59Z",
            "unit": "ac-ft",
            "page-size": 20000,
        },
        timeout=120,
    )
    if resp.status_code >= 400:
        resp = sess.get(
            "https://cwms-data.usace.army.mil/cwms-data/timeseries",
            params={
                "name": item["name"],
                "office": item["office"],
                "begin": f"{start.isoformat()}T00:00:00Z",
                "end": f"{end.isoformat()}T23:59:59Z",
                "page-size": 20000,
            },
            timeout=120,
        )
        resp.raise_for_status()
        payload = resp.json() or {}
        values = payload.get("values") or []
        unit = str(payload.get("units") or "").lower()
        scale = 1.0 / M3_PER_AF if unit in {"m3", "cms", "m³"} else 1.0
    else:
        payload = resp.json() or {}
        values = payload.get("values") or []
        scale = 1.0
    by_day: dict[str, list[float]] = {}
    for row in values:
        if not row or row[1] is None:
            continue
        when = datetime.fromtimestamp(row[0] / 1000.0, tz=timezone.utc).date()
        by_day.setdefault(when.isoformat(), []).append(float(row[1]) * scale)
    return [(day, float(pd.Series(vals).median())) for day, vals in by_day.items()]


def ingest_twdb(start: date, end: date) -> Path:
    """Texas Water Data individual-reservoir daily storage."""
    sess = requests.Session()
    sess.headers["User-Agent"] = USER_AGENT
    resp = sess.get(
        "https://www.waterdatafortexas.org/reservoirs/recent-conditions.json",
        timeout=90,
    )
    resp.raise_for_status()
    catalog = resp.json()
    slugs = _twdb_slugs(sess)
    lakes: list[dict] = []
    for key, info in catalog.items():
        loc = (info.get("gauge_location") or {}).get("coordinates") or []
        if len(loc) < 2:
            continue
        raw = info.get("condensed_name") or key
        lakes.append(
            {
                "name": slugs.get(_twdb_key(raw), _twdb_slug(raw)),
                "full": info.get("full_name") or key,
                "lon": float(loc[0]),
                "lat": float(loc[1]),
            }
        )
    logger.info("TWDB reservoirs: %s", len(lakes))
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {
            pool.submit(_twdb_csv, sess, lake["name"], start, end): lake
            for lake in lakes
        }
        done = 0
        for fut in as_completed(futs):
            lake = futs[fut]
            done += 1
            try:
                values = fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("TWDB %s: %s", lake["name"], exc)
                continue
            for when, value in values:
                rows.append(
                    {
                        "source_id": f"twdb-{lake['name']}",
                        "date": when,
                        "storage_af": value,
                        "name": lake["full"],
                        "lon": lake["lon"],
                        "lat": lake["lat"],
                    }
                )
            if done % 25 == 0 or done == len(lakes):
                logger.info("TWDB csv %s/%s", done, len(lakes))
    if not rows:
        raise RuntimeError("No TWDB storage rows")
    daily = pd.DataFrame(rows)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["source"] = "twdb"
    return _write_source(daily, TWDB_RAW)


def _twdb_key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _twdb_slug(name: str) -> str:
    """Water Data for Texas CSVs are lowercase slugs, e.g. travis.csv."""
    return _twdb_key(name)


def _twdb_slugs(sess: requests.Session) -> dict[str, str]:
    """Map 'alanhenry' -> 'alan-henry' from the statewide page."""
    import re

    resp = sess.get(
        "https://www.waterdatafortexas.org/reservoirs/statewide", timeout=90
    )
    resp.raise_for_status()
    found = set(re.findall(r"/reservoirs/individual/([a-z0-9-]+)", resp.text))
    return {_twdb_key(slug): slug for slug in found}


def _twdb_csv(
    sess: requests.Session, name: str, start: date, end: date
) -> list[tuple[str, float]]:
    resp = sess.get(
        f"https://www.waterdatafortexas.org/reservoirs/individual/{name}.csv",
        timeout=90,
    )
    resp.raise_for_status()
    lines = [ln for ln in resp.text.splitlines() if ln and not ln.startswith("#")]
    if len(lines) < 2:
        return []
    header = [h.strip() for h in lines[0].split(",")]
    try:
        date_i = header.index("date")
        store_i = header.index("reservoir_storage")
    except ValueError:
        return []
    out: list[tuple[str, float]] = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) <= max(date_i, store_i):
            continue
        when = parts[date_i][:10]
        if when < start.isoformat() or when > end.isoformat():
            continue
        value = _to_float(parts[store_i])
        if value is not None:
            out.append((when, value))
    return out


def assign_basins(
    *,
    hist_path: Path = WEEKLY_HIST_PATH,
    out_sites: Path = OUT_SITES,
    out_daily: Path = OUT_DAILY,
    out_basin: Path = OUT_BASIN,
) -> Path:
    pieces = []
    for path in (USGS_RAW, RISE_RAW, USACE_RAW, TWDB_RAW):
        if path.exists():
            pieces.append(pd.read_parquet(path))
            logger.info("loaded %s (%s rows)", path.name, len(pieces[-1]))
    if not pieces:
        raise RuntimeError("No reservoir source files; run the source pulls first")
    daily = pd.concat(pieces, ignore_index=True)
    daily = daily.dropna(subset=["lon", "lat", "storage_af"])
    sites = (
        daily.groupby(["source", "source_id"], sort=False)
        .agg(name=("name", "first"), lon=("lon", "first"), lat=("lat", "first"))
        .reset_index()
    )
    sites = _dedup_sites(sites)
    keep = set(zip(sites["source"], sites["source_id"]))
    daily = daily.loc[
        [pair in keep for pair in zip(daily["source"], daily["source_id"])]
    ].copy()
    logger.info(
        "after dedup: %s lakes, %s daily rows",
        len(sites),
        len(daily),
    )

    hist = pd.read_parquet(hist_path, columns=["StaID"])
    sta_ids = sorted(hist["StaID"].astype(str).unique())
    members = points_in_basins(
        sites["lon"].tolist(), sites["lat"].tolist(), sta_ids
    )
    sites = sites.copy()
    sites["n_basins"] = [len(m) for m in members]
    out_sites.parent.mkdir(parents=True, exist_ok=True)
    sites.to_parquet(out_sites, index=False)

    lookup = {
        (src, sid): basins
        for src, sid, basins in zip(sites["source"], sites["source_id"], members)
    }
    exploded: list[pd.DataFrame] = []
    for (source, source_id), grp in daily.groupby(["source", "source_id"], sort=False):
        basins = lookup.get((source, source_id)) or []
        if not basins:
            continue
        block = grp[["date", "storage_af"]].copy()
        block["StaID"] = [basins] * len(block)
        exploded.append(block.explode("StaID", ignore_index=True))
    if not exploded:
        raise RuntimeError("No reservoirs fell inside a DroughtCast basin")
    long = pd.concat(exploded, ignore_index=True)
    basin = (
        long.groupby(["StaID", "date"], sort=False)
        .agg(
            reservoirs_storage_total=("storage_af", "sum"),
            reservoirs_n_total=("storage_af", "count"),
        )
        .reset_index()
    )
    daily.to_parquet(out_daily, index=False)
    basin.to_parquet(out_basin, index=False)
    logger.info(
        "wrote %s rows (%s gages, %s lakes) %s to %s -> %s (%.1f MB)",
        len(basin),
        basin["StaID"].nunique(),
        int((sites["n_basins"] > 0).sum()),
        basin["date"].min().date(),
        basin["date"].max().date(),
        out_basin,
        out_basin.stat().st_size / 1e6,
    )
    return out_basin


def _dedup_sites(sites: pd.DataFrame) -> pd.DataFrame:
    """Keep one lake per ~2 km cell. USGS, then RISE, then Corps, then Texas."""
    frame = sites.copy()
    frame["cell"] = (
        (frame["lat"] / DEDUP_DEG).round().astype(int).astype(str)
        + ":"
        + (frame["lon"] / DEDUP_DEG).round().astype(int).astype(str)
    )
    frame["rank"] = frame["source"].map(SOURCE_RANK).fillna(9)
    frame = (
        frame.sort_values(["cell", "rank"])
        .drop_duplicates("cell", keep="first")
        .drop(columns=["cell", "rank"])
        .reset_index(drop=True)
    )
    return frame


def _write_source(daily: pd.DataFrame, path: Path) -> Path:
    cols = ["source", "source_id", "name", "lon", "lat", "date", "storage_af"]
    daily = daily[cols].dropna(subset=["storage_af"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        daily = (
            pd.concat([existing, daily], ignore_index=True)
            .sort_values(["source_id", "date"])
            .drop_duplicates(["source_id", "date"], keep="last")
        )
    daily.to_parquet(path, index=False)
    logger.info(
        "wrote %s rows (%s lakes) %s to %s -> %s (%.1f MB)",
        len(daily),
        daily["source_id"].nunique(),
        daily["date"].min().date(),
        daily["date"].max().date(),
        path,
        path.stat().st_size / 1e6,
    )
    return path


def _to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_date(text: str) -> date:
    return date.fromisoformat(text)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    today = datetime.now(timezone.utc).date()
    parser = argparse.ArgumentParser(
        description="Ingest live reservoir storage and assign to gage basins."
    )
    parser.add_argument("--start", type=_parse_date, default=date(2024, 1, 1))
    parser.add_argument("--end", type=_parse_date, default=today)
    parser.add_argument("--skip-usgs", action="store_true")
    parser.add_argument("--skip-rise", action="store_true")
    parser.add_argument("--skip-usace", action="store_true")
    parser.add_argument("--skip-twdb", action="store_true")
    parser.add_argument(
        "--assign-only",
        action="store_true",
        help="Rebuild basin totals from source parquets already on disk.",
    )
    args = parser.parse_args(argv)
    if args.assign_only:
        assign_basins()
        return
    skip = {
        name
        for name, flag in (
            ("usgs", args.skip_usgs),
            ("rise", args.skip_rise),
            ("usace", args.skip_usace),
            ("twdb", args.skip_twdb),
        )
        if flag
    }
    ingest_reservoirs(args.start, args.end, skip=skip)


if __name__ == "__main__":
    main()
