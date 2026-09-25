"""Average a WGS84 grid onto DroughtCast watersheds.

Watersheds nest, so each basin is averaged on its own cells.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import shapefile
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import Point, shape
from shapely.ops import transform as shp_transform
from shapely.strtree import STRtree
from shapely.validation import make_valid

from streamflow.config import RAW_DIR

logger = logging.getLogger(__name__)

BASIN_SHP = RAW_DIR / "rdews_gages" / "rdews_gages.shp"


def north_up_lat(lat: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return latitudes north-to-south and whether the source was flipped."""
    if float(lat[0]) < float(lat[-1]):
        return lat[::-1].copy(), True
    return lat, False


def basin_cell_index(
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    res: float,
    sta_ids: list[str],
    shp_path: Path = BASIN_SHP,
) -> tuple[list[str], list[np.ndarray]]:
    """Per-basin flat cell indices. `lat`/`lon` must already be north-up."""
    if not shp_path.exists():
        raise RuntimeError(f"Missing {shp_path}; run ingest-basins")
    wanted = set(sta_ids)
    transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    reader = shapefile.Reader(str(shp_path.with_suffix("")))
    field_names = [field[0] for field in reader.fields[1:]]
    sta_idx = field_names.index("StaID")
    west = float(lon[0]) - res / 2.0
    north = float(lat[0]) + res / 2.0
    transform = from_origin(west, north, res, res)
    out_shape = (int(lat.size), int(lon.size))

    ordered: list[str] = []
    cells: list[np.ndarray] = []
    empty: list[str] = []
    for rec, shp in zip(reader.iterRecords(), reader.iterShapes()):
        sta = str(rec[sta_idx])
        if sta not in wanted:
            continue
        geom = shp_transform(transformer.transform, shape(shp.__geo_interface__))
        mask = rasterize(
            [(geom, 1)],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        )
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            mask = rasterize(
                [(geom, 1)],
                out_shape=out_shape,
                transform=transform,
                fill=0,
                dtype=np.uint8,
                all_touched=True,
            )
            idx = np.flatnonzero(mask)
        if idx.size == 0:
            empty.append(sta)
            continue
        ordered.append(sta)
        cells.append(idx.astype(np.int32, copy=False))
        if len(ordered) % 400 == 0:
            logger.info("indexed %s / ~%s basins", len(ordered), len(wanted))
    if empty:
        logger.warning("%s basins missed the grid (first: %s)", len(empty), empty[:8])
    if not ordered:
        raise RuntimeError("No basin polygons matched DroughtCast StaIDs")
    logger.info(
        "indexed %s basins; median cells %s",
        len(ordered),
        int(np.median([c.size for c in cells])),
    )
    return ordered, cells


def load_droughtcast_polygons(
    sta_ids: list[str],
    shp_path: Path = BASIN_SHP,
) -> tuple[list[str], list]:
    """DroughtCast basin polygons in WGS84. Watersheds nest."""
    if not shp_path.exists():
        raise RuntimeError(f"Missing {shp_path}; run ingest-basins")
    wanted = set(sta_ids)
    transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    reader = shapefile.Reader(str(shp_path.with_suffix("")))
    field_names = [field[0] for field in reader.fields[1:]]
    sta_idx = field_names.index("StaID")
    ordered: list[str] = []
    geoms: list = []
    for rec, shp in zip(reader.iterRecords(), reader.iterShapes()):
        sta = str(rec[sta_idx])
        if sta not in wanted:
            continue
        geom = make_valid(
            shp_transform(transformer.transform, shape(shp.__geo_interface__))
        )
        if geom.is_empty:
            continue
        ordered.append(sta)
        geoms.append(geom)
        if len(ordered) % 800 == 0:
            logger.info("loaded %s / ~%s basin polygons", len(ordered), len(wanted))
    if not ordered:
        raise RuntimeError("No basin polygons matched DroughtCast StaIDs")
    logger.info("loaded %s DroughtCast polygons", len(ordered))
    return ordered, geoms


def points_in_basins(
    lons: list[float],
    lats: list[float],
    sta_ids: list[str],
    shp_path: Path = BASIN_SHP,
) -> list[list[str]]:
    """For each point, every DroughtCast basin that contains it (nested)."""
    ordered, geoms = load_droughtcast_polygons(sta_ids, shp_path=shp_path)
    tree = STRtree(geoms)
    members: list[list[str]] = []
    for lon, lat in zip(lons, lats):
        pt = Point(float(lon), float(lat))
        hits = tree.query(pt)
        inside = [ordered[i] for i in hits if geoms[i].covers(pt)]
        members.append(inside)
    logger.info(
        "assigned %s points; median basins per point %s",
        len(members),
        int(np.median([len(m) for m in members])) if members else 0,
    )
    return members


def basin_means(
    grid: np.ndarray,
    ordered: list[str],
    cells: list[np.ndarray],
) -> np.ndarray:
    """One mean per basin from a 2-d north-up grid."""
    flat = np.asarray(grid, dtype=np.float32).ravel()
    out = np.full(len(ordered), np.nan, dtype=np.float32)
    for i, idx in enumerate(cells):
        block = flat[idx]
        with np.errstate(all="ignore"):
            out[i] = np.nanmean(block)
    return out
