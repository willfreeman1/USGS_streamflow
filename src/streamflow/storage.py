"""Small atomic-write helpers for production state."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def atomic_parquet(
    frame: pd.DataFrame,
    path: Path,
    *,
    index: bool = False,
) -> Path:
    """Write beside the destination, then atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_parquet(temp, index=index)
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()
    return path
