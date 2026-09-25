"""Saved trees and the keep/replace row for the model in use."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

import lightgbm as lgb

from streamflow.config import BOOTSTRAP_INSTALLED, MODELS_DIR, REGISTRY_PATH

logger = logging.getLogger(__name__)


def _json_ready(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def load_registry(path: Path = REGISTRY_PATH) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(state: dict, path: Path = REGISTRY_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {key: _json_ready(val) for key, val in state.items()}
    path.write_text(json.dumps(clean, indent=2) + "\n", encoding="utf-8")
    return path


def model_path_for(installed_through) -> Path:
    day = _json_ready(installed_through)
    return MODELS_DIR / f"drought_{day}.txt"


def save_model(model: lgb.Booster, installed_through, last_promote=None) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = model_path_for(installed_through)
    model.save_model(str(dest))
    state = {
        "installed_through": installed_through,
        "last_promote": last_promote or installed_through,
        "model_path": str(dest),
        "bootstrap_if_missing": str(BOOTSTRAP_INSTALLED),
    }
    save_registry(state)
    logger.info("saved model %s", dest)
    return state


def load_model(path: Path | None = None) -> lgb.Booster:
    state = load_registry()
    if path is None:
        if state is None:
            raise RuntimeError("No saved model yet. Run weekly-job once to grow the 2019 trees.")
        path = Path(state["model_path"])
    if not path.exists():
        raise RuntimeError(f"Missing model file {path}")
    return lgb.Booster(model_file=str(path))
