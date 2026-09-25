"""USGS Water Data API (OGC). Paginated GeoJSON feature collections."""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests

from streamflow.config import usgs_api_key

logger = logging.getLogger(__name__)

_MAX_LIMIT = 10000
_RETRIES = 8
_RETRY_STATUSES = {429, 500, 502, 503, 504}


class UsgsClient:
    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "X-Api-Key": usgs_api_key(),
                "User-Agent": "usgs-streamflow-monitor (local)",
                "Accept": "application/json",
            }
        )

    def iter_features(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        page_limit: int = _MAX_LIMIT,
    ) -> Iterator[dict[str, Any]]:
        params = dict(params or {})
        params.setdefault("limit", page_limit)
        next_url: str | None = url
        next_params: dict[str, Any] | None = params
        pages = 0
        while next_url:
            payload = self._get_json(next_url, next_params)
            pages += 1
            features = payload.get("features") or []
            logger.info(
                "USGS page %s: %s features (returned=%s)",
                pages,
                len(features),
                payload.get("numberReturned"),
            )
            yield from features
            next_link = _next_href(payload)
            if not next_link:
                break
            next_url = _strip_api_key(next_link)
            next_params = None

    def _get_json(
        self, url: str, params: dict[str, Any] | None
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(_RETRIES):
            try:
                resp = self.session.get(url, params=params, timeout=120)
                if resp.status_code in _RETRY_STATUSES:
                    wait = min(90, 2**attempt)
                    logger.warning(
                        "USGS %s on %s; retry in %ss",
                        resp.status_code,
                        _safe_url(resp.url),
                        wait,
                    )
                    time.sleep(wait)
                    last_error = requests.HTTPError(
                        f"{resp.status_code} for {_safe_url(resp.url)}"
                    )
                    continue
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                wait = min(90, 2**attempt)
                logger.warning(
                    "USGS request failed (%s); retry in %ss",
                    _safe_error(exc),
                    wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"USGS request failed after retries: {_safe_error(last_error)}"
        )


def _next_href(payload: dict[str, Any]) -> str | None:
    for link in payload.get("links") or []:
        if link.get("rel") == "next" and link.get("href"):
            return str(link["href"])
    return None


def _strip_api_key(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query.pop("api_key", None)
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def _safe_url(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "api_key" in query:
        query["api_key"] = ["REDACTED"]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def _safe_error(exc: Exception | None) -> str:
    if exc is None:
        return "unknown error"
    return _safe_url(str(exc))
