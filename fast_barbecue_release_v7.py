#!/usr/bin/env python3
"""Rate-limited high-yield barbecue dataset release builder."""

from __future__ import annotations

import hashlib
import io
import threading
import time
from pathlib import Path

import imagehash
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import fast_barbecue_release as base
import fast_barbecue_release_v3 as combined

combined.QUERIES[:] = [
    "person grilling filetype:bitmap",
    "people grilling filetype:bitmap",
    "barbecue cooking filetype:bitmap",
    "barbecue cook filetype:bitmap",
    "outdoor grill cooking filetype:bitmap",
    "grilling hamburgers person filetype:bitmap",
    "grilling sausages person filetype:bitmap",
    "grilling steak person filetype:bitmap",
    "grilling chicken person filetype:bitmap",
    "grilling fish person filetype:bitmap",
    "grilling vegetables person filetype:bitmap",
    "meat grilling vendor filetype:bitmap",
    "food festival grilling filetype:bitmap",
    "park barbecue cooking filetype:bitmap",
    "camping grill cooking filetype:bitmap",
    "barbecue filetype:bitmap",
    "barbeque filetype:bitmap",
    "BBQ grilling filetype:bitmap",
    "grill cook filetype:bitmap",
    "braai filetype:bitmap",
    "churrasco filetype:bitmap",
    "asado parrilla filetype:bitmap",
    "mangal barbecue filetype:bitmap",
    "kebab grilling filetype:bitmap",
    "satay grilling filetype:bitmap",
]

_start_lock = threading.Lock()
_next_start = 0.0
_REQUEST_INTERVAL = 0.16


def pace_request() -> None:
    global _next_start
    with _start_lock:
        now = time.monotonic()
        if now < _next_start:
            time.sleep(_next_start - now)
        _next_start = time.monotonic() + _REQUEST_INTERVAL


def reliable_download_one(item, index: int, folder: Path, max_side: int, min_side: int):
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    client = requests.Session()
    client.headers.update({
        "User-Agent": base.USER_AGENT,
        "Referer": "https://commons.wikimedia.org/",
    })
    client.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=2, pool_maxsize=2))
    try:
        pace_request()
        response = client.get(item.image_url, timeout=(15, 75), stream=True)
        response.raise_for_status()
        data = bytearray()
        for chunk in response.iter_content(262144):
            if not chunk:
                continue
            data.extend(chunk)
            if len(data) > 25_000_000:
                raise ValueError("image too large")
        normalized = base.normalize(bytes(data), max_side, min_side)
        path = folder / f"{index:05d}.jpg"
        path.write_bytes(normalized)
        item.local_path = str(path)
        item.sha256 = hashlib.sha256(normalized).hexdigest()
        with Image.open(io.BytesIO(normalized)) as image:
            item.width, item.height = image.size
            item.phash = str(imagehash.phash(image, hash_size=8))
            item.dhash = str(imagehash.dhash(image, hash_size=8))
        return index, item, ""
    except Exception as exc:
        return index, None, f"{item.title}\t{type(exc).__name__}: {exc}"
    finally:
        client.close()


base.download_one = reliable_download_one

if __name__ == "__main__":
    raise SystemExit(base.main())
