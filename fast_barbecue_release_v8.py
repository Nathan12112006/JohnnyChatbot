#!/usr/bin/env python3
"""Expanded barbecue dataset builder using a caching image proxy."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import imagehash
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import fast_barbecue_release as base
import fast_barbecue_release_v7  # noqa: F401 - installs high-yield bitmap query set


def proxy_download_one(item, index: int, folder: Path, max_side: int, min_side: int):
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    client = requests.Session()
    client.headers.update({"User-Agent": base.USER_AGENT})
    client.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=3, pool_maxsize=3))
    try:
        response = client.get(
            "https://wsrv.nl/",
            params={
                "url": item.image_url,
                "w": max_side,
                "output": "jpg",
                "q": 84,
                "we": "",
            },
            timeout=(15, 90),
            stream=True,
        )
        response.raise_for_status()
        data = bytearray()
        for chunk in response.iter_content(262144):
            if not chunk:
                continue
            data.extend(chunk)
            if len(data) > 18_000_000:
                raise ValueError("proxied image too large")
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


base.download_one = proxy_download_one

if __name__ == "__main__":
    raise SystemExit(base.main())
