#!/usr/bin/env python3
"""Fast, reproducible Wikimedia Commons collector for active-barbecuing images.

The output is a machine-filtered positive-image dataset. It uses Wikimedia
Commons metadata, CLIP relevance scoring, and exact/perceptual deduplication.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import random
import re
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageStat
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from urllib3.util.retry import Retry

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "ActiveBarbecuingDatasetBuilder/3.0 "
    "(research image collection; contact: GitHub user Nathan12112006)"
)

SEARCH_QUERIES = [
    "barbecue people cooking",
    "barbeque people cooking",
    "people actively grilling",
    "person grilling barbecue",
    "barbecue cook",
    "barbecue cooking people",
    "people grilling meat",
    "cook using barbecue grill",
    "pitmaster barbecue",
    "barbecue competition cooking",
    "barbecue festival cook",
    "backyard barbecue grilling",
    "park barbecue people",
    "beach barbecue cooking",
    "tailgate grilling people",
    "cookout grill people",
    "braai cooking people",
    "churrasco grilling people",
    "asado parrilla people",
    "parrillero cooking",
    "mangal cooking people",
    "shashlik grilling",
    "kebab grilling vendor",
    "satay grilling vendor",
    "yakitori grilling cook",
    "Grillmeister grillt",
    "churrasqueiro churrasco",
    "barbecue personne cuisine",
    "バーベキュー 人 焼く",
    "烧烤 烤肉 人",
]

GRILL_TERMS = {
    "barbecue", "barbeque", "barbecuing", "barbequing", "bbq", "grill",
    "grilling", "braai", "churrasco", "churrasqueira", "asado", "parrilla",
    "parrillero", "mangal", "shashlik", "kebab", "satay", "yakitori",
}
NEGATIVE_METADATA_TERMS = {
    "logo", "diagram", "illustration", "drawing", "poster", "advertisement",
    "icon", "vector", "render", "map", "menu", "recipe card", "product photo",
    "car grille", "radiator grille", "locomotive", "motorcycle", "album cover",
    "clip art", "coat of arms", "floor plan", "screenshot",
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}

POSITIVE_PROMPTS = [
    "a real photograph of a person actively cooking food on a barbecue grill",
    "a real photograph of people actively grilling meat at a barbecue",
    "a real photograph of someone using tongs over an outdoor grill",
    "a real photograph of a pitmaster tending barbecue meat in a smoker",
    "a real photograph of a cook grilling food over charcoal",
    "a real photograph of a street food vendor actively grilling skewers",
]
NEGATIVE_PROMPTS = [
    "a photograph of an empty barbecue grill with nobody cooking",
    "a product photograph of a barbecue grill",
    "a close-up photograph of grilled food with no person visible",
    "a photograph of people eating at a picnic without cooking",
    "a photograph of a campfire without a barbecue grill",
    "a photograph of a car front grille",
    "a landscape photograph unrelated to cooking",
    "a logo illustration poster diagram or menu",
    "a photograph of a kitchen appliance with nobody cooking",
]


@dataclass
class Candidate:
    pageid: int
    title: str
    query: str
    query_rank: int
    matched_queries: list[str] = field(default_factory=list)
    description: str = ""
    categories: str = ""
    creator: str = ""
    creator_url: str = ""
    attribution: str = ""
    license_short: str = ""
    license_url: str = ""
    source_page: str = ""
    image_url: str = ""
    source_sha1: str = ""
    width: int = 0
    height: int = 0
    metadata_score: float = 0.0
    local_path: str = ""
    normalized_sha256: str = ""
    phash: str = ""
    dhash: str = ""
    clip_positive: float = 0.0
    clip_negative: float = 0.0
    clip_margin: float = 0.0
    final_score: float = 0.0
    selected_filename: str = ""
    split: str = ""


def clean(value: Any, limit: int = 6000) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def metadata_value(metadata: dict[str, Any], key: str, limit: int = 6000) -> str:
    value = metadata.get(key, "")
    if isinstance(value, dict):
        value = value.get("value", "")
    return clean(value, limit)


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def make_session() -> requests.Session:
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset({"GET", "POST"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class Throttle:
    def __init__(self, interval: float) -> None:
        self.interval = max(0.0, interval)
        self.last = 0.0

    def wait(self) -> None:
        delta = time.monotonic() - self.last
        if delta < self.interval:
            time.sleep(self.interval - delta)
        self.last = time.monotonic()


def commons_request(
    session: requests.Session,
    throttle: Throttle,
    params: dict[str, Any],
    *,
    attempts: int = 8,
) -> dict[str, Any]:
    payload = dict(params)
    payload.setdefault("format", "json")
    payload.setdefault("formatversion", 2)
    payload.setdefault("maxlag", 5)
    last_error: Exception | None = None

    for attempt in range(attempts):
        throttle.wait()
        try:
            response = session.post(COMMONS_API, data=payload, timeout=(25, 150))
            if response.status_code in {429, 500, 502, 503, 504}:
                retry_after = safe_int(response.headers.get("Retry-After"))
                delay = max(retry_after, min(90, 2 ** (attempt + 1)))
                print(f"Commons HTTP {response.status_code}; retrying after {delay}s", file=sys.stderr)
                time.sleep(delay)
                continue
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                code = str(data["error"].get("code", ""))
                if code in {"maxlag", "ratelimited", "readonly"}:
                    delay = min(90, 2 ** (attempt + 1))
                    print(f"Commons API {code}; retrying after {delay}s", file=sys.stderr)
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Commons API error: {data['error']}")
            return data
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            delay = min(90, 2 ** (attempt + 1))
            print(f"Commons request failed ({exc}); retrying after {delay}s", file=sys.stderr)
            time.sleep(delay)

    raise RuntimeError(f"Commons request failed after {attempts} attempts: {last_error}")


def license_allowed(short_name: str, usage_terms: str) -> bool:
    value = re.sub(r"[_-]+", " ", f"{short_name} {usage_terms}".lower())
    value = re.sub(r"\s+", " ", value)
    if any(term in value for term in ("noncommercial", "no derivatives", " cc nc", " cc nd")):
        return False
    return any(
        term in value
        for term in (
            "cc0",
            "creative commons zero",
            "public domain",
            "no known restrictions",
            "cc by",
            "creative commons attribution",
            "attribution share alike",
            "attribution-sharealike",
        )
    )


def collect_search_hits(
    session: requests.Session,
    throttle: Throttle,
    per_query: int,
    maximum: int,
) -> list[dict[str, Any]]:
    hits: dict[str, dict[str, Any]] = {}
    for query_index, query in enumerate(SEARCH_QUERIES, start=1):
        data = commons_request(
            session,
            throttle,
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srnamespace": 6,
                "srlimit": min(max(per_query, 1), 500),
                "srprop": "",
            },
        )
        rows = data.get("query", {}).get("search", [])
        for rank, row in enumerate(rows, start=1):
            title = clean(row.get("title", ""), 1000)
            if not title:
                continue
            suffix = Path(title.removeprefix("File:")).suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                continue
            current = hits.get(title)
            if current is None:
                hits[title] = {
                    "pageid": safe_int(row.get("pageid")),
                    "title": title,
                    "query": query,
                    "query_rank": rank,
                    "matched_queries": [query],
                }
            else:
                if query not in current["matched_queries"]:
                    current["matched_queries"].append(query)
                if rank < current["query_rank"]:
                    current["query_rank"] = rank
                    current["query"] = query

        print(
            f"[search {query_index:02d}/{len(SEARCH_QUERIES)}] "
            f"{query!r}: returned={len(rows)}, unique={len(hits)}"
        )
        if len(hits) >= maximum:
            break

    ordered = sorted(
        hits.values(),
        key=lambda item: (-len(item["matched_queries"]), item["query_rank"], item["title"]),
    )
    return ordered[:maximum]


def batched(items: list[Any], size: int) -> list[list[Any]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def metadata_bonus(candidate: Candidate) -> float:
    searchable = (
        f"{candidate.title} {candidate.description} "
        f"{candidate.categories} {' '.join(candidate.matched_queries)}"
    ).lower()
    score = 0.0
    score += min(len(candidate.matched_queries), 5) * 0.06
    score += max(0.0, 0.22 - min(candidate.query_rank, 220) / 1000.0)
    score += min(sum(term in searchable for term in GRILL_TERMS), 5) * 0.025
    score -= min(sum(term in searchable for term in NEGATIVE_METADATA_TERMS), 4) * 0.18
    if any(term in searchable for term in ("people", "person", "cook", "chef", "pitmaster", "vendor", "family")):
        score += 0.10
    if any(term in searchable for term in ("grilling", "cooking", "tending", "turning", "roasting")):
        score += 0.10
    return score


def collect_metadata(
    session: requests.Session,
    throttle: Throttle,
    hits: list[dict[str, Any]],
    max_records: int,
) -> list[Candidate]:
    hit_by_title = {item["title"]: item for item in hits}
    candidates: list[Candidate] = []
    source_sha1_seen: set[str] = set()

    batches = batched(hits, 50)
    for batch_index, batch in enumerate(batches, start=1):
        data = commons_request(
            session,
            throttle,
            {
                "action": "query",
                "titles": "|".join(item["title"] for item in batch),
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata|sha1",
                "iiurlwidth": 1024,
            },
        )
        for page in data.get("query", {}).get("pages", []):
            title = clean(page.get("title", ""), 1000)
            hit = hit_by_title.get(title)
            rows = page.get("imageinfo") or []
            if hit is None or not rows:
                continue
            info = rows[0]
            mime = clean(info.get("mime", ""), 100).lower()
            width = safe_int(info.get("width"))
            height = safe_int(info.get("height"))
            if mime not in ALLOWED_MIME or min(width, height) < 360:
                continue

            source_sha1 = clean(info.get("sha1", ""), 200)
            if source_sha1 and source_sha1 in source_sha1_seen:
                continue

            metadata = info.get("extmetadata") or {}
            license_short = metadata_value(metadata, "LicenseShortName", 300)
            usage_terms = metadata_value(metadata, "UsageTerms", 1000)
            if not license_allowed(license_short, usage_terms):
                continue

            description = metadata_value(metadata, "ImageDescription")
            categories = metadata_value(metadata, "Categories")
            object_name = metadata_value(metadata, "ObjectName", 1000)
            searchable = f"{title} {description} {categories} {object_name}".lower()
            if not any(term in searchable for term in GRILL_TERMS):
                continue
            if sum(term in searchable for term in NEGATIVE_METADATA_TERMS) >= 2:
                continue

            creator = metadata_value(metadata, "Artist", 1500)
            attribution = metadata_value(metadata, "Attribution", 2500)
            if not attribution:
                attribution = metadata_value(metadata, "Credit", 2500)
            source_page = clean(info.get("descriptionurl", ""), 4000)
            image_url = clean(info.get("thumburl") or info.get("url"), 4000)
            if not image_url:
                continue

            candidate = Candidate(
                pageid=safe_int(page.get("pageid", hit["pageid"])),
                title=title,
                query=hit["query"],
                query_rank=hit["query_rank"],
                matched_queries=list(hit["matched_queries"]),
                description=description,
                categories=categories,
                creator=creator,
                creator_url=metadata_value(metadata, "ArtistProfile", 2000),
                attribution=attribution,
                license_short=license_short,
                license_url=metadata_value(metadata, "LicenseUrl", 2000)
                or metadata_value(metadata, "License", 2000),
                source_page=source_page,
                image_url=image_url,
                source_sha1=source_sha1,
                width=width,
                height=height,
            )
            candidate.metadata_score = metadata_bonus(candidate)
            candidates.append(candidate)
            if source_sha1:
                source_sha1_seen.add(source_sha1)

        if batch_index % 10 == 0 or batch_index == len(batches):
            print(
                f"[metadata] batch {batch_index}/{len(batches)}; "
                f"accepted={len(candidates)}"
            )
        if len(candidates) >= max_records:
            break

    candidates.sort(
        key=lambda item: (
            item.metadata_score,
            -item.query_rank,
            item.width * item.height,
        ),
        reverse=True,
    )
    return candidates[:max_records]


def normalize_image(raw: bytes, max_side: int, min_side: int) -> tuple[bytes, int, int]:
    if len(raw) < 4096:
        raise ValueError("file too small")
    with Image.open(io.BytesIO(raw)) as source:
        source.load()
        image = ImageOps.exif_transpose(source).convert("RGB")
    width, height = image.size
    if min(width, height) < min_side:
        raise ValueError("image too small")
    ratio = max(width, height) / max(1, min(width, height))
    if ratio > 4.2:
        raise ValueError("extreme aspect ratio")
    stat = ImageStat.Stat(image.resize((64, 64)))
    if sum(stat.var) < 40:
        raise ValueError("nearly blank")

    if max(width, height) > max_side:
        scale = max_side / max(width, height)
        image = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    output = io.BytesIO()
    image.save(
        output,
        format="JPEG",
        quality=86,
        optimize=True,
        progressive=True,
        subsampling=1,
    )
    return output.getvalue(), image.width, image.height


def download_one(
    candidate: Candidate,
    index: int,
    download_dir: Path,
    max_side: int,
    min_side: int,
) -> tuple[int, Candidate | None, str | None]:
    session = make_session()
    try:
        response = session.get(candidate.image_url, timeout=(25, 150), stream=True)
        response.raise_for_status()
        content_length = safe_int(response.headers.get("Content-Length"))
        if content_length and content_length > 30_000_000:
            raise ValueError("file too large")
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(1024 * 256):
            if not chunk:
                continue
            total += len(chunk)
            if total > 30_000_000:
                raise ValueError("file too large")
            chunks.append(chunk)
        normalized, width, height = normalize_image(b"".join(chunks), max_side, min_side)
        path = download_dir / f"candidate_{index:05d}.jpg"
        path.write_bytes(normalized)
        candidate.local_path = str(path)
        candidate.width = width
        candidate.height = height
        candidate.normalized_sha256 = hashlib.sha256(normalized).hexdigest()
        with Image.open(path) as image:
            candidate.phash = str(imagehash.phash(image, hash_size=8))
            candidate.dhash = str(imagehash.dhash(image, hash_size=8))
        return index, candidate, None
    except Exception as exc:
        return index, None, f"{candidate.title}: {type(exc).__name__}: {exc}"
    finally:
        session.close()


def hamming_hex(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def initial_deduplicate(candidates: list[Candidate], phash_threshold: int, dhash_threshold: int) -> list[Candidate]:
    accepted: list[Candidate] = []
    exact_seen: set[str] = set()
    hash_buckets: dict[str, list[int]] = defaultdict(list)

    def prefixes(value: str) -> list[str]:
        return [value[:2], value[2:4], value[4:6], value[6:8]]

    for candidate in candidates:
        if candidate.normalized_sha256 in exact_seen:
            continue
        neighbor_indices: set[int] = set()
        for prefix in prefixes(candidate.phash):
            neighbor_indices.update(hash_buckets[prefix])
        duplicate = False
        for idx in neighbor_indices:
            other = accepted[idx]
            if (
                hamming_hex(candidate.phash, other.phash) <= phash_threshold
                and hamming_hex(candidate.dhash, other.dhash) <= dhash_threshold
            ):
                duplicate = True
                break
        if duplicate:
            continue
        exact_seen.add(candidate.normalized_sha256)
        accepted.append(candidate)
        new_index = len(accepted) - 1
        for prefix in prefixes(candidate.phash):
            hash_buckets[prefix].append(new_index)
    return accepted


def clip_score(
    candidates: list[Candidate],
    batch_size: int,
) -> tuple[list[Candidate], np.ndarray]:
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    text_inputs = processor(
        text=POSITIVE_PROMPTS + NEGATIVE_PROMPTS,
        return_tensors="pt",
        padding=True,
    )
    with torch.inference_mode():
        text_features = model.get_text_features(**{key: value.to(device) for key, value in text_inputs.items()})
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    embeddings: list[np.ndarray] = []
    for start in tqdm(range(0, len(candidates), batch_size), desc="CLIP scoring"):
        batch = candidates[start : start + batch_size]
        images: list[Image.Image] = []
        for candidate in batch:
            with Image.open(candidate.local_path) as source:
                images.append(source.convert("RGB").copy())
        inputs = processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            image_features = model.get_image_features(**{key: value.to(device) for key, value in inputs.items()})
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ text_features.T
        similarities_np = similarities.cpu().numpy()
        embeddings.append(image_features.cpu().numpy().astype(np.float32))
        positive_count = len(POSITIVE_PROMPTS)
        for candidate, row in zip(batch, similarities_np, strict=True):
            positive = float(np.max(row[:positive_count]))
            negative = float(np.max(row[positive_count:]))
            margin = positive - negative
            candidate.clip_positive = positive
            candidate.clip_negative = negative
            candidate.clip_margin = margin
            candidate.final_score = margin + candidate.metadata_score * 0.45

    return candidates, np.concatenate(embeddings, axis=0)


def select_diverse(
    candidates: list[Candidate],
    embeddings: np.ndarray,
    target: int,
    embedding_threshold: float,
    max_per_creator: int,
    max_per_query: int,
) -> tuple[list[Candidate], np.ndarray]:
    order = sorted(
        range(len(candidates)),
        key=lambda index: candidates[index].final_score,
        reverse=True,
    )
    selected: list[Candidate] = []
    selected_embeddings: list[np.ndarray] = []
    creator_counts: Counter[str] = Counter()
    query_counts: Counter[str] = Counter()

    for index in order:
        candidate = candidates[index]
        creator_key = clean(candidate.creator, 300).lower() or f"unknown-{candidate.pageid}"
        query_key = candidate.query
        if creator_counts[creator_key] >= max_per_creator:
            continue
        if query_counts[query_key] >= max_per_query:
            continue

        embedding = embeddings[index]
        if selected_embeddings:
            matrix = np.stack(selected_embeddings[-1200:])
            if float(np.max(matrix @ embedding)) >= embedding_threshold:
                continue

        selected.append(candidate)
        selected_embeddings.append(embedding)
        creator_counts[creator_key] += 1
        query_counts[query_key] += 1
        if len(selected) >= target:
            break

    if len(selected) < target:
        selected_ids = {candidate.pageid for candidate in selected}
        for index in order:
            candidate = candidates[index]
            if candidate.pageid in selected_ids:
                continue
            embedding = embeddings[index]
            if selected_embeddings:
                matrix = np.stack(selected_embeddings[-1200:])
                if float(np.max(matrix @ embedding)) >= embedding_threshold:
                    continue
            selected.append(candidate)
            selected_embeddings.append(embedding)
            selected_ids.add(candidate.pageid)
            if len(selected) >= target:
                break

    return selected, np.stack(selected_embeddings) if selected_embeddings else np.empty((0, 512), dtype=np.float32)


def assign_splits(selected: list[Candidate], seed: int) -> None:
    rng = random.Random(seed)
    indices = list(range(len(selected)))
    rng.shuffle(indices)
    train_end = round(len(indices) * 0.80)
    valid_end = train_end + round(len(indices) * 0.10)
    for position, index in enumerate(indices):
        selected[index].split = (
            "train" if position < train_end else "valid" if position < valid_end else "test"
        )


def safe_filename(index: int, candidate: Candidate) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", candidate.title.removeprefix("File:")).strip("_")
    slug = slug[:60] or f"commons_{candidate.pageid}"
    return f"active_barbecuing_{index:04d}_{candidate.pageid}_{slug}.jpg"


def package_dataset(
    selected: list[Candidate],
    output_dir: Path,
    requested_target: int,
) -> tuple[Path, Path]:
    dataset_root = output_dir / "active_barbecuing_dataset_1000"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    for split in ("train", "valid", "test"):
        (dataset_root / split / "active_barbecuing").mkdir(parents=True, exist_ok=True)

    for index, candidate in enumerate(selected, start=1):
        filename = safe_filename(index, candidate)
        destination = dataset_root / candidate.split / "active_barbecuing" / filename
        shutil.copy2(candidate.local_path, destination)
        candidate.selected_filename = str(destination.relative_to(dataset_root)).replace(os.sep, "/")

    fieldnames = [
        "filename", "split", "title", "source_page", "image_url", "creator",
        "creator_url", "attribution", "license", "license_url", "query",
        "matched_queries", "width", "height", "sha256", "phash", "dhash",
        "metadata_score", "clip_positive", "clip_negative", "clip_margin",
        "final_score",
    ]
    with (dataset_root / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in selected:
            writer.writerow(
                {
                    "filename": candidate.selected_filename,
                    "split": candidate.split,
                    "title": candidate.title,
                    "source_page": candidate.source_page,
                    "image_url": candidate.image_url,
                    "creator": candidate.creator,
                    "creator_url": candidate.creator_url,
                    "attribution": candidate.attribution,
                    "license": candidate.license_short,
                    "license_url": candidate.license_url,
                    "query": candidate.query,
                    "matched_queries": " | ".join(candidate.matched_queries),
                    "width": candidate.width,
                    "height": candidate.height,
                    "sha256": candidate.normalized_sha256,
                    "phash": candidate.phash,
                    "dhash": candidate.dhash,
                    "metadata_score": f"{candidate.metadata_score:.6f}",
                    "clip_positive": f"{candidate.clip_positive:.6f}",
                    "clip_negative": f"{candidate.clip_negative:.6f}",
                    "clip_margin": f"{candidate.clip_margin:.6f}",
                    "final_score": f"{candidate.final_score:.6f}",
                }
            )

    with (dataset_root / "attributions.txt").open("w", encoding="utf-8") as handle:
        for candidate in selected:
            handle.write(
                f"{candidate.selected_filename}\n"
                f"Title: {candidate.title}\n"
                f"Creator: {candidate.creator or 'not supplied'}\n"
                f"Attribution: {candidate.attribution or 'see source page'}\n"
                f"License: {candidate.license_short}\n"
                f"License URL: {candidate.license_url}\n"
                f"Source: {candidate.source_page}\n\n"
            )

    split_counts = Counter(candidate.split for candidate in selected)
    license_counts = Counter(candidate.license_short for candidate in selected)
    readme = f"""# Active Barbecuing Image Dataset

This archive contains {len(selected)} machine-filtered, real photographs from
Wikimedia Commons for the image-level positive class `active_barbecuing`.

A positive example is intended to show at least one visible person actively
cooking, tending food, or using tools at a barbecue, grill, smoker, or closely
related outdoor cooking setup. The collection pipeline used Commons metadata,
CLIP relevance ranking, exact hashes, perceptual hashes, and CLIP-embedding
similarity filtering. Automated selection can still leave borderline or
incorrect examples, so review the supplied contact sheets before training.

Requested target: {requested_target}
Delivered images: {len(selected)}
Train/valid/test: {split_counts['train']}/{split_counts['valid']}/{split_counts['test']}
License counts: {dict(sorted(license_counts.items()))}

Folders:
- train/active_barbecuing
- valid/active_barbecuing
- test/active_barbecuing

Files:
- metadata.csv: source, license, hashes, and automated scores
- attributions.txt: attribution details required by many Commons licenses

This is a positive-only dataset. A binary classifier also needs hard negatives,
including unattended grills, grill products, food-only close-ups, people merely
eating near a grill, campfires, and unrelated outdoor scenes.
"""
    (dataset_root / "README.md").write_text(readme, encoding="utf-8")

    dataset_zip = output_dir / "active_barbecuing_dataset_1000.zip"
    with zipfile.ZipFile(dataset_zip, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(dataset_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(dataset_root))

    review_dir = output_dir / "review_sheets"
    review_dir.mkdir(parents=True, exist_ok=True)
    make_contact_sheets(selected, review_dir)
    review_zip = output_dir / "active_barbecuing_review_sheets.zip"
    with zipfile.ZipFile(review_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(review_dir.glob("*.jpg")):
            archive.write(path, path.name)

    return dataset_zip, review_zip


def make_contact_sheets(selected: list[Candidate], output_dir: Path) -> None:
    font = ImageFont.load_default()
    per_sheet = 100
    columns = 10
    cell_width = 180
    cell_height = 150
    thumb_height = 120
    for sheet_index, start in enumerate(range(0, len(selected), per_sheet), start=1):
        batch = selected[start : start + per_sheet]
        rows = math.ceil(len(batch) / columns)
        canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
        draw = ImageDraw.Draw(canvas)
        for offset, candidate in enumerate(batch):
            row, column = divmod(offset, columns)
            x = column * cell_width
            y = row * cell_height
            with Image.open(candidate.local_path) as source:
                thumb = source.convert("RGB")
                thumb.thumbnail((cell_width, thumb_height), Image.Resampling.LANCZOS)
                px = x + (cell_width - thumb.width) // 2
                py = y + (thumb_height - thumb.height) // 2
                canvas.paste(thumb, (px, py))
            label = f"{start + offset + 1:04d} score={candidate.final_score:.3f}"
            draw.text((x + 3, y + thumb_height + 2), label, fill="black", font=font)
        canvas.save(
            output_dir / f"review_{sheet_index:02d}.jpg",
            "JPEG",
            quality=88,
            optimize=True,
        )


def validate_dataset_zip(dataset_zip: Path, target: int) -> dict[str, Any]:
    image_entries: list[str] = []
    hashes: list[str] = []
    split_counts: Counter[str] = Counter()
    with zipfile.ZipFile(dataset_zip, "r") as archive:
        bad_entry = archive.testzip()
        if bad_entry:
            raise RuntimeError(f"ZIP integrity failure at {bad_entry}")
        for info in archive.infolist():
            if info.is_dir():
                continue
            suffix = Path(info.filename).suffix.lower()
            if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
                raw = archive.read(info)
                with Image.open(io.BytesIO(raw)) as image:
                    image.verify()
                image_entries.append(info.filename)
                hashes.append(hashlib.sha256(raw).hexdigest())
                split_counts[info.filename.split("/", 1)[0]] += 1

    if len(image_entries) != target:
        raise RuntimeError(f"Expected {target} images, found {len(image_entries)}")
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("Exact duplicate files detected in final ZIP")

    return {
        "zip_path": str(dataset_zip),
        "zip_size_bytes": dataset_zip.stat().st_size,
        "image_count": len(image_entries),
        "unique_sha256_count": len(set(hashes)),
        "split_counts": dict(split_counts),
        "integrity_test": "passed",
        "all_images_decode": True,
    }


def write_jsonl(path: Path, candidates: list[Candidate]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(asdict(candidate), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("quick_barbecue_output"))
    parser.add_argument("--per-query", type=int, default=350)
    parser.add_argument("--max-search-hits", type=int, default=3400)
    parser.add_argument("--max-metadata-records", type=int, default=2800)
    parser.add_argument("--max-downloads", type=int, default=2500)
    parser.add_argument("--download-workers", type=int, default=12)
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--min-side", type=int, default=360)
    parser.add_argument("--request-pause", type=float, default=0.8)
    parser.add_argument("--phash-threshold", type=int, default=6)
    parser.add_argument("--dhash-threshold", type=int, default=5)
    parser.add_argument("--clip-batch-size", type=int, default=48)
    parser.add_argument("--embedding-threshold", type=float, default=0.996)
    parser.add_argument("--max-per-creator", type=int, default=45)
    parser.add_argument("--max-per-query", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    download_dir = args.output / "downloaded_candidates"
    if download_dir.exists():
        shutil.rmtree(download_dir)
    download_dir.mkdir(parents=True)

    session = make_session()
    throttle = Throttle(args.request_pause)
    try:
        hits = collect_search_hits(
            session,
            throttle,
            per_query=args.per_query,
            maximum=args.max_search_hits,
        )
        print(f"Collected {len(hits)} distinct Commons file search hits.")
        candidates = collect_metadata(
            session,
            throttle,
            hits,
            max_records=args.max_metadata_records,
        )
    finally:
        session.close()

    print(f"Accepted {len(candidates)} open-license metadata records.")
    if len(candidates) < args.target:
        raise RuntimeError(
            f"Only {len(candidates)} eligible metadata records for target {args.target}"
        )
    write_jsonl(args.output / "commons_metadata_candidates.jsonl", candidates)

    download_candidates = candidates[: args.max_downloads]
    downloaded_by_index: dict[int, Candidate] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        futures = [
            executor.submit(
                download_one,
                candidate,
                index,
                download_dir,
                args.max_side,
                args.min_side,
            )
            for index, candidate in enumerate(download_candidates)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            index, candidate, error = future.result()
            if candidate is not None:
                downloaded_by_index[index] = candidate
            elif error:
                errors.append(error)

    downloaded = [downloaded_by_index[index] for index in sorted(downloaded_by_index)]
    (args.output / "download_errors.txt").write_text("\n".join(errors), encoding="utf-8")
    print(f"Downloaded and decoded {len(downloaded)} images; failures={len(errors)}.")

    deduplicated = initial_deduplicate(
        downloaded,
        phash_threshold=args.phash_threshold,
        dhash_threshold=args.dhash_threshold,
    )
    print(f"After exact/perceptual deduplication: {len(deduplicated)} images.")
    if len(deduplicated) < args.target:
        raise RuntimeError(
            f"Only {len(deduplicated)} images remain after deduplication; target={args.target}"
        )

    scored, embeddings = clip_score(deduplicated, batch_size=args.clip_batch_size)
    selected, _ = select_diverse(
        scored,
        embeddings,
        target=args.target,
        embedding_threshold=args.embedding_threshold,
        max_per_creator=args.max_per_creator,
        max_per_query=args.max_per_query,
    )
    print(f"Selected {len(selected)} ranked images.")
    if len(selected) != args.target:
        raise RuntimeError(f"Could only select {len(selected)} images for target {args.target}")

    assign_splits(selected, args.seed)
    dataset_zip, review_zip = package_dataset(selected, args.output, args.target)
    validation = validate_dataset_zip(dataset_zip, args.target)
    validation.update(
        {
            "review_zip_path": str(review_zip),
            "metadata_candidates": len(candidates),
            "downloaded_candidates": len(downloaded),
            "deduplicated_candidates": len(deduplicated),
            "selected_candidates": len(selected),
            "lowest_selected_score": min(candidate.final_score for candidate in selected),
            "highest_selected_score": max(candidate.final_score for candidate in selected),
            "license_counts": dict(Counter(candidate.license_short for candidate in selected)),
        }
    )
    (args.output / "build_summary.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_jsonl(args.output / "selected_metadata.jsonl", selected)
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
