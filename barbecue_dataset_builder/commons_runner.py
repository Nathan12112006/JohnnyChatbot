#!/usr/bin/env python3
"""Wikimedia Commons source adapter for the active-barbecuing dataset builder."""

from __future__ import annotations

import html
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

import build as base

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "ActiveBarbecuingDatasetBuilder/2.0 "
    "(research image collection; contact: GitHub user Nathan12112006)"
)

SEARCH_QUERIES = [
    "barbecue",
    "barbeque",
    "barbecuing",
    "barbequing",
    "BBQ grilling",
    "barbecue cooking people",
    "person grilling",
    "people grilling",
    "barbecue cook",
    "outdoor grill cooking",
    "barbecue festival",
    "barbecue competition",
    "pitmaster barbecue",
    "backyard barbecue",
    "park barbecue grilling",
    "beach barbecue",
    "tailgate grilling",
    "cookout grill",
    "braai cooking",
    "churrasco grilling",
    "churrasqueira cooking",
    "asado parrilla",
    "parrillero asado",
    "mangal cooking",
    "shashlik grill",
    "kebab grilling",
    "satay grilling",
    "yakitori grilling",
    "Grillmeister grillt",
    "grillen personen",
    "churrasqueiro churrasco",
    "parrillada personas",
    "barbecue personne cuisine",
    "バーベキュー 人 焼く",
    "烧烤 人 烤肉",
]

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
NEGATIVE_METADATA_TERMS = {
    "logo", "diagram", "illustration", "drawing", "poster", "advertisement",
    "icon", "vector", "render", "map", "menu", "recipe", "product photo",
    "car grille", "radiator grille", "locomotive", "motorcycle", "album cover",
}


@dataclass
class SearchHit:
    title: str
    pageid: int
    best_query: str
    best_rank: int
    queries: list[str] = field(default_factory=list)


class Throttle:
    def __init__(self, interval: float) -> None:
        self.interval = max(interval, 0.0)
        self.last_call = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        remaining = self.interval - (now - self.last_call)
        if remaining > 0:
            time.sleep(remaining)
        self.last_call = time.monotonic()


def clean(value: Any, limit: int = 5000) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def ext_value(metadata: dict[str, Any], key: str, limit: int = 5000) -> str:
    value = metadata.get(key, {})
    if isinstance(value, dict):
        value = value.get("value", "")
    return clean(value, limit)


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def license_fields(short_name: str, usage_terms: str) -> tuple[str, str, bool]:
    text = re.sub(r"[_-]+", " ", f"{short_name} {usage_terms}".lower())
    text = re.sub(r"\s+", " ", text)
    if any(term in text for term in ("noncommercial", "no derivatives", " cc nc", " cc nd")):
        return "", "", False
    if "cc0" in text or "creative commons zero" in text:
        return "cc0", short_name or "CC0", True
    if any(term in text for term in ("public domain", "no known restrictions", "pdm", "pd us", "pd old")):
        return "pdm", short_name or "Public domain", True
    if "cc by sa" in text or "attribution share alike" in text or "attribution-sharealike" in text:
        return "by-sa", short_name or "CC BY-SA", True
    if "cc by" in text or "creative commons attribution" in text:
        return "by", short_name or "CC BY", True
    return "", "", False


def make_session() -> requests.Session:
    session = base.build_session(total_retries=3)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def commons_post(
    session: requests.Session,
    throttle: Throttle,
    params: dict[str, Any],
    attempts: int = 8,
) -> dict[str, Any]:
    data = dict(params)
    data.setdefault("format", "json")
    data.setdefault("formatversion", 2)
    data.setdefault("maxlag", 5)
    last_error: Exception | None = None
    for attempt in range(attempts):
        throttle.wait()
        try:
            response = session.post(COMMONS_API, data=data, timeout=(20, 120))
            if response.status_code in {429, 500, 502, 503, 504}:
                retry_after = safe_int(response.headers.get("Retry-After"))
                delay = max(retry_after, min(90, 3 * (2**attempt)))
                print(f"Commons API {response.status_code}; sleeping {delay}s", file=sys.stderr)
                time.sleep(delay)
                continue
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                code = str(payload["error"].get("code", ""))
                if code in {"maxlag", "ratelimited", "readonly"}:
                    delay = min(90, 3 * (2**attempt))
                    print(f"Commons API {code}; sleeping {delay}s", file=sys.stderr)
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Commons API error: {payload['error']}")
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            delay = min(90, 3 * (2**attempt))
            print(f"Commons request failed ({exc}); sleeping {delay}s", file=sys.stderr)
            time.sleep(delay)
    raise RuntimeError(f"Commons request failed after {attempts} attempts: {last_error}")


def collect_search_hits(
    session: requests.Session,
    throttle: Throttle,
    per_query: int,
    maximum: int,
) -> list[SearchHit]:
    hits: dict[str, SearchHit] = {}
    for query_index, query in enumerate(SEARCH_QUERIES, start=1):
        payload = commons_post(
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
        rows = payload.get("query", {}).get("search", [])
        for rank, row in enumerate(rows, start=1):
            title = clean(row.get("title", ""), 1000)
            if not title or Path(title.removeprefix("File:")).suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            existing = hits.get(title)
            if existing is None:
                hits[title] = SearchHit(
                    title=title,
                    pageid=safe_int(row.get("pageid")),
                    best_query=query,
                    best_rank=rank,
                    queries=[query],
                )
            elif query not in existing.queries:
                existing.queries.append(query)
                if rank < existing.best_rank:
                    existing.best_rank = rank
                    existing.best_query = query
        print(
            f"[Commons search] {query_index:02d}/{len(SEARCH_QUERIES)} "
            f"{query!r}: returned={len(rows)}, unique={len(hits)}"
        )
        if len(hits) >= maximum:
            break
    ordered = sorted(hits.values(), key=lambda item: (-len(item.queries), item.best_rank, item.title))
    print(f"Commons search produced {len(ordered)} distinct image titles.")
    return ordered


def chunks(items: list[SearchHit], size: int = 50):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def candidate_from_page(page: dict[str, Any], hit: SearchHit) -> base.Candidate | None:
    rows = page.get("imageinfo") or []
    if not rows:
        return None
    info = rows[0]
    mime = clean(info.get("mime", ""), 100).lower()
    width = safe_int(info.get("width"))
    height = safe_int(info.get("height"))
    if mime not in ALLOWED_MIME or min(width, height) < 384:
        return None

    metadata = info.get("extmetadata") or {}
    short_name = ext_value(metadata, "LicenseShortName", 300)
    usage_terms = ext_value(metadata, "UsageTerms", 1000)
    license_code, license_display, allowed = license_fields(short_name, usage_terms)
    if not allowed:
        return None

    title = clean(page.get("title", hit.title), 1000)
    description = ext_value(metadata, "ImageDescription", 4000)
    categories = ext_value(metadata, "Categories", 5000)
    object_name = ext_value(metadata, "ObjectName", 1000)
    searchable = f"{title} {description} {categories} {object_name}".lower()
    if sum(term in searchable for term in NEGATIVE_METADATA_TERMS) >= 2:
        return None

    creator = ext_value(metadata, "Artist", 2000)
    creator_url = ext_value(metadata, "ArtistProfile", 2000)
    attribution = ext_value(metadata, "Attribution", 3000)
    credit = ext_value(metadata, "Credit", 3000)
    if not attribution:
        attribution = credit or f"{title} by {creator or 'unknown creator'}"
    license_url = ext_value(metadata, "LicenseUrl", 2000)
    if not license_url:
        license_url = ext_value(metadata, "License", 2000)

    original_url = clean(info.get("url", ""), 4000)
    thumbnail_url = clean(info.get("thumburl", ""), 4000)
    description_url = clean(info.get("descriptionurl", ""), 4000)
    if not original_url and not thumbnail_url:
        return None

    tags = " | ".join(
        part for part in (
            description,
            object_name,
            categories,
            "matched searches: " + " ; ".join(hit.queries),
        )
        if part
    )
    filetype = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}[mime]
    candidate = base.Candidate(
        openverse_id=f"commons-{page.get('pageid', hit.pageid)}",
        title=title,
        creator=creator,
        creator_url=creator_url,
        license=license_code,
        license_version=license_display,
        license_url=license_url,
        attribution=attribution,
        provider="Wikimedia Foundation",
        source="Wikimedia Commons",
        foreign_landing_url=description_url,
        original_url=original_url,
        thumbnail_url=thumbnail_url,
        filetype=filetype,
        width_reported=width,
        height_reported=height,
        query=hit.best_query,
        tags=tags,
        category="photograph",
        api_rank=hit.best_rank,
    )
    candidate.metadata_score = base.compute_metadata_score(candidate)
    candidate.metadata_score += min(len(hit.queries), 5) * 0.08
    candidate.water_context = base.has_water_context(candidate)
    return candidate


def collect_metadata(output_dir: Path, per_query: int, request_pause: float) -> list[base.Candidate]:
    session = make_session()
    throttle = Throttle(max(request_pause, 1.8))
    maximum = max(7500, per_query * 18)
    hits = collect_search_hits(session, throttle, per_query=min(per_query, 500), maximum=maximum)
    hit_by_title = {hit.title: hit for hit in hits}
    candidates: list[base.Candidate] = []
    source_sha1_seen: set[str] = set()

    batches = list(chunks(hits, 50))
    for batch_index, batch in enumerate(batches, start=1):
        payload = commons_post(
            session,
            throttle,
            {
                "action": "query",
                "titles": "|".join(hit.title for hit in batch),
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata|sha1",
                "iiurlwidth": 1600,
            },
        )
        for page in payload.get("query", {}).get("pages", []):
            title = clean(page.get("title", ""), 1000)
            hit = hit_by_title.get(title)
            if hit is None:
                continue
            info_rows = page.get("imageinfo") or []
            source_sha1 = clean(info_rows[0].get("sha1", ""), 200) if info_rows else ""
            if source_sha1 and source_sha1 in source_sha1_seen:
                continue
            candidate = candidate_from_page(page, hit)
            if candidate is None:
                continue
            if source_sha1:
                source_sha1_seen.add(source_sha1)
            candidates.append(candidate)
        if batch_index % 20 == 0:
            print(f"[Commons metadata] {batch_index}/{len(batches)} batches; accepted={len(candidates)}")

    candidates.sort(
        key=lambda item: (item.metadata_score, -item.api_rank, item.width_reported * item.height_reported),
        reverse=True,
    )
    metadata_path = output_dir / "commons_candidates.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(base.candidate_for_json(candidate), ensure_ascii=False) + "\n")
    print(f"Accepted {len(candidates)} Commons records after license/type/size filtering.")
    print("License counts:", dict(Counter(item.license_version for item in candidates)))
    return candidates


def write_readme(dataset_root: Path, selected: list[base.Candidate], requested_target: int) -> None:
    split_counts = Counter(item.split for item in selected)
    license_counts = Counter(item.license_version for item in selected)
    tier_counts = Counter(item.confidence_tier for item in selected)
    water_count = sum(1 for item in selected if item.water_context)
    text = f"""# Active Barbecuing Open-License Image Dataset

This package contains {len(selected)} real photographs selected for the
image-level label `active_barbecuing`.

## Inclusion rule

A positive image should visibly show at least one person actively cooking,
tending food, or using tools at a barbecue or grill. An unattended grill,
food-only close-up, grill product image, illustration, ordinary picnic without
visible cooking, or a person merely posing near a grill is outside the intended
class.

## Layout

- `train/active_barbecuing/`: {split_counts['train']} images
- `valid/active_barbecuing/`: {split_counts['valid']} images
- `test/active_barbecuing/`: {split_counts['test']} images
- `metadata.csv`: source, creator, license, hashes, and automated QA scores
- `attributions.txt`: per-image source and license information

Requested target: {requested_target}
Delivered: {len(selected)}
License counts: {dict(sorted(license_counts.items()))}
Automated confidence tiers: {dict(sorted(tier_counts.items()))}
Metadata-indicated waterside images: {water_count}

## Collection and QA

Candidates were discovered through Wikimedia Commons. Only files whose Commons
metadata indicated CC0, public domain, CC BY, or CC BY-SA were retained. The
pipeline validates image decoding, normalizes orientation and format, removes
exact and visual near-duplicates using SHA-256, perceptual hashes, and CLIP
embeddings, scores active-barbecuing relevance with CLIP, and requires a visible
person using a COCO detector. Creator groups are kept together when assigning
approximately 80/10/10 train, validation, and test splits.

Automated filtering is not a substitute for human verification. The separate
review ZIP provides contact sheets covering every selected image. Source pages,
creator details, license information, and attribution text returned by Commons
are preserved in the metadata. CC BY requires attribution; CC BY-SA also has
ShareAlike obligations.

This package contains positive examples only. A binary classifier also needs a
separate negative class, particularly hard negatives such as empty grills,
people eating near a grill, food-only images, campfires, and grill products.
"""
    (dataset_root / "README.md").write_text(text, encoding="utf-8")


base.POSITIVE_PROMPTS[:] = [
    "a real photograph of a person actively cooking food on a barbecue grill",
    "a real photograph of someone using tongs to turn food on an outdoor grill",
    "a real photograph of people actively grilling at a barbecue",
    "a real photograph of a pitmaster tending meat in a barbecue smoker",
    "a real photograph of a cook standing over a charcoal grill",
    "a real photograph of a street food cook grilling meat over fire",
    "a real photograph of someone preparing food on a barbecue",
]
base.NEGATIVE_PROMPTS[:] = [
    "a photograph of an empty barbecue grill with nobody cooking",
    "a photograph of food cooking on a grill with no person visible",
    "a close-up photograph of grilled food on a plate",
    "a photograph of people eating at a picnic without grilling",
    "a product photograph of a barbecue grill",
    "a photograph of a campfire without barbecue cooking",
    "a photograph of a car front grille",
    "a photograph of a building or landscape",
    "an illustration logo poster or diagram",
    "a photograph of a kitchen appliance",
    "a photograph unrelated to outdoor cooking",
]
base.collect_metadata = collect_metadata
base.write_readme = write_readme

if __name__ == "__main__":
    raise SystemExit(base.main())
