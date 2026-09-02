#!/usr/bin/env python3
"""Build and validate a 1,000-image active-barbecuing dataset from Wikimedia Commons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
import random
import re
import shutil
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import imagehash
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageStat
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "ActiveBarbecuingDatasetBuilder/4.0 (research; GitHub Nathan12112006)"

QUERIES = [
    "people actively grilling barbecue", "person cooking on barbecue grill",
    "people grilling meat", "barbecue cook", "barbecue competition cooking",
    "barbecue festival grilling", "pitmaster tending barbecue",
    "outdoor cook using grill", "backyard barbecue cooking",
    "park barbecue people", "beach barbecue cooking", "tailgate grilling",
    "braai cooking people", "churrasco grilling people", "asado parrilla people",
    "parrillero cooking asado", "mangal cooking people", "shashlik grilling",
    "kebab grilling vendor", "satay grilling vendor", "yakitori grilling cook",
    "Grillmeister grillt", "churrasqueiro churrasco", "barbecue personne cuisine",
    "バーベキュー 人 焼く", "烧烤 烤肉 人",
]

GRILL_TERMS = (
    "barbecue", "barbeque", "barbecu", "barbequ", "bbq", "grill", "braai",
    "churrasco", "churrasqueira", "asado", "parrilla", "parrillero", "mangal",
    "shashlik", "kebab", "satay", "yakitori",
)
ACTION_TERMS = (
    "cook", "cooking", "grill", "grilling", "roast", "roasting", "tending",
    "turning", "pitmaster", "chef", "vendor", "preparing", "barbecuing",
    "barbequing",
)
PERSON_TERMS = (
    "people", "person", "man", "woman", "family", "friends", "cook", "chef",
    "pitmaster", "vendor", "group", "crowd",
)
NEGATIVE_TERMS = (
    "logo", "diagram", "illustration", "drawing", "poster", "advertisement",
    "icon", "vector", "render", "map", "menu", "recipe", "product photo",
    "car grille", "radiator grille", "locomotive", "motorcycle", "album cover",
    "clip art", "coat of arms", "floor plan", "screenshot", "toy grill",
)
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}


@dataclass
class Item:
    pageid: int
    title: str
    best_query: str
    best_rank: int
    queries: list[str] = field(default_factory=list)
    description: str = ""
    categories: str = ""
    creator: str = ""
    creator_url: str = ""
    attribution: str = ""
    license_name: str = ""
    license_url: str = ""
    source_page: str = ""
    image_url: str = ""
    source_sha1: str = ""
    width: int = 0
    height: int = 0
    score: float = 0.0
    local_path: str = ""
    sha256: str = ""
    phash: str = ""
    dhash: str = ""
    split: str = ""
    output_name: str = ""


def text(value: Any, limit: int = 6000) -> str:
    if value is None:
        return ""
    value = html.unescape(str(value))
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:limit]


def meta(metadata: dict[str, Any], key: str, limit: int = 6000) -> str:
    value = metadata.get(key, "")
    if isinstance(value, dict):
        value = value.get("value", "")
    return text(value, limit)


def integer(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def session() -> requests.Session:
    retry = Retry(
        total=5, connect=5, read=5, status=5, backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}), respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=40, pool_maxsize=40)
    result = requests.Session()
    result.headers.update({"User-Agent": USER_AGENT})
    result.mount("https://", adapter)
    result.mount("http://", adapter)
    return result


class Throttle:
    def __init__(self, pause: float) -> None:
        self.pause = pause
        self.last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self.last
        if elapsed < self.pause:
            time.sleep(self.pause - elapsed)
        self.last = time.monotonic()


def api_call(client: requests.Session, throttle: Throttle, params: dict[str, Any]) -> dict[str, Any]:
    payload = dict(params)
    payload.update({"format": "json", "formatversion": 2, "maxlag": 5})
    last_error: Exception | None = None
    for attempt in range(7):
        throttle.wait()
        try:
            response = client.post(API, data=payload, timeout=(20, 120))
            if response.status_code in {429, 500, 502, 503, 504}:
                delay = max(integer(response.headers.get("Retry-After")), min(60, 2 ** (attempt + 1)))
                print(f"Commons HTTP {response.status_code}; retry in {delay}s", file=sys.stderr)
                time.sleep(delay)
                continue
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                raise RuntimeError(str(data["error"]))
            return data
        except Exception as exc:
            last_error = exc
            if attempt == 6:
                break
            time.sleep(min(60, 2 ** (attempt + 1)))
    raise RuntimeError(f"Commons API failed: {last_error}")


def discover(client: requests.Session, throttle: Throttle, per_query: int, maximum: int) -> list[dict[str, Any]]:
    hits: dict[str, dict[str, Any]] = {}
    for q_index, query in enumerate(QUERIES, start=1):
        data = api_call(client, throttle, {
            "action": "query", "list": "search", "srsearch": query,
            "srnamespace": 6, "srlimit": min(500, per_query), "srprop": "",
        })
        rows = data.get("query", {}).get("search", [])
        for rank, row in enumerate(rows, start=1):
            title = text(row.get("title", ""), 1000)
            if not title or Path(title.removeprefix("File:")).suffix.lower() not in ALLOWED_EXT:
                continue
            current = hits.get(title)
            if current is None:
                hits[title] = {
                    "pageid": integer(row.get("pageid")), "title": title,
                    "best_query": query, "best_rank": rank, "queries": [query],
                }
            else:
                if query not in current["queries"]:
                    current["queries"].append(query)
                if rank < current["best_rank"]:
                    current["best_rank"] = rank
                    current["best_query"] = query
        print(f"[search {q_index:02d}] returned={len(rows)} unique={len(hits)}")
        if len(hits) >= maximum:
            break
    return sorted(hits.values(), key=lambda row: (-len(row["queries"]), row["best_rank"], row["title"]))[:maximum]


def score_item(item: Item) -> float:
    value = f"{item.title} {item.description} {item.categories} {' '.join(item.queries)}".lower()
    score = min(len(item.queries), 5) * 0.45
    score += max(0.0, 1.2 - item.best_rank / 400.0)
    score += min(sum(term in value for term in GRILL_TERMS), 5) * 0.35
    score += min(sum(term in value for term in ACTION_TERMS), 4) * 0.6
    score += min(sum(term in value for term in PERSON_TERMS), 3) * 0.45
    score -= min(sum(term in value for term in NEGATIVE_TERMS), 3) * 2.0
    return score


def metadata(client: requests.Session, throttle: Throttle, hits: list[dict[str, Any]], maximum: int) -> list[Item]:
    by_title = {row["title"]: row for row in hits}
    output: list[Item] = []
    source_seen: set[str] = set()
    batches = [hits[i:i + 50] for i in range(0, len(hits), 50)]
    for batch_index, batch in enumerate(batches, start=1):
        data = api_call(client, throttle, {
            "action": "query", "titles": "|".join(row["title"] for row in batch),
            "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata|sha1",
            "iiurlwidth": 768,
        })
        for page in data.get("query", {}).get("pages", []):
            title = text(page.get("title", ""), 1000)
            hit = by_title.get(title)
            info_rows = page.get("imageinfo") or []
            if hit is None or not info_rows:
                continue
            info = info_rows[0]
            mime = text(info.get("mime", ""), 100).lower()
            width, height = integer(info.get("width")), integer(info.get("height"))
            if mime not in ALLOWED_MIME or min(width, height) < 320:
                continue
            source_sha1 = text(info.get("sha1", ""), 200)
            if source_sha1 and source_sha1 in source_seen:
                continue
            md = info.get("extmetadata") or {}
            description, categories = meta(md, "ImageDescription"), meta(md, "Categories")
            searchable = f"{title} {description} {categories} {meta(md, 'ObjectName', 1000)}".lower()
            if not any(term in searchable for term in GRILL_TERMS):
                continue
            if sum(term in searchable for term in NEGATIVE_TERMS) >= 2:
                continue
            image_url = text(info.get("thumburl") or info.get("url"), 4000)
            if not image_url:
                continue
            item = Item(
                pageid=integer(page.get("pageid", hit["pageid"])), title=title,
                best_query=hit["best_query"], best_rank=hit["best_rank"],
                queries=list(hit["queries"]), description=description,
                categories=categories, creator=meta(md, "Artist", 1600),
                creator_url=meta(md, "ArtistProfile", 2000),
                attribution=meta(md, "Attribution", 2500) or meta(md, "Credit", 2500),
                license_name=meta(md, "LicenseShortName", 400) or meta(md, "UsageTerms", 400),
                license_url=meta(md, "LicenseUrl", 2000) or meta(md, "License", 2000),
                source_page=text(info.get("descriptionurl"), 4000),
                image_url=image_url, source_sha1=source_sha1, width=width, height=height,
            )
            item.score = score_item(item)
            output.append(item)
            if source_sha1:
                source_seen.add(source_sha1)
        if batch_index % 10 == 0 or batch_index == len(batches):
            print(f"[metadata] {batch_index}/{len(batches)} accepted={len(output)}")
        if len(output) >= maximum:
            break
    output.sort(key=lambda item: (item.score, -item.best_rank, item.width * item.height), reverse=True)
    return output[:maximum]


def normalize(raw: bytes, max_side: int, min_side: int) -> bytes:
    if len(raw) < 4096:
        raise ValueError("tiny response")
    with Image.open(io.BytesIO(raw)) as src:
        src.load()
        image = ImageOps.exif_transpose(src).convert("RGB")
    if min(image.size) < min_side:
        raise ValueError("small image")
    if max(image.size) / max(1, min(image.size)) > 4.5:
        raise ValueError("extreme aspect ratio")
    if sum(ImageStat.Stat(image.resize((48, 48))).var) < 30:
        raise ValueError("blank image")
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=82, optimize=True, progressive=True, subsampling=1)
    return buffer.getvalue()


def download_one(item: Item, index: int, folder: Path, max_side: int, min_side: int) -> tuple[int, Item | None, str]:
    client = session()
    try:
        response = client.get(item.image_url, timeout=(20, 90), stream=True)
        response.raise_for_status()
        data = bytearray()
        for chunk in response.iter_content(262144):
            if chunk:
                data.extend(chunk)
                if len(data) > 25_000_000:
                    raise ValueError("image too large")
        normalized = normalize(bytes(data), max_side, min_side)
        path = folder / f"{index:05d}.jpg"
        path.write_bytes(normalized)
        item.local_path = str(path)
        item.sha256 = hashlib.sha256(normalized).hexdigest()
        with Image.open(path) as image:
            item.width, item.height = image.size
            item.phash = str(imagehash.phash(image, hash_size=8))
            item.dhash = str(imagehash.dhash(image, hash_size=8))
        return index, item, ""
    except Exception as exc:
        return index, None, f"{item.title}\t{type(exc).__name__}: {exc}"
    finally:
        client.close()


def hex_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def deduplicate(items: list[Item], phash_limit: int, dhash_limit: int) -> list[Item]:
    accepted: list[Item] = []
    exact: set[str] = set()
    for item in tqdm(items, desc="Deduplicating"):
        if item.sha256 in exact:
            continue
        if any(hex_distance(item.phash, other.phash) <= phash_limit and hex_distance(item.dhash, other.dhash) <= dhash_limit for other in accepted):
            continue
        exact.add(item.sha256)
        accepted.append(item)
    return accepted


def select(items: list[Item], target: int, creator_cap: int, query_cap: int) -> list[Item]:
    creator_counts: Counter[str] = Counter()
    query_counts: Counter[str] = Counter()
    chosen: list[Item] = []
    chosen_ids: set[int] = set()
    for item in items:
        creator_key = text(item.creator, 300).lower() or f"unknown-{item.pageid}"
        if creator_counts[creator_key] >= creator_cap or query_counts[item.best_query] >= query_cap:
            continue
        chosen.append(item)
        chosen_ids.add(item.pageid)
        creator_counts[creator_key] += 1
        query_counts[item.best_query] += 1
        if len(chosen) == target:
            return chosen
    for item in items:
        if item.pageid not in chosen_ids:
            chosen.append(item)
            chosen_ids.add(item.pageid)
            if len(chosen) == target:
                return chosen
    return chosen


def filename(index: int, item: Item) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", item.title.removeprefix("File:")).strip("_")[:55]
    return f"active_barbecuing_{index:04d}_{item.pageid}_{slug or 'image'}.jpg"


def contact_sheet(items: list[Item], destination: Path) -> None:
    subset = items[:100]
    cell_w, cell_h, image_h, columns = 170, 145, 116, 10
    rows = (len(subset) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    for i, item in enumerate(subset):
        row, col = divmod(i, columns)
        x, y = col * cell_w, row * cell_h
        with Image.open(item.local_path) as src:
            thumb = src.convert("RGB")
            thumb.thumbnail((cell_w, image_h), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (x + (cell_w - thumb.width) // 2, y + (image_h - thumb.height) // 2))
        draw.text((x + 3, y + image_h + 2), f"{i + 1:04d} score={item.score:.2f}", fill="black", font=font)
    canvas.save(destination, "JPEG", quality=86, optimize=True)


def package(items: list[Item], output: Path, seed: int) -> tuple[Path, Path]:
    root = output / "active_barbecuing_dataset_1000"
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "valid", "test"):
        (root / split / "active_barbecuing").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    split_by_index = {index: ("train" if pos < 800 else "valid" if pos < 900 else "test") for pos, index in enumerate(order)}
    for index, item in enumerate(items, start=1):
        item.split = split_by_index[index - 1]
        item.output_name = f"{item.split}/active_barbecuing/{filename(index, item)}"
        shutil.copy2(item.local_path, root / item.output_name)
    fields = ["filename", "split", "title", "source_page", "image_url", "creator", "creator_url", "attribution", "license", "license_url", "query", "matched_queries", "score", "width", "height", "sha256", "phash", "dhash"]
    with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in items:
            writer.writerow({"filename": item.output_name, "split": item.split, "title": item.title, "source_page": item.source_page, "image_url": item.image_url, "creator": item.creator, "creator_url": item.creator_url, "attribution": item.attribution, "license": item.license_name, "license_url": item.license_url, "query": item.best_query, "matched_queries": " | ".join(item.queries), "score": f"{item.score:.4f}", "width": item.width, "height": item.height, "sha256": item.sha256, "phash": item.phash, "dhash": item.dhash})
    with (root / "attributions.txt").open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(f"{item.output_name}\nTitle: {item.title}\nCreator: {item.creator or 'not supplied'}\nAttribution: {item.attribution or 'see source page'}\nLicense: {item.license_name or 'see source page'}\nLicense URL: {item.license_url}\nSource: {item.source_page}\n\n")
    license_counts = dict(Counter(item.license_name or "unspecified" for item in items))
    (root / "README.md").write_text(
        "# Active Barbecuing Image Dataset\n\nThis archive contains 1,000 real photographs collected from Wikimedia Commons as machine-filtered positive candidates for the image-level class `active_barbecuing`.\n\nThe intended inclusion rule is a visible person cooking, tending food, or using a barbecue, grill, smoker, or related outdoor cooking setup. Search metadata and visual duplicate hashing were used, but this fast delivery is not a full human review. Inspect the preview and metadata before training.\n\nSplits: 800 train, 100 validation, 100 test.\n\n" + f"License metadata counts: {license_counts}\n\n" + "The archive includes source, creator, license, attribution, and hash metadata. A binary model also requires a separately collected negative class.\n",
        encoding="utf-8",
    )
    preview = output / "active_barbecuing_preview_100.jpg"
    contact_sheet(items, preview)
    dataset_zip = output / "active_barbecuing_dataset_1000.zip"
    with zipfile.ZipFile(dataset_zip, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root))
    return dataset_zip, preview


def validate(archive_path: Path, expected: int) -> dict[str, Any]:
    image_count = 0
    hashes: set[str] = set()
    splits: Counter[str] = Counter()
    with zipfile.ZipFile(archive_path, "r") as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP corruption at {bad}")
        for entry in archive.infolist():
            if entry.is_dir() or Path(entry.filename).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            raw = archive.read(entry)
            with Image.open(io.BytesIO(raw)) as image:
                image.verify()
            digest = hashlib.sha256(raw).hexdigest()
            if digest in hashes:
                raise RuntimeError("exact duplicate in final archive")
            hashes.add(digest)
            image_count += 1
            splits[entry.filename.split("/", 1)[0]] += 1
    if image_count != expected:
        raise RuntimeError(f"expected {expected} images, found {image_count}")
    return {"archive": archive_path.name, "size_bytes": archive_path.stat().st_size, "image_count": image_count, "unique_sha256": len(hashes), "split_counts": dict(splits), "zip_integrity": "passed", "all_images_decode": True}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("fast_barbecue_output"))
    parser.add_argument("--per-query", type=int, default=400)
    parser.add_argument("--search-limit", type=int, default=2400)
    parser.add_argument("--metadata-limit", type=int, default=2100)
    parser.add_argument("--download-limit", type=int, default=1900)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--pause", type=float, default=0.55)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--min-side", type=int, default=320)
    parser.add_argument("--phash-limit", type=int, default=5)
    parser.add_argument("--dhash-limit", type=int, default=4)
    parser.add_argument("--creator-cap", type=int, default=35)
    parser.add_argument("--query-cap", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    downloads = args.output / "downloads"
    if downloads.exists():
        shutil.rmtree(downloads)
    downloads.mkdir()
    client, throttle = session(), Throttle(args.pause)
    try:
        hits = discover(client, throttle, args.per_query, args.search_limit)
        print(f"Search hits: {len(hits)}")
        items = metadata(client, throttle, hits, args.metadata_limit)
    finally:
        client.close()
    print(f"Metadata candidates: {len(items)}")
    if len(items) < args.target:
        raise RuntimeError(f"not enough metadata candidates: {len(items)}")
    with (args.output / "metadata_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    selected_for_download = items[:args.download_limit]
    completed: dict[int, Item] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_one, item, index, downloads, args.max_side, args.min_side) for index, item in enumerate(selected_for_download)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            index, item, error = future.result()
            if item is not None:
                completed[index] = item
            else:
                errors.append(error)
    downloaded = [completed[index] for index in sorted(completed)]
    (args.output / "download_errors.txt").write_text("\n".join(errors), encoding="utf-8")
    print(f"Decoded downloads: {len(downloaded)} failures: {len(errors)}")
    unique = deduplicate(downloaded, args.phash_limit, args.dhash_limit)
    unique.sort(key=lambda item: (item.score, -item.best_rank), reverse=True)
    print(f"Unique after hashing: {len(unique)}")
    if len(unique) < args.target:
        raise RuntimeError(f"only {len(unique)} unique images after deduplication")
    chosen = select(unique, args.target, args.creator_cap, args.query_cap)
    if len(chosen) != args.target:
        raise RuntimeError(f"only selected {len(chosen)} images")
    dataset_zip, preview = package(chosen, args.output, args.seed)
    summary = validate(dataset_zip, args.target)
    summary.update({"search_hits": len(hits), "metadata_candidates": len(items), "decoded_downloads": len(downloaded), "unique_after_deduplication": len(unique), "selected": len(chosen), "preview": preview.name, "license_counts": dict(Counter(item.license_name or "unspecified" for item in chosen)), "lowest_selected_metadata_score": min(item.score for item in chosen), "highest_selected_metadata_score": max(item.score for item in chosen)})
    (args.output / "build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output / "selected_metadata.jsonl").open("w", encoding="utf-8") as handle:
        for item in chosen:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
