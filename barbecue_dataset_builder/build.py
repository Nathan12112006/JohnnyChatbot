#!/usr/bin/env python3
"""Build an openly licensed image dataset of people actively barbecuing.

The script queries the Openverse API, downloads source images, validates and
normalizes them, applies exact/perceptual/embedding deduplication, ranks them
with CLIP plus a person detector, and packages a classification-style dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import sys
import time
import urllib.parse
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import imagehash
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFile, ImageFont, ImageOps
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from urllib3.util.retry import Retry

ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = 80_000_000

OPENVERSE_ENDPOINT = "https://api.openverse.org/v1/images/"
USER_AGENT = (
    "ActiveBarbecuingDatasetBuilder/1.0 "
    "(research dataset builder; https://github.com/Nathan12112006/JohnnyChatbot)"
)
ALLOWED_LICENSES = {"cc0", "pdm", "by", "by-sa"}
ALLOWED_FILETYPES = {"jpg", "jpeg", "png", "webp"}

SEARCH_QUERIES = [
    "person actively grilling barbecue",
    "person cooking on barbecue grill",
    "man grilling barbecue",
    "woman grilling barbecue",
    "people barbecuing",
    "people barbequing",
    "barbecue cook",
    "BBQ cook grill",
    "pitmaster tending barbecue",
    "cook using outdoor grill",
    "person tending grill",
    "person using tongs grill",
    "person cooking meat on grill",
    "person grilling food outdoors",
    "family cooking barbecue",
    "friends grilling barbecue",
    "backyard barbecue cooking",
    "charcoal grill person cooking",
    "gas grill person cooking",
    "street barbecue vendor",
    "outdoor cook barbecue",
    "barbecue competition pitmaster",
    "barbecue festival cooking",
    "person grilling hamburgers",
    "person grilling kebabs",
    "campground barbecue cooking",
    "beach barbecue cooking",
    "lakeside barbecue cooking",
    "riverside barbecue cooking",
    "park barbecue grilling",
    "churrasco pessoa grelhando",
    "churrasqueiro churrasqueira",
    "asado parrilla cocinando persona",
    "parrillero cocinando asado",
    "Person grillen am Grill",
    "Grillmeister grillt",
    "personne cuisine barbecue",
    "cuisinier barbecue extérieur",
    "バーベキュー 焼く 人",
    "烧烤 烤肉 人",
]

POSITIVE_PROMPTS = [
    "a real photograph of a person actively cooking food on a barbecue grill",
    "a real photograph of someone standing at a grill and using cooking tools",
    "a real photograph of a pitmaster tending meat on a barbecue",
    "a real photograph of people grilling food outdoors",
    "a real photograph of a person cooking on a charcoal grill",
    "a real photograph of a person turning food with tongs on a grill",
]

NEGATIVE_PROMPTS = [
    "a photograph of an empty barbecue grill with nobody cooking",
    "a photograph of food on a grill with no person visible",
    "a photograph of people eating at a picnic without grilling",
    "a photograph of a campfire with people nearby",
    "a photograph of a restaurant meal",
    "a product photograph of a barbecue grill",
    "a photograph of a person merely posing next to a grill",
    "a kitchen stove indoors",
    "a drawing illustration cartoon or computer generated image",
    "a photograph unrelated to barbecue cooking",
]

POSITIVE_TERMS = {
    "barbecue", "barbeque", "barbecuing", "barbequing", "bbq", "grill",
    "grilling", "grilled", "pitmaster", "asado", "parrilla", "parrillero",
    "churrasco", "churrasqueira", "churrasqueiro", "grillen", "grillmeister",
    "烧烤", "烤肉", "バーベキュー",
}
ACTIVE_TERMS = {
    "cook", "cooking", "chef", "person", "people", "man", "woman", "family",
    "friends", "tending", "turning", "tongs", "vendor", "festival",
    "competition", "pitmaster", "cocinando", "persona", "pessoa", "grelhando",
    "cuisine", "cuisinier", "焼く", "人",
}
NEGATIVE_TERMS = {
    "isolated", "product", "cover", "logo", "icon", "illustration", "drawing",
    "cartoon", "render", "vector", "menu", "recipe", "clipart", "diagram",
    "sticker", "toy", "miniature", "empty", "unused", "packaging",
}
WATER_TERMS = {
    "water", "beach", "sea", "ocean", "lake", "lakeside", "river", "riverside",
    "stream", "creek", "pond", "pool", "waterfront", "harbor", "harbour",
    "canal", "shore", "coast", "coastal", "dock", "marina", "lagoon", "fjord",
}


@dataclass
class Candidate:
    openverse_id: str
    title: str
    creator: str
    creator_url: str
    license: str
    license_version: str
    license_url: str
    attribution: str
    provider: str
    source: str
    foreign_landing_url: str
    original_url: str
    thumbnail_url: str
    filetype: str
    width_reported: int
    height_reported: int
    query: str
    tags: str
    category: str
    api_rank: int
    download_order: int = 0
    local_path: str = ""
    downloaded_from: str = ""
    width: int = 0
    height: int = 0
    sha256: str = ""
    phash: str = ""
    dhash: str = ""
    metadata_score: float = 0.0
    clip_positive: float = 0.0
    clip_negative: float = 0.0
    clip_margin: float = 0.0
    person_confidence: float = 0.0
    final_score: float = 0.0
    confidence_tier: str = ""
    water_context: bool = False
    split: str = ""
    dataset_path: str = ""
    embedding: np.ndarray | None = field(default=None, repr=False, compare=False)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(value))
    return re.sub(r"\s+", " ", text).strip()


def tags_to_text(tags: Any) -> str:
    if not tags:
        return ""
    if isinstance(tags, list):
        values: list[str] = []
        for tag in tags:
            if isinstance(tag, dict):
                values.append(clean_text(tag.get("name", "")))
            else:
                values.append(clean_text(tag))
        return " | ".join(v for v in values if v)
    return clean_text(tags)


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def build_session(total_retries: int = 4) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json,image/avif,image/webp,image/*,*/*;q=0.8",
        }
    )
    return session


def query_openverse(
    session: requests.Session,
    query: str,
    page_size: int,
    page: int = 1,
) -> list[dict[str, Any]]:
    params = {
        "q": query,
        "license": ",".join(sorted(ALLOWED_LICENSES)),
        "categories": "photograph",
        "page_size": page_size,
        "page": page,
    }
    for requested_size in (page_size, 200, 100):
        params["page_size"] = requested_size
        response = session.get(OPENVERSE_ENDPOINT, params=params, timeout=90)
        if response.status_code == 400 and requested_size != 100:
            continue
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not isinstance(results, list):
            raise RuntimeError(f"Unexpected Openverse response for query {query!r}")
        return results
    return []


def collect_metadata(
    output_dir: Path,
    per_query: int,
    request_pause: float,
) -> list[Candidate]:
    session = build_session()
    per_query_results: list[list[Candidate]] = []
    seen_ids: set[str] = set()

    for query_index, query in enumerate(SEARCH_QUERIES, start=1):
        print(f"[Openverse] {query_index}/{len(SEARCH_QUERIES)}: {query}")
        try:
            results = query_openverse(session, query, page_size=max(per_query, 300))
        except Exception as exc:
            print(f"  query failed: {exc}", file=sys.stderr)
            per_query_results.append([])
            time.sleep(max(request_pause, 2.0))
            continue

        bucket: list[Candidate] = []
        for rank, item in enumerate(results[:per_query], start=1):
            oid = clean_text(item.get("id"))
            license_code = clean_text(item.get("license")).lower()
            filetype = clean_text(item.get("filetype")).lower().lstrip(".")
            if not oid or oid in seen_ids or license_code not in ALLOWED_LICENSES:
                continue
            if filetype and filetype not in ALLOWED_FILETYPES:
                continue
            original_url = clean_text(item.get("url"))
            thumbnail_url = clean_text(item.get("thumbnail"))
            landing_url = clean_text(item.get("foreign_landing_url"))
            if not original_url and not thumbnail_url:
                continue
            title = clean_text(item.get("title"))
            creator = clean_text(item.get("creator"))
            tags = tags_to_text(item.get("tags"))
            category = clean_text(item.get("category"))
            candidate = Candidate(
                openverse_id=oid,
                title=title,
                creator=creator,
                creator_url=clean_text(item.get("creator_url")),
                license=license_code,
                license_version=clean_text(item.get("license_version")),
                license_url=clean_text(item.get("license_url")),
                attribution=clean_text(item.get("attribution")),
                provider=clean_text(item.get("provider")),
                source=clean_text(item.get("source")),
                foreign_landing_url=landing_url,
                original_url=original_url,
                thumbnail_url=thumbnail_url,
                filetype=filetype,
                width_reported=safe_int(item.get("width")),
                height_reported=safe_int(item.get("height")),
                query=query,
                tags=tags,
                category=category,
                api_rank=rank,
            )
            candidate.metadata_score = compute_metadata_score(candidate)
            candidate.water_context = has_water_context(candidate)
            seen_ids.add(oid)
            bucket.append(candidate)
        per_query_results.append(bucket)
        print(f"  accepted metadata: {len(bucket)}")
        time.sleep(request_pause)

    interleaved: list[Candidate] = []
    max_len = max((len(bucket) for bucket in per_query_results), default=0)
    for index in range(max_len):
        for bucket in per_query_results:
            if index < len(bucket):
                interleaved.append(bucket[index])

    for order, candidate in enumerate(interleaved, start=1):
        candidate.download_order = order

    metadata_path = output_dir / "openverse_candidates.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for candidate in interleaved:
            handle.write(json.dumps(candidate_for_json(candidate), ensure_ascii=False) + "\n")
    print(f"Collected {len(interleaved)} unique Openverse records.")
    return interleaved


def candidate_text(candidate: Candidate) -> str:
    return " ".join(
        [candidate.title, candidate.tags, candidate.query, candidate.category]
    ).lower()


def compute_metadata_score(candidate: Candidate) -> float:
    text = candidate_text(candidate)
    positive_hits = sum(1 for term in POSITIVE_TERMS if term in text)
    active_hits = sum(1 for term in ACTIVE_TERMS if term in text)
    negative_hits = sum(1 for term in NEGATIVE_TERMS if term in text)
    rank_bonus = 1.0 / math.sqrt(max(candidate.api_rank, 1))
    score = (
        min(positive_hits, 4) * 0.22
        + min(active_hits, 3) * 0.12
        + rank_bonus * 0.18
        - min(negative_hits, 4) * 0.28
    )
    return round(score, 6)


def has_water_context(candidate: Candidate) -> bool:
    text = candidate_text(candidate)
    return any(term in text for term in WATER_TERMS)


def candidate_for_json(candidate: Candidate) -> dict[str, Any]:
    data = asdict(candidate)
    data.pop("embedding", None)
    return data


def preferred_urls(candidate: Candidate) -> list[str]:
    urls: list[str] = []
    for value in (candidate.original_url, candidate.thumbnail_url):
        if value and value not in urls and value.startswith(("http://", "https://")):
            urls.append(value)
    return urls


def read_limited_response(response: requests.Response, max_bytes: int) -> bytes:
    content_length = safe_int(response.headers.get("Content-Length"))
    if content_length and content_length > max_bytes:
        raise ValueError(f"Content-Length {content_length} exceeds limit")
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_content(chunk_size=128 * 1024):
        if not chunk:
            continue
        size += len(chunk)
        if size > max_bytes:
            raise ValueError("Downloaded payload exceeds byte limit")
        chunks.append(chunk)
    return b"".join(chunks)


def normalize_image(raw: bytes, max_side: int, min_side: int) -> tuple[bytes, int, int, str, str]:
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")
        width, height = image.size
        if min(width, height) < min_side:
            raise ValueError(f"Image too small: {width}x{height}")
        if width * height > Image.MAX_IMAGE_PIXELS:
            raise ValueError(f"Image too large: {width}x{height}")
        if max(width, height) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        width, height = image.size
        phash = str(imagehash.phash(image, hash_size=8))
        dhash = str(imagehash.dhash(image, hash_size=8))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True, progressive=True, subsampling=1)
        normalized = buffer.getvalue()
    return normalized, width, height, phash, dhash


def download_one(
    candidate: Candidate,
    raw_dir: Path,
    max_bytes: int,
    max_side: int,
    min_side: int,
) -> tuple[Candidate | None, str]:
    session = build_session(total_retries=2)
    errors: list[str] = []
    for url in preferred_urls(candidate):
        try:
            response = session.get(url, timeout=(20, 90), stream=True, allow_redirects=True)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if content_type and not (content_type.startswith("image/") or "octet-stream" in content_type):
                raise ValueError(f"Not an image content type: {content_type}")
            raw = read_limited_response(response, max_bytes=max_bytes)
            normalized, width, height, phash, dhash = normalize_image(raw, max_side=max_side, min_side=min_side)
            digest = hashlib.sha256(normalized).hexdigest()
            path = raw_dir / f"{candidate.openverse_id}.jpg"
            path.write_bytes(normalized)
            candidate.local_path = str(path)
            candidate.downloaded_from = response.url
            candidate.width = width
            candidate.height = height
            candidate.sha256 = digest
            candidate.phash = phash
            candidate.dhash = dhash
            return candidate, ""
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    return None, " || ".join(errors)[:2000]


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def perceptual_duplicate(
    candidate: Candidate,
    accepted: Sequence[Candidate],
    phash_threshold: int,
    dhash_threshold: int,
) -> bool:
    for prior in accepted:
        if (
            hamming_hex(candidate.phash, prior.phash) <= phash_threshold
            and hamming_hex(candidate.dhash, prior.dhash) <= dhash_threshold
        ):
            return True
        if hamming_hex(candidate.phash, prior.phash) <= max(2, phash_threshold - 3):
            return True
    return False


def download_candidates(
    candidates: list[Candidate],
    output_dir: Path,
    desired_valid: int,
    max_attempts: int,
    workers: int,
    max_bytes: int,
    max_side: int,
    min_side: int,
    phash_threshold: int,
    dhash_threshold: int,
) -> tuple[list[Candidate], Counter[str]]:
    raw_dir = output_dir / "candidates"
    raw_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[Candidate] = []
    seen_sha: set[str] = set()
    stats: Counter[str] = Counter()
    error_log = output_dir / "download_errors.jsonl"

    attempted_candidates = candidates[:max_attempts]
    with ThreadPoolExecutor(max_workers=workers) as executor, error_log.open("w", encoding="utf-8") as error_handle:
        future_map = {
            executor.submit(download_one, candidate, raw_dir, max_bytes, max_side, min_side): candidate
            for candidate in attempted_candidates
        }
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Downloading and validating"):
            candidate = future_map[future]
            try:
                downloaded, error = future.result()
            except Exception as exc:
                downloaded, error = None, f"{type(exc).__name__}: {exc}"

            if downloaded is None:
                stats["download_or_decode_failed"] += 1
                error_handle.write(json.dumps({"openverse_id": candidate.openverse_id, "title": candidate.title, "error": error}, ensure_ascii=False) + "\n")
                continue
            if downloaded.sha256 in seen_sha:
                stats["exact_duplicate"] += 1
                Path(downloaded.local_path).unlink(missing_ok=True)
                continue
            if perceptual_duplicate(downloaded, accepted, phash_threshold=phash_threshold, dhash_threshold=dhash_threshold):
                stats["perceptual_duplicate"] += 1
                Path(downloaded.local_path).unlink(missing_ok=True)
                continue

            seen_sha.add(downloaded.sha256)
            accepted.append(downloaded)
            stats["valid_unique"] += 1
            if len(accepted) >= desired_valid:
                for pending in future_map:
                    pending.cancel()
                break

    accepted_paths = {Path(c.local_path).resolve() for c in accepted}
    for path in raw_dir.glob("*.jpg"):
        if path.resolve() not in accepted_paths:
            path.unlink(missing_ok=True)

    print(f"Downloaded {len(accepted)} valid unique candidates.")
    return accepted, stats


def load_clip(model_name: str, device: torch.device) -> tuple[CLIPModel, CLIPProcessor, torch.Tensor]:
    print(f"Loading CLIP model: {model_name}")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.to(device)
    model.eval()
    text_inputs = processor(text=POSITIVE_PROMPTS + NEGATIVE_PROMPTS, return_tensors="pt", padding=True)
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    with torch.inference_mode():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return model, processor, text_features


def clip_score_candidates(
    candidates: list[Candidate],
    model_name: str,
    batch_size: int,
    device: torch.device,
) -> None:
    model, processor, text_features = load_clip(model_name, device)
    pos_count = len(POSITIVE_PROMPTS)

    for start in tqdm(range(0, len(candidates), batch_size), desc="CLIP scoring"):
        batch = candidates[start : start + batch_size]
        images: list[Image.Image] = []
        valid_batch: list[Candidate] = []
        for candidate in batch:
            try:
                with Image.open(candidate.local_path) as image:
                    images.append(image.convert("RGB").copy())
                valid_batch.append(candidate)
            except Exception:
                continue
        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            image_features = model.get_image_features(pixel_values=pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ text_features.T

        features_np = image_features.detach().cpu().numpy().astype(np.float32)
        sims_np = similarities.detach().cpu().numpy()
        for candidate, embedding, sims in zip(valid_batch, features_np, sims_np):
            candidate.embedding = embedding
            candidate.clip_positive = float(np.max(sims[:pos_count]))
            candidate.clip_negative = float(np.max(sims[pos_count:]))
            candidate.clip_margin = candidate.clip_positive - candidate.clip_negative

    del model, processor, text_features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_person_detector(device: torch.device):
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
    )

    print("Loading person detector.")
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    model.to(device)
    model.eval()
    return model


def score_person_presence(
    candidates: list[Candidate],
    batch_size: int,
    device: torch.device,
) -> None:
    from torchvision.transforms.functional import pil_to_tensor

    model = load_person_detector(device)
    for start in tqdm(range(0, len(candidates), batch_size), desc="Person detection"):
        batch = candidates[start : start + batch_size]
        tensors: list[torch.Tensor] = []
        valid_batch: list[Candidate] = []
        for candidate in batch:
            try:
                with Image.open(candidate.local_path) as image:
                    rgb = image.convert("RGB")
                    tensor = pil_to_tensor(rgb).float().div_(255.0)
                tensors.append(tensor.to(device))
                valid_batch.append(candidate)
            except Exception:
                continue
        if not tensors:
            continue
        with torch.inference_mode():
            outputs = model(tensors)
        for candidate, output in zip(valid_batch, outputs):
            labels = output.get("labels", torch.empty(0, device=device))
            scores = output.get("scores", torch.empty(0, device=device))
            person_scores = scores[labels == 1]
            candidate.person_confidence = float(person_scores.max().item()) if person_scores.numel() else 0.0
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def finalize_scores(candidates: list[Candidate]) -> None:
    for candidate in candidates:
        candidate.final_score = (
            5.0 * candidate.clip_margin
            + 0.75 * candidate.clip_positive
            + 0.85 * candidate.person_confidence
            + 0.35 * candidate.metadata_score
        )
        if candidate.person_confidence >= 0.72 and candidate.clip_margin >= 0.015:
            candidate.confidence_tier = "high"
        elif candidate.person_confidence >= 0.45 and candidate.clip_margin >= -0.005:
            candidate.confidence_tier = "medium"
        else:
            candidate.confidence_tier = "fallback"


def creator_key(candidate: Candidate) -> str:
    if candidate.creator.strip():
        return f"creator:{candidate.creator.strip().lower()}"
    if candidate.foreign_landing_url:
        parsed = urllib.parse.urlparse(candidate.foreign_landing_url)
        pieces = [p for p in parsed.path.split("/") if p]
        prefix = "/".join(pieces[:2])
        return f"url:{parsed.netloc.lower()}:{prefix.lower()}"
    return f"source:{candidate.source.lower()}:{candidate.openverse_id}"


def embedding_duplicate(embedding: np.ndarray, selected: Sequence[Candidate], threshold: float) -> bool:
    if not selected:
        return False
    for prior in selected:
        if prior.embedding is None:
            continue
        if float(np.dot(embedding, prior.embedding)) >= threshold:
            return True
    return False


def select_final(
    candidates: list[Candidate],
    target: int,
    min_person_confidence: float,
    embedding_similarity_threshold: float,
    max_per_creator: int,
    max_per_query: int,
) -> tuple[list[Candidate], Counter[str]]:
    candidates = [candidate for candidate in candidates if candidate.embedding is not None and candidate.person_confidence >= min_person_confidence]
    candidates.sort(
        key=lambda c: (
            c.confidence_tier == "high",
            c.confidence_tier == "medium",
            c.final_score,
            c.person_confidence,
            c.metadata_score,
        ),
        reverse=True,
    )

    selected: list[Candidate] = []
    creator_counts: Counter[str] = Counter()
    query_counts: Counter[str] = Counter()
    stats: Counter[str] = Counter()

    for candidate in candidates:
        key = creator_key(candidate)
        if creator_counts[key] >= max_per_creator:
            stats["creator_cap"] += 1
            continue
        if query_counts[candidate.query] >= max_per_query:
            stats["query_cap"] += 1
            continue
        assert candidate.embedding is not None
        if embedding_duplicate(candidate.embedding, selected, threshold=embedding_similarity_threshold):
            stats["embedding_near_duplicate"] += 1
            continue
        selected.append(candidate)
        creator_counts[key] += 1
        query_counts[candidate.query] += 1
        if len(selected) >= target:
            break

    if len(selected) < target:
        selected_ids = {c.openverse_id for c in selected}
        for candidate in candidates:
            if candidate.openverse_id in selected_ids:
                continue
            assert candidate.embedding is not None
            if embedding_duplicate(candidate.embedding, selected, threshold=embedding_similarity_threshold):
                continue
            selected.append(candidate)
            selected_ids.add(candidate.openverse_id)
            stats["relaxed_diversity_fill"] += 1
            if len(selected) >= target:
                break
    return selected, stats


def assign_grouped_splits(
    selected: list[Candidate],
    seed: int,
    train_fraction: float = 0.8,
    valid_fraction: float = 0.1,
) -> None:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in selected:
        grouped[creator_key(candidate)].append(candidate)

    rng = random.Random(seed)
    groups = list(grouped.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    targets = {"train": round(len(selected) * train_fraction), "valid": round(len(selected) * valid_fraction)}
    targets["test"] = len(selected) - targets["train"] - targets["valid"]
    counts = Counter({"train": 0, "valid": 0, "test": 0})

    for group in groups:
        split = min(("train", "valid", "test"), key=lambda name: counts[name] / max(targets[name], 1))
        for candidate in group:
            candidate.split = split
        counts[split] += len(group)


def copy_dataset_files(selected: list[Candidate], dataset_root: Path) -> None:
    split_order = {"train": 0, "valid": 1, "test": 2}
    ordered = sorted(selected, key=lambda c: (split_order.get(c.split, 9), -c.final_score, c.openverse_id))
    counters: Counter[str] = Counter()
    for candidate in ordered:
        counters[candidate.split] += 1
        filename = f"barbecue_{candidate.split}_{counters[candidate.split]:04d}.jpg"
        relative = Path(candidate.split) / "active_barbecuing" / filename
        destination = dataset_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate.local_path, destination)
        candidate.dataset_path = relative.as_posix()


def write_metadata(selected: list[Candidate], dataset_root: Path) -> None:
    columns = [
        "dataset_path", "split", "label", "openverse_id", "title", "creator",
        "creator_url", "license", "license_version", "license_url", "attribution",
        "provider", "source", "foreign_landing_url", "original_url", "downloaded_from",
        "query", "tags", "category", "width", "height", "sha256", "phash", "dhash",
        "metadata_score", "clip_positive", "clip_negative", "clip_margin",
        "person_confidence", "final_score", "confidence_tier", "water_context",
    ]
    with (dataset_root / "metadata.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for candidate in sorted(selected, key=lambda c: c.dataset_path):
            row = {key: getattr(candidate, key) for key in columns if key != "label"}
            row["label"] = "active_barbecuing"
            writer.writerow(row)

    with (dataset_root / "attributions.txt").open("w", encoding="utf-8") as handle:
        for candidate in sorted(selected, key=lambda c: c.dataset_path):
            attribution = candidate.attribution or (
                f"{candidate.title or 'Untitled'} by {candidate.creator or 'unknown creator'} "
                f"({candidate.license.upper()} {candidate.license_version})"
            )
            handle.write(f"{candidate.dataset_path}\n")
            handle.write(f"  {attribution}\n")
            handle.write(f"  Source: {candidate.foreign_landing_url}\n")
            handle.write(f"  License: {candidate.license_url}\n\n")


def write_readme(dataset_root: Path, selected: list[Candidate], requested_target: int) -> None:
    split_counts = Counter(candidate.split for candidate in selected)
    license_counts = Counter(candidate.license for candidate in selected)
    tier_counts = Counter(candidate.confidence_tier for candidate in selected)
    water_count = sum(1 for candidate in selected if candidate.water_context)
    text = f"""# Active Barbecuing Open-License Image Dataset

This package contains {len(selected)} real photographs collected through the
Openverse API and automatically selected for the visual concept
`active_barbecuing`.

## Inclusion rule

A positive image should visibly contain at least one person actively cooking,
tending food, or using tools at a barbecue/grill. A person merely standing near
a grill, an unattended grill, food-only closeups, product images, illustrations,
and ordinary picnics without visible cooking are outside the intended class.

## Layout

- `train/active_barbecuing/`: {split_counts['train']} images
- `valid/active_barbecuing/`: {split_counts['valid']} images
- `test/active_barbecuing/`: {split_counts['test']} images
- `metadata.csv`: source, creator, license, hashes, automated scores, and split
- `attributions.txt`: per-file attribution and source/license links

Requested target: {requested_target}
Delivered images: {len(selected)}
Images whose metadata suggests a waterside context: {water_count}

License counts: {dict(sorted(license_counts.items()))}
Automated confidence tiers: {dict(sorted(tier_counts.items()))}

## Important limitations

Selection used search metadata, CLIP image-text similarity, a COCO person
detector, exact hashing, perceptual hashing, and CLIP-embedding near-duplicate
suppression. These checks reduce obvious false positives and repeated images,
but automated review is not equivalent to human verification. Review the
contact sheets supplied beside the dataset before using it for a final
production training run.

Openverse aggregates licensing metadata supplied by source repositories and
advises users to verify licensing at each source page. The source page, creator,
license code, license URL, and attribution returned by Openverse are retained in
`metadata.csv` and `attributions.txt`. The package includes only records marked
CC0, public domain, CC BY, or CC BY-SA; ShareAlike obligations still apply to
CC BY-SA works.

This is a positive-image collection. A binary classifier also needs a
separately designed negative class, including difficult examples such as
unattended grills, people eating near grills, campfires, and food-only grill
images.
"""
    (dataset_root / "README.md").write_text(text, encoding="utf-8")


def make_contact_sheets(
    selected: list[Candidate],
    dataset_root: Path,
    review_dir: Path,
    per_sheet: int = 40,
    columns: int = 5,
    thumb_size: tuple[int, int] = (220, 165),
) -> None:
    review_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(selected, key=lambda c: c.dataset_path)
    rows = math.ceil(per_sheet / columns)
    label_height = 42
    cell_width = thumb_size[0]
    cell_height = thumb_size[1] + label_height
    font = ImageFont.load_default()

    index_rows: list[dict[str, Any]] = []
    for sheet_index, start in enumerate(range(0, len(ordered), per_sheet), start=1):
        batch = ordered[start : start + per_sheet]
        canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
        draw = ImageDraw.Draw(canvas)
        for position, candidate in enumerate(batch):
            row, column = divmod(position, columns)
            x = column * cell_width
            y = row * cell_height
            image_path = dataset_root / candidate.dataset_path
            with Image.open(image_path) as image:
                thumb = ImageOps.fit(image.convert("RGB"), thumb_size, method=Image.Resampling.LANCZOS)
            canvas.paste(thumb, (x, y))
            label = (
                f"{Path(candidate.dataset_path).name}\n"
                f"score={candidate.final_score:.3f} person={candidate.person_confidence:.2f} {candidate.confidence_tier}"
            )
            draw.multiline_text((x + 3, y + thumb_size[1] + 2), label, fill="black", font=font)
            index_rows.append({
                "sheet": sheet_index,
                "position": position + 1,
                "dataset_path": candidate.dataset_path,
                "openverse_id": candidate.openverse_id,
                "title": candidate.title,
                "source_page": candidate.foreign_landing_url,
            })
        canvas.save(review_dir / f"contact_sheet_{sheet_index:03d}.jpg", quality=88, optimize=True)

    with (review_dir / "contact_sheet_index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sheet", "position", "dataset_path", "openverse_id", "title", "source_page"])
        writer.writeheader()
        writer.writerows(index_rows)


def zip_directory(source_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir).as_posix())


def validate_package(
    selected: list[Candidate],
    dataset_root: Path,
    zip_path: Path,
    target: int,
) -> dict[str, Any]:
    image_paths = sorted(
        path for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    sha_values: set[str] = set()
    phashes: list[str] = []
    decode_failures: list[str] = []
    for path in tqdm(image_paths, desc="Final validation"):
        try:
            raw = path.read_bytes()
            sha_values.add(hashlib.sha256(raw).hexdigest())
            with Image.open(path) as image:
                image.load()
                phashes.append(str(imagehash.phash(image.convert("RGB"), hash_size=8)))
        except Exception as exc:
            decode_failures.append(f"{path}: {exc}")

    with zipfile.ZipFile(zip_path, "r") as archive:
        bad_member = archive.testzip()
        zip_entries = archive.namelist()

    summary = {
        "target_requested": target,
        "images_delivered": len(image_paths),
        "selected_records": len(selected),
        "all_images_decode": not decode_failures,
        "decode_failures": decode_failures,
        "unique_file_sha256": len(sha_values),
        "unique_phash_values": len(set(phashes)),
        "zip_exists": zip_path.exists(),
        "zip_size_bytes": zip_path.stat().st_size if zip_path.exists() else 0,
        "zip_integrity_ok": bad_member is None,
        "zip_entry_count": len(zip_entries),
        "split_counts": dict(Counter(candidate.split for candidate in selected)),
        "license_counts": dict(Counter(candidate.license for candidate in selected)),
        "confidence_tier_counts": dict(Counter(candidate.confidence_tier for candidate in selected)),
        "water_context_count": sum(1 for candidate in selected if candidate.water_context),
        "source_counts": dict(Counter(candidate.source for candidate in selected)),
        "provider_counts": dict(Counter(candidate.provider for candidate in selected)),
    }

    if len(image_paths) != len(selected):
        raise RuntimeError(f"Dataset image count {len(image_paths)} != selected count {len(selected)}")
    if len(selected) < target:
        raise RuntimeError(f"Only {len(selected)} images selected; target was {target}")
    if decode_failures:
        raise RuntimeError(f"Final decode failures: {decode_failures[:5]}")
    if len(sha_values) != len(image_paths):
        raise RuntimeError("Exact duplicate normalized files detected in final dataset")
    if bad_member is not None:
        raise RuntimeError(f"ZIP integrity failure at {bad_member}")
    return summary


def write_ranked_candidates(candidates: list[Candidate], output_path: Path) -> None:
    columns = [
        "openverse_id", "title", "creator", "license", "source", "foreign_landing_url",
        "query", "width", "height", "metadata_score", "clip_positive", "clip_negative",
        "clip_margin", "person_confidence", "final_score", "confidence_tier",
        "water_context", "sha256", "phash", "dhash",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for candidate in sorted(candidates, key=lambda c: c.final_score, reverse=True):
            writer.writerow({key: getattr(candidate, key) for key in columns})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("barbecue_dataset_output"))
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--per-query", type=int, default=350)
    parser.add_argument("--request-pause", type=float, default=1.0)
    parser.add_argument("--desired-valid-candidates", type=int, default=5200)
    parser.add_argument("--max-download-attempts", type=int, default=9000)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--max-download-mb", type=int, default=24)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--min-side", type=int, default=480)
    parser.add_argument("--phash-threshold", type=int, default=6)
    parser.add_argument("--dhash-threshold", type=int, default=5)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--person-batch-size", type=int, default=8)
    parser.add_argument("--person-pool", type=int, default=3600)
    parser.add_argument("--min-person-confidence", type=float, default=0.24)
    parser.add_argument("--embedding-similarity-threshold", type=float, default=0.988)
    parser.add_argument("--max-per-creator", type=int, default=18)
    parser.add_argument("--max-per-query", type=int, default=90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(os.cpu_count() or 2, 8)))

    output_dir = args.output.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = collect_metadata(output_dir, per_query=args.per_query, request_pause=args.request_pause)
    if len(candidates) < args.target * 2:
        raise RuntimeError(f"Only {len(candidates)} metadata candidates found; not enough for target {args.target}")

    downloaded, download_stats = download_candidates(
        candidates,
        output_dir=output_dir,
        desired_valid=args.desired_valid_candidates,
        max_attempts=args.max_download_attempts,
        workers=args.download_workers,
        max_bytes=args.max_download_mb * 1024 * 1024,
        max_side=args.max_side,
        min_side=args.min_side,
        phash_threshold=args.phash_threshold,
        dhash_threshold=args.dhash_threshold,
    )
    if len(downloaded) < args.target * 2:
        raise RuntimeError(
            f"Only {len(downloaded)} valid unique images downloaded; need a larger pool for target {args.target}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")
    clip_score_candidates(downloaded, model_name=args.clip_model, batch_size=args.clip_batch_size, device=device)
    finalize_scores(downloaded)

    preliminary = sorted(
        downloaded,
        key=lambda c: (c.clip_margin + 0.25 * c.metadata_score, c.clip_positive),
        reverse=True,
    )
    person_pool = preliminary[: args.person_pool]
    water_reserve = [c for c in preliminary[args.person_pool :] if c.water_context][:300]
    seen_pool = {c.openverse_id for c in person_pool}
    person_pool.extend(c for c in water_reserve if c.openverse_id not in seen_pool)
    score_person_presence(person_pool, batch_size=args.person_batch_size, device=device)
    finalize_scores(person_pool)
    write_ranked_candidates(person_pool, output_dir / "ranked_review_candidates.csv")

    selected, selection_stats = select_final(
        person_pool,
        target=args.target,
        min_person_confidence=args.min_person_confidence,
        embedding_similarity_threshold=args.embedding_similarity_threshold,
        max_per_creator=args.max_per_creator,
        max_per_query=args.max_per_query,
    )
    if len(selected) < args.target:
        raise RuntimeError(f"Only {len(selected)} images passed selection; target was {args.target}")

    assign_grouped_splits(selected, seed=args.seed)
    dataset_root = output_dir / "active_barbecuing_open_license_dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    copy_dataset_files(selected, dataset_root)
    write_metadata(selected, dataset_root)
    write_readme(dataset_root, selected, requested_target=args.target)

    review_root = output_dir / "review"
    make_contact_sheets(selected, dataset_root, review_root)

    dataset_zip = output_dir / f"active_barbecuing_open_license_{args.target}.zip"
    zip_directory(dataset_root, dataset_zip)
    review_zip = output_dir / f"active_barbecuing_review_{args.target}.zip"
    zip_directory(review_root, review_zip)

    summary = validate_package(selected, dataset_root=dataset_root, zip_path=dataset_zip, target=args.target)
    summary["download_stats"] = dict(download_stats)
    summary["selection_stats"] = dict(selection_stats)
    summary["parameters"] = vars(args) | {"output": str(args.output)}
    (output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    shutil.rmtree(dataset_root)
    shutil.rmtree(review_root)
    shutil.rmtree(output_dir / "candidates", ignore_errors=True)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Dataset ZIP: {dataset_zip}")
    print(f"Review ZIP: {review_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
