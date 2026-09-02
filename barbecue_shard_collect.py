#!/usr/bin/env python3
"""Collect one independent shard of an active-barbecuing image dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import imagehash
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

import fast_barbecue_release as base
import fast_barbecue_release_v3 as combined


MASTER_QUERIES = [
    "person grilling filetype:bitmap",
    "people grilling filetype:bitmap",
    "man grilling barbecue filetype:bitmap",
    "woman grilling barbecue filetype:bitmap",
    "family barbecue cooking filetype:bitmap",
    "friends barbecue cooking filetype:bitmap",
    "cook using barbecue grill filetype:bitmap",
    "chef cooking on grill filetype:bitmap",
    "pitmaster tending barbecue filetype:bitmap",
    "outdoor grill cooking people filetype:bitmap",
    "barbecue cooking filetype:bitmap",
    "barbecue cook filetype:bitmap",
    "people barbecuing filetype:bitmap",
    "people barbequing filetype:bitmap",
    "barbecue tongs person filetype:bitmap",
    "turning food on grill filetype:bitmap",
    "preparing food barbecue filetype:bitmap",
    "charcoal grill cooking person filetype:bitmap",
    "gas grill cooking person filetype:bitmap",
    "smoker barbecue pitmaster filetype:bitmap",
    "grilling hamburgers person filetype:bitmap",
    "grilling cheeseburgers person filetype:bitmap",
    "grilling sausages person filetype:bitmap",
    "grilling hot dogs person filetype:bitmap",
    "grilling steak person filetype:bitmap",
    "grilling beef person filetype:bitmap",
    "grilling pork person filetype:bitmap",
    "grilling ribs person filetype:bitmap",
    "grilling chicken person filetype:bitmap",
    "grilling turkey person filetype:bitmap",
    "grilling fish person filetype:bitmap",
    "grilling seafood person filetype:bitmap",
    "grilling shrimp person filetype:bitmap",
    "grilling vegetables person filetype:bitmap",
    "grilling corn person filetype:bitmap",
    "grilling kebabs person filetype:bitmap",
    "grilling skewers vendor filetype:bitmap",
    "satay grilling vendor filetype:bitmap",
    "yakitori grilling cook filetype:bitmap",
    "shashlik grilling people filetype:bitmap",
    "barbecue competition cooking filetype:bitmap",
    "barbecue contest pitmaster filetype:bitmap",
    "barbecue festival grilling filetype:bitmap",
    "food festival grilling vendor filetype:bitmap",
    "street food grilling vendor filetype:bitmap",
    "market grilling cook filetype:bitmap",
    "fair barbecue cooking filetype:bitmap",
    "community barbecue cooking filetype:bitmap",
    "fundraiser barbecue cooking filetype:bitmap",
    "church barbecue cooking filetype:bitmap",
    "backyard barbecue cooking filetype:bitmap",
    "garden barbecue cooking filetype:bitmap",
    "park barbecue cooking filetype:bitmap",
    "picnic barbecue cooking filetype:bitmap",
    "campground barbecue cooking filetype:bitmap",
    "camping grill cooking filetype:bitmap",
    "beach barbecue cooking filetype:bitmap",
    "lakeside barbecue cooking filetype:bitmap",
    "riverside barbecue cooking filetype:bitmap",
    "waterfront barbecue cooking filetype:bitmap",
    "tailgate grilling people filetype:bitmap",
    "sports tailgate barbecue filetype:bitmap",
    "cookout grill people filetype:bitmap",
    "summer barbecue cooking filetype:bitmap",
    "winter barbecue cooking filetype:bitmap",
    "holiday barbecue cooking filetype:bitmap",
    "birthday barbecue cooking filetype:bitmap",
    "wedding barbecue cooking filetype:bitmap",
    "restaurant outdoor grill cook filetype:bitmap",
    "catering barbecue cook filetype:bitmap",
    "mobile barbecue vendor filetype:bitmap",
    "food truck grilling cook filetype:bitmap",
    "roadside barbecue vendor filetype:bitmap",
    "open fire grill cooking person filetype:bitmap",
    "wood fire barbecue cooking filetype:bitmap",
    "rotisserie barbecue cook filetype:bitmap",
    "whole hog barbecue pitmaster filetype:bitmap",
    "barbecue smoker cook filetype:bitmap",
    "kettle grill cooking filetype:bitmap",
    "hibachi outdoor cook filetype:bitmap",
    "braai cooking people filetype:bitmap",
    "braai master cooking filetype:bitmap",
    "South African braai people filetype:bitmap",
    "churrasco grilling people filetype:bitmap",
    "churrasqueiro churrasco filetype:bitmap",
    "Brazil barbecue cook filetype:bitmap",
    "asado parrilla people filetype:bitmap",
    "parrillero cooking asado filetype:bitmap",
    "Argentina asado cook filetype:bitmap",
    "mangal cooking people filetype:bitmap",
    "Turkish mangal barbecue filetype:bitmap",
    "kebab grilling cook filetype:bitmap",
    "shawarma grill cook filetype:bitmap",
    "tandoor grill cook outdoors filetype:bitmap",
    "grillen menschen filetype:bitmap",
    "Grillmeister grillt filetype:bitmap",
    "grillfest kochen filetype:bitmap",
    "barbecue personne cuisine filetype:bitmap",
    "personnes grillades barbecue filetype:bitmap",
    "cuisinier barbecue extérieur filetype:bitmap",
    "grigliata persone filetype:bitmap",
    "cuoco barbecue filetype:bitmap",
    "barbecue famiglia cucina filetype:bitmap",
    "parrillada personas filetype:bitmap",
    "persona cocinando parrilla filetype:bitmap",
    "carne asada cocinero filetype:bitmap",
    "churrasco pessoa grelhando filetype:bitmap",
    "pessoa churrasqueira filetype:bitmap",
    "шашлык готовят filetype:bitmap",
    "люди жарят мясо filetype:bitmap",
    "барбекю повар filetype:bitmap",
    "バーベキュー 人 焼く filetype:bitmap",
    "焼肉 グリル 人 filetype:bitmap",
    "屋外 バーベキュー 料理 filetype:bitmap",
    "烧烤 烤肉 人 filetype:bitmap",
    "户外 烧烤 人 filetype:bitmap",
    "烧烤 厨师 filetype:bitmap",
    "바베큐 굽는 사람 filetype:bitmap",
    "야외 바베큐 요리 filetype:bitmap",
    "mangal yapan insanlar filetype:bitmap",
]


def make_download_session() -> requests.Session:
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    client = requests.Session()
    client.headers.update({"User-Agent": base.USER_AGENT})
    client.mount(
        "https://",
        HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4),
    )
    return client


def collect_items(queries: list[str], per_query: int, pause: float) -> list[base.Item]:
    client = base.session()
    throttle = base.Throttle(pause)
    by_pageid: dict[int, base.Item] = {}
    sha_seen: set[str] = set()
    try:
        for query_index, query in enumerate(queries, start=1):
            try:
                data = base.api_call(
                    client,
                    throttle,
                    {
                        "action": "query",
                        "generator": "search",
                        "gsrsearch": query,
                        "gsrnamespace": 6,
                        "gsrlimit": min(max(per_query, 1), 500),
                        "prop": "imageinfo",
                        "iiprop": "url|size|mime|extmetadata|sha1",
                        "iiurlwidth": 720,
                    },
                )
            except Exception as exc:
                print(f"[query {query_index}] failed {query!r}: {exc}", file=sys.stderr)
                continue

            pages = data.get("query", {}).get("pages", [])
            pages = sorted(
                pages,
                key=lambda row: base.integer(row.get("index")) or 999999,
            )
            added = 0
            for fallback_rank, page in enumerate(pages, start=1):
                item = combined.build_item(page, query, fallback_rank)
                if item is None:
                    continue
                if item.source_sha1 and item.source_sha1 in sha_seen:
                    continue
                if item.pageid in by_pageid:
                    continue
                by_pageid[item.pageid] = item
                if item.source_sha1:
                    sha_seen.add(item.source_sha1)
                added += 1
            print(
                f"[query {query_index:02d}/{len(queries)}] "
                f"pages={len(pages)} added={added} total={len(by_pageid)}"
            )
    finally:
        client.close()

    return sorted(
        by_pageid.values(),
        key=lambda item: (
            item.score,
            len(item.queries),
            -item.best_rank,
            item.width * item.height,
        ),
        reverse=True,
    )


def candidate_urls(item: base.Item, max_side: int) -> list[tuple[str, dict[str, str] | None]]:
    parsed = urllib.parse.urlsplit(item.image_url)
    wp_path = f"{parsed.netloc}{parsed.path}"
    if parsed.query:
        wp_path += f"?{parsed.query}"
    file_name = item.title.removeprefix("File:")
    redirect_url = (
        "https://commons.wikimedia.org/wiki/Special:Redirect/file/"
        + urllib.parse.quote(file_name, safe="")
    )
    return [
        (
            "https://wsrv.nl/",
            {
                "url": item.image_url,
                "w": str(max_side),
                "output": "jpg",
                "q": "84",
                "we": "",
            },
        ),
        (
            "https://images.weserv.nl/",
            {
                "url": item.image_url,
                "w": str(max_side),
                "output": "jpg",
                "q": "84",
                "we": "",
            },
        ),
        (
            f"https://i0.wp.com/{wp_path}",
            {"w": str(max_side), "quality": "84", "strip": "all"},
        ),
        (
            "https://external-content.duckduckgo.com/iu/",
            {"u": item.image_url, "f": "1", "nofb": "1"},
        ),
        (
            "https://images1-focus-opensocial.googleusercontent.com/gadgets/proxy",
            {
                "url": item.image_url,
                "container": "focus",
                "refresh": "2592000",
            },
        ),
        (redirect_url, {"width": str(max_side)}),
        (item.image_url, None),
    ]


def download_one(
    item: base.Item,
    index: int,
    folder: Path,
    max_side: int,
    min_side: int,
) -> tuple[int, base.Item | None, str]:
    client = make_download_session()
    errors: list[str] = []
    try:
        for url, params in candidate_urls(item, max_side):
            try:
                response = client.get(
                    url,
                    params=params,
                    timeout=(12, 55),
                    stream=True,
                    headers={"Referer": "https://commons.wikimedia.org/"},
                )
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "image" not in content_type and url != item.image_url:
                    raise ValueError(f"non-image content type {content_type}")
                data = bytearray()
                for chunk in response.iter_content(262144):
                    if not chunk:
                        continue
                    data.extend(chunk)
                    if len(data) > 20_000_000:
                        raise ValueError("image too large")
                normalized = base.normalize(bytes(data), max_side, min_side)
                path = folder / f"candidate_{index:05d}.jpg"
                path.write_bytes(normalized)
                item.local_path = str(path)
                item.sha256 = hashlib.sha256(normalized).hexdigest()
                with Image.open(io.BytesIO(normalized)) as image:
                    item.width, item.height = image.size
                    item.phash = str(imagehash.phash(image, hash_size=8))
                    item.dhash = str(imagehash.dhash(image, hash_size=8))
                return index, item, ""
            except Exception as exc:
                errors.append(f"{urllib.parse.urlsplit(url).netloc}:{type(exc).__name__}")
        return index, None, f"{item.title}\t{' | '.join(errors)}"
    finally:
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-query", type=int, default=300)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--pause", type=float, default=0.35)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--min-side", type=int, default=160)
    parser.add_argument("--phash-limit", type=int, default=3)
    parser.add_argument("--dhash-limit", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")

    queries = [
        query
        for query_index, query in enumerate(MASTER_QUERIES)
        if query_index % args.shard_count == args.shard_index
    ]
    print(f"Shard {args.shard_index}/{args.shard_count}: {len(queries)} queries")
    items = collect_items(queries, args.per_query, args.pause)
    print(f"Metadata candidates: {len(items)}")

    work = args.output / f"work_{args.shard_index:02d}"
    final = args.output / f"shard_{args.shard_index:02d}"
    if work.exists():
        shutil.rmtree(work)
    if final.exists():
        shutil.rmtree(final)
    work.mkdir(parents=True)
    (final / "images").mkdir(parents=True)

    completed: dict[int, base.Item] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_one,
                item,
                index,
                work,
                args.max_side,
                args.min_side,
            )
            for index, item in enumerate(items)
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Downloading shard {args.shard_index}",
        ):
            index, item, error = future.result()
            if item is not None:
                completed[index] = item
            elif error:
                errors.append(error)

    downloaded = [completed[index] for index in sorted(completed)]
    unique = base.deduplicate(downloaded, args.phash_limit, args.dhash_limit)
    unique.sort(key=lambda item: (item.score, -item.best_rank), reverse=True)

    with (final / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for accepted_index, item in enumerate(unique):
            name = f"shard_{args.shard_index:02d}_{accepted_index:04d}_{item.pageid}.jpg"
            destination = final / "images" / name
            shutil.copy2(item.local_path, destination)
            record = asdict(item)
            record["shard_file"] = f"images/{name}"
            record["local_path"] = ""
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    (final / "errors.txt").write_text("\n".join(errors), encoding="utf-8")
    summary = {
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "query_count": len(queries),
        "metadata_candidates": len(items),
        "decoded_downloads": len(downloaded),
        "unique_images": len(unique),
        "failed_downloads": len(errors),
    }
    (final / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
