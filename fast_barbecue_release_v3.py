#!/usr/bin/env python3
"""API-efficient Commons collector for the 1,000-image barbecue release."""

from __future__ import annotations

from collections import Counter
from typing import Any

import fast_barbecue_release as base

QUERIES = [
    "people grilling barbecue",
    "person grilling",
    "people grilling",
    "barbecue cooking",
    "barbecue cook",
    "barbecue festival",
    "barbecue competition",
    "pitmaster barbecue",
    "outdoor grill cooking",
    "backyard barbecue",
    "park barbecue",
    "beach barbecue",
    "barbecue",
    "barbeque",
    "barbecuing",
    "barbequing",
    "BBQ grilling",
    "grill cook",
    "braai",
    "churrasco",
    "asado parrilla",
    "mangal barbecue",
    "kebab grilling",
    "satay grilling",
]

COLLECTED: list[base.Item] = []


def build_item(page: dict[str, Any], query: str, fallback_rank: int) -> base.Item | None:
    rows = page.get("imageinfo") or []
    if not rows:
        return None
    info = rows[0]
    mime = base.text(info.get("mime", ""), 100).lower()
    width = base.integer(info.get("width"))
    height = base.integer(info.get("height"))
    if mime not in base.ALLOWED_MIME or min(width, height) < 256:
        return None

    md = info.get("extmetadata") or {}
    title = base.text(page.get("title", ""), 1000)
    description = base.meta(md, "ImageDescription")
    categories = base.meta(md, "Categories")
    object_name = base.meta(md, "ObjectName", 1000)
    real_text = f"{title} {description} {categories} {object_name}".lower()
    if sum(term in real_text for term in base.NEGATIVE_TERMS) >= 2:
        return None
    image_url = base.text(info.get("thumburl") or info.get("url"), 4000)
    if not image_url:
        return None

    item = base.Item(
        pageid=base.integer(page.get("pageid")),
        title=title,
        best_query=query,
        best_rank=base.integer(page.get("index")) or fallback_rank,
        queries=[query],
        description=description,
        categories=categories,
        creator=base.meta(md, "Artist", 1600),
        creator_url=base.meta(md, "ArtistProfile", 2000),
        attribution=base.meta(md, "Attribution", 2500)
        or base.meta(md, "Credit", 2500),
        license_name=base.meta(md, "LicenseShortName", 400)
        or base.meta(md, "UsageTerms", 400),
        license_url=base.meta(md, "LicenseUrl", 2000)
        or base.meta(md, "License", 2000),
        source_page=base.text(info.get("descriptionurl"), 4000),
        image_url=image_url,
        source_sha1=base.text(info.get("sha1", ""), 200),
        width=width,
        height=height,
    )
    person_hits = sum(term in real_text for term in base.PERSON_TERMS)
    action_hits = sum(term in real_text for term in base.ACTION_TERMS)
    item.score = base.score_item(item)
    item.score += min(person_hits, 4) * 0.85
    item.score += min(action_hits, 4) * 0.80
    if person_hits == 0:
        item.score -= 0.55
    if action_hits == 0:
        item.score -= 0.35
    return item


def combined_discover(client, throttle, per_query: int, maximum: int):
    global COLLECTED
    by_pageid: dict[int, base.Item] = {}
    sha_to_pageid: dict[str, int] = {}

    for query_index, query in enumerate(QUERIES, start=1):
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
                "iiurlwidth": 640,
            },
        )
        pages = data.get("query", {}).get("pages", [])
        pages = sorted(pages, key=lambda row: base.integer(row.get("index")) or 999999)
        added = 0
        for fallback_rank, page in enumerate(pages, start=1):
            item = build_item(page, query, fallback_rank)
            if item is None:
                continue
            if item.source_sha1 and item.source_sha1 in sha_to_pageid:
                existing = by_pageid.get(sha_to_pageid[item.source_sha1])
                if existing is not None and query not in existing.queries:
                    existing.queries.append(query)
                    existing.score = base.score_item(existing)
                continue
            existing = by_pageid.get(item.pageid)
            if existing is not None:
                if query not in existing.queries:
                    existing.queries.append(query)
                    existing.score = base.score_item(existing)
                continue
            by_pageid[item.pageid] = item
            if item.source_sha1:
                sha_to_pageid[item.source_sha1] = item.pageid
            added += 1
        print(
            f"[combined {query_index:02d}/{len(QUERIES)}] {query!r}: "
            f"pages={len(pages)} added={added} total={len(by_pageid)}"
        )
        if len(by_pageid) >= maximum:
            break

    COLLECTED = sorted(
        by_pageid.values(),
        key=lambda item: (item.score, len(item.queries), -item.best_rank, item.width * item.height),
        reverse=True,
    )[:maximum]
    print("Combined source-query counts:", Counter(item.best_query for item in COLLECTED).most_common(12))
    return COLLECTED


def combined_metadata(client, throttle, hits, maximum: int):
    del client, throttle, hits
    return COLLECTED[:maximum]


base.discover = combined_discover
base.metadata = combined_metadata

if __name__ == "__main__":
    raise SystemExit(base.main())
