#!/usr/bin/env python3
"""Broader source adapter for fast_barbecue_release.py."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import fast_barbecue_release as base

# Broad terms produce several thousand distinct Commons file hits. Specific
# action terms come first so their highly ranked results receive priority.
base.QUERIES[:] = [
    "people actively grilling barbecue",
    "person cooking on barbecue grill",
    "people grilling meat",
    "barbecue cooking people",
    "person grilling",
    "people grilling",
    "barbecue cook",
    "pitmaster barbecue",
    "outdoor grill cooking",
    "barbecue competition",
    "barbecue festival",
    "backyard barbecue",
    "park barbecue",
    "beach barbecue",
    "riverside barbecue",
    "lakeside barbecue",
    "tailgate grilling",
    "cookout grill",
    "barbecue",
    "barbeque",
    "barbecuing",
    "barbequing",
    "BBQ grilling",
    "grill cook",
    "braai",
    "churrasco",
    "churrasqueira cooking",
    "asado parrilla",
    "parrillero",
    "mangal barbecue",
    "shashlik grill",
    "kebab grilling",
    "satay grilling",
    "yakitori grilling",
    "Grillmeister",
    "grillen personen",
    "churrasqueiro",
    "parrillada personas",
    "barbecue personne",
    "焼肉 バーベキュー",
    "烧烤 人",
]


def metadata_v2(
    client,
    throttle,
    hits: list[dict[str, Any]],
    maximum: int,
) -> list[base.Item]:
    """Fetch image metadata with a less brittle multilingual relevance gate."""
    by_title = {row["title"]: row for row in hits}
    output: list[base.Item] = []
    source_seen: set[str] = set()
    batches = [hits[i : i + 50] for i in range(0, len(hits), 50)]

    for batch_index, batch in enumerate(batches, start=1):
        data = base.api_call(
            client,
            throttle,
            {
                "action": "query",
                "titles": "|".join(row["title"] for row in batch),
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata|sha1",
                "iiurlwidth": 640,
            },
        )

        for page in data.get("query", {}).get("pages", []):
            title = base.text(page.get("title", ""), 1000)
            hit = by_title.get(title)
            rows = page.get("imageinfo") or []
            if hit is None or not rows:
                continue

            info = rows[0]
            mime = base.text(info.get("mime", ""), 100).lower()
            width = base.integer(info.get("width"))
            height = base.integer(info.get("height"))
            if mime not in base.ALLOWED_MIME or min(width, height) < 256:
                continue

            source_sha1 = base.text(info.get("sha1", ""), 200)
            if source_sha1 and source_sha1 in source_seen:
                continue

            md = info.get("extmetadata") or {}
            description = base.meta(md, "ImageDescription")
            categories = base.meta(md, "Categories")
            object_name = base.meta(md, "ObjectName", 1000)
            real_text = f"{title} {description} {categories} {object_name}".lower()
            query_text = " ".join(hit["queries"]).lower()

            # Commons search already supplies barbecue relevance. Only reject
            # obvious non-photographic/product-like records here; ranking below
            # favors visible people and active cooking language.
            if sum(term in real_text for term in base.NEGATIVE_TERMS) >= 2:
                continue
            if not any(term in f"{real_text} {query_text}" for term in base.GRILL_TERMS):
                continue

            image_url = base.text(info.get("thumburl") or info.get("url"), 4000)
            if not image_url:
                continue

            item = base.Item(
                pageid=base.integer(page.get("pageid", hit["pageid"])),
                title=title,
                best_query=hit["best_query"],
                best_rank=hit["best_rank"],
                queries=list(hit["queries"]),
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
                source_sha1=source_sha1,
                width=width,
                height=height,
            )

            person_hits = sum(term in real_text for term in base.PERSON_TERMS)
            action_hits = sum(term in real_text for term in base.ACTION_TERMS)
            item.score = base.score_item(item)
            item.score += min(person_hits, 4) * 0.8
            item.score += min(action_hits, 4) * 0.75
            if person_hits == 0:
                item.score -= 0.45
            if action_hits == 0:
                item.score -= 0.30

            output.append(item)
            if source_sha1:
                source_seen.add(source_sha1)

        if batch_index % 10 == 0 or batch_index == len(batches):
            print(
                f"[metadata-v2] {batch_index}/{len(batches)} "
                f"accepted={len(output)}"
            )
        if len(output) >= maximum:
            break

    output.sort(
        key=lambda item: (item.score, -item.best_rank, item.width * item.height),
        reverse=True,
    )
    print("Top source-query counts:", Counter(item.best_query for item in output).most_common(12))
    return output[:maximum]


base.metadata = metadata_v2

if __name__ == "__main__":
    raise SystemExit(base.main())
