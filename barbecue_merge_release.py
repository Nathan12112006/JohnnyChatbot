#!/usr/bin/env python3
"""Merge independent barbecue image shards into a validated 1,000-image release."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

import fast_barbecue_release as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--phash-limit", type=int, default=4)
    parser.add_argument("--dhash-limit", type=int, default=4)
    parser.add_argument("--creator-cap", type=int, default=45)
    parser.add_argument("--query-cap", type=int, default=220)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def load_items(root: Path) -> tuple[list[base.Item], list[dict]]:
    items: list[base.Item] = []
    shard_summaries: list[dict] = []
    for summary_path in sorted(root.rglob("summary.json")):
        try:
            shard_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        except Exception:
            pass

    for metadata_path in sorted(root.rglob("metadata.jsonl")):
        shard_root = metadata_path.parent
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                shard_file = record.pop("shard_file")
                record["local_path"] = str(shard_root / shard_file)
                item = base.Item(**record)
                path = Path(item.local_path)
                if not path.exists() or path.stat().st_size == 0:
                    continue
                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception:
                    continue
                items.append(item)
    return items, shard_summaries


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    loaded, shard_summaries = load_items(args.input)
    print(f"Loaded {len(loaded)} decodable shard images.")
    if len(loaded) < args.target:
        raise RuntimeError(
            f"Only {len(loaded)} shard images available for target {args.target}"
        )

    unique = base.deduplicate(
        loaded,
        phash_limit=args.phash_limit,
        dhash_limit=args.dhash_limit,
    )
    unique.sort(
        key=lambda item: (
            item.score,
            len(item.queries),
            -item.best_rank,
            item.width * item.height,
        ),
        reverse=True,
    )
    print(f"After cross-shard deduplication: {len(unique)}")
    if len(unique) < args.target:
        raise RuntimeError(
            f"Only {len(unique)} unique images after cross-shard deduplication"
        )

    chosen = base.select(
        unique,
        args.target,
        args.creator_cap,
        args.query_cap,
    )
    if len(chosen) != args.target:
        raise RuntimeError(f"Only selected {len(chosen)} images")

    dataset_zip, preview = base.package(chosen, args.output, args.seed)
    validation = base.validate(dataset_zip, args.target)
    validation.update(
        {
            "source": "Wikimedia Commons",
            "collection_method": "independent sharded runners with multi-route image retrieval",
            "loaded_shard_images": len(loaded),
            "unique_after_cross_shard_deduplication": len(unique),
            "selected_images": len(chosen),
            "preview": preview.name,
            "shard_count": len(shard_summaries),
            "shard_summaries": shard_summaries,
            "license_counts": dict(
                Counter(item.license_name or "unspecified" for item in chosen)
            ),
            "lowest_selected_score": min(item.score for item in chosen),
            "highest_selected_score": max(item.score for item in chosen),
        }
    )
    (args.output / "build_summary.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (args.output / "selected_metadata.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for item in chosen:
            handle.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")

    print(json.dumps(validation, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
