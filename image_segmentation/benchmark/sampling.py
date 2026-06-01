from __future__ import annotations

from collections import Counter
import random


def sample_manifest_rows(
    rows: list[dict],
    max_samples: int | None,
    seed: int = 42,
    category_balance: bool = True,
    image_diversity: bool = False,
    max_instances_per_image: int | None = None,
) -> list[dict]:
    if max_samples is None or max_samples >= len(rows):
        sampled = list(rows)
        return _cap_instances_per_image(sampled, max_instances_per_image) if image_diversity else sampled
    rng = random.Random(seed)
    if not category_balance:
        sampled = list(rows)
        rng.shuffle(sampled)
        if image_diversity:
            sampled = _cap_instances_per_image(sampled, max_instances_per_image)
        return sampled[:max_samples]
    by_category: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(str(row.get("category_id")), []).append(row)
    selected: list[dict] = []
    image_counts: Counter[str] = Counter()
    category_keys = sorted(by_category)
    while len(selected) < max_samples and category_keys:
        progressed = False
        for key in list(category_keys):
            bucket = by_category[key]
            if not bucket:
                category_keys.remove(key)
                continue
            pick_index = _pick_index(bucket, rng, image_counts, max_instances_per_image if image_diversity else None)
            if pick_index is None:
                category_keys.remove(key)
                continue
            row = bucket.pop(pick_index)
            selected.append(row)
            image_counts[str(row.get("image_id"))] += 1
            progressed = True
            if len(selected) >= max_samples:
                break
        if not progressed:
            break
    return selected


def _pick_index(
    bucket: list[dict],
    rng: random.Random,
    image_counts: Counter[str],
    max_instances_per_image: int | None,
) -> int | None:
    if max_instances_per_image is None:
        return rng.randrange(len(bucket))
    eligible = [
        index
        for index, row in enumerate(bucket)
        if image_counts[str(row.get("image_id"))] < int(max_instances_per_image)
    ]
    if not eligible:
        return None
    return eligible[rng.randrange(len(eligible))]


def _cap_instances_per_image(rows: list[dict], max_instances_per_image: int | None) -> list[dict]:
    if max_instances_per_image is None:
        return rows
    counts: Counter[str] = Counter()
    out = []
    for row in rows:
        image_id = str(row.get("image_id"))
        if counts[image_id] >= int(max_instances_per_image):
            continue
        counts[image_id] += 1
        out.append(row)
    return out
