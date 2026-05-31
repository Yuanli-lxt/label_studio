from __future__ import annotations

import random


def sample_manifest_rows(rows: list[dict], max_samples: int | None, seed: int = 42) -> list[dict]:
    if max_samples is None or max_samples >= len(rows):
        return rows
    rng = random.Random(seed)
    by_category: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(str(row.get("category_id")), []).append(row)
    selected: list[dict] = []
    category_keys = sorted(by_category)
    while len(selected) < max_samples and category_keys:
        progressed = False
        for key in list(category_keys):
            bucket = by_category[key]
            if not bucket:
                category_keys.remove(key)
                continue
            selected.append(bucket.pop(rng.randrange(len(bucket))))
            progressed = True
            if len(selected) >= max_samples:
                break
        if not progressed:
            break
    return selected

