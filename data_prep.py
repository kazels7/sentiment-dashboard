#!/usr/bin/env python3
"""
data_prep.py — Prepare a balanced 3-class subsample from the Amazon Reviews '23
\"Sports_and_Outdoors\" raw review file.

Pipeline
--------
1. Stream the .jsonl.gz line at a time (never loads the whole file into memory).
2. Keep only the fields we need: rating, title, text, asin.
3. Classify:  rating >= 4  ->  positive ;  rating == 3  ->  neutral ;
             rating <= 2  ->  negative.
4. Reservoir-sample exactly --per-class items from each class so the subsample is
   balanced and uniform without knowing the file size.
5. Shuffle the combined sample with a fixed seed (deterministic across runs).
6. Write the sample to --out as JSONL.

Usage
-----
    python data_prep.py --src data/Sports_and_Outdoors.jsonl.gz \
        --out data/balanced_3class.jsonl --per-class 50
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys

# --------------------------------------------------------------------------
# Reservoir sampling (uniform without knowing total size)
# --------------------------------------------------------------------------
class Reservoir:
    """Maintain a uniform sample of at most `capacity` items from a stream."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items = []
        self.seen = 0

    def add(self, item) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            j = random.randint(0, self.seen - 1)
            if j < self.capacity:
                self.items[j] = item

    def sample(self):
        return list(self.items)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
LABELS = {
    "positive": Reservoir,  # rating >= 4
    "neutral": Reservoir,   # rating == 3
    "negative": Reservoir,  # rating <= 2
}


def classify(rating: float) -> str | None:
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    elif rating <= 2:
        return "negative"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="data/Sports_and_Outdoors.jsonl.gz",
                        help="Path to the gzipped JSONL file")
    parser.add_argument("--out", default="data/balanced_3class.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--per-class", type=int, default=50,
                        help="Number of reviews to sample per class")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic sampling")
    args = parser.parse_args()

    # Initialise reservoirs per class
    reservoirs: dict[str, Reservoir] = {
        label: Reservoir(args.per_class) for label in ("positive", "neutral", "negative")
    }

    total = 0
    kept = {"positive": 0, "neutral": 0, "negative": 0}

    print(f"Streaming {args.src} …", file=sys.stderr)

    with gzip.open(args.src, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            rating = rec.get("rating")
            if rating is None:
                continue

            label = classify(rating)
            if label is None:
                continue

            reservoirs[label].add({
                "asin": rec.get("asin", ""),
                "rating": rating,
                "title": rec.get("title", ""),
                "text": rec.get("text", ""),
                "label": label,
            })
            kept[label] = reservoirs[label].seen
            total += 1

    print(f"  total reviews scanned: {total}", file=sys.stderr)

    # Collect samples
    all_items = []
    for label in ("positive", "neutral", "negative"):
        items = reservoirs[label].sample()
        print(f"  {label}: {len(items)} (scanned {kept[label]})", file=sys.stderr)
        all_items.extend(items)

    # Deterministic shuffle
    random.seed(args.seed)
    random.shuffle(all_items)

    # Write output
    with open(args.out, "w", encoding="utf-8") as fh:
        for item in all_items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_items)} reviews → {args.out}", file=sys.stderr)
    print(f"  Positive: {sum(1 for x in all_items if x['label']=='positive')}", file=sys.stderr)
    print(f"  Neutral:  {sum(1 for x in all_items if x['label']=='neutral')}", file=sys.stderr)
    print(f"  Negative: {sum(1 for x in all_items if x['label']=='negative')}", file=sys.stderr)


if __name__ == "__main__":
    main()
