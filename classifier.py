#!/usr/bin/env python3
"""
classifier.py — Zero-shot binary sentiment classifier for Amazon reviews.

For every review in the sample it sends the review *title + text* to an
OpenAI-compatible chat-completions endpoint and asks the LLM to return
POSITIVE or NEGATIVE.

Ground-truth labelling (per the assignment):
  rating >= 4  ->  positive
  rating <= 2  ->  negative
  rating == 3  ->  dropped (not present in subsample)

Endpoint configuration (all optional; resolved in this order):
  1. CLI flags   --base-url / --model / --api-key
  2. Env vars    OPENAI_BASE_URL / OPENAI_MODEL / OPENAI_API_KEY
  3. Your Hermes config.yaml (custom provider)  <-- the default fallback

Outputs
-------
  results/predictions.csv     one row per review: label, rating, prediction, emotion, raw
  results/metrics.txt         accuracy, per-class precision/recall/F1, confusion matrix
  results/errors.txt          any endpoint failures (cleared each run)

Usage
-----
    python classifier.py --sample data/balanced_3class.jsonl
    python classifier.py --sample data/balanced_3class.jsonl --max-reviews 40
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter

from openai import OpenAI

# --------------------------------------------------------------------------
# Config resolution
# --------------------------------------------------------------------------
def _load_hermes_config() -> dict | None:
    """Return the custom provider block from ~/.hermes/config.yaml, if present."""
    try:
        import yaml
        cfg_path = os.path.expanduser("~/.hermes/config.yaml")
        if not os.path.exists(cfg_path):
            return None
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        providers = cfg.get("custom_providers") or []
        for p in providers:
            if p.get("base_url") and p.get("model"):
                return {
                    "base_url": p["base_url"],
                    "model": p["model"],
                    "api_key": p.get("api_key"),
                }
        # Fallback to top-level model block
        model_block = cfg.get("model") or {}
        if model_block.get("base_url"):
            return {
                "base_url": model_block["base_url"],
                "model": model_block.get("model", ""),
                "api_key": model_block.get("api_key"),
            }
        return None
    except Exception:
        return None


def resolve_config(args) -> dict:
    """Resolve endpoint config from CLI > env > config.yaml."""
    cfg = _load_hermes_config()
    return {
        "base_url": args.base_url or os.environ.get("OPENAI_BASE_URL") or (cfg and cfg["base_url"]) or "http://localhost:8000/v1",
        "model": args.model or os.environ.get("OPENAI_MODEL") or (cfg and cfg["model"]) or "gpt-4o-mini",
        "api_key": args.api_key or os.environ.get("OPENAI_API_KEY") or (cfg and cfg.get("api_key")) or "sk-no-key-required",
    }


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------
CLASSIFIER_PROMPT = """Classify sentiment: POSITIVE (4-5 stars, happy) or NEGATIVE (1-2 stars, unhappy).

Format: SENTIMENT

Title: {title}
Text: {text}

Output:"""


def classify_review(client: OpenAI, model: str, title: str, text: str, timeout: float = 30.0) -> tuple[str, float]:
    """Send one review to the LLM. Returns (sentiment, latency_s)."""
    prompt = CLASSIFIER_PROMPT.format(title=title or "", text=text)
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512,
            timeout=timeout,
        )
        raw = resp.choices[0].message.content.strip()
        latency = time.time() - start
        # Parse single token sentiment
        upper_raw = raw.upper().strip()
        if upper_raw in ("POSITIVE", "NEGATIVE"):
            return upper_raw, latency
        # Fallback: try to extract sentiment keyword
        for keyword in ("POSITIVE", "NEGATIVE"):
            if keyword in upper_raw:
                return keyword, latency
        return "UNKNOWN", latency
    except Exception as e:
        latency = time.time() - start
        return "ERROR", latency


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_metrics(predictions: list, labels: list) -> dict:
    """Compute accuracy, per-class precision/recall/F1, confusion matrix."""
    classes = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
    n = len(predictions)
    if n == 0:
        return {}

    # Overall accuracy
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    accuracy = correct / n

    # Per-class metrics
    metrics = {"accuracy": accuracy, "total": n, "correct": correct, "per_class": {}}
    confusion = {l: {p: 0 for p in classes} for l in classes}

    for pred, label in zip(predictions, labels):
        # Only count valid classes
        if pred not in classes:
            continue
        if label not in classes:
            continue
        confusion[label][pred] += 1

    for cls in classes:
        tp = confusion[cls][cls]
        fp = sum(confusion[o][cls] for o in classes if o != cls)
        fn = sum(confusion[cls][o] for o in classes if o != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics["per_class"][cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for l in labels if l == cls),
        }
        metrics["confusion"] = confusion

    # Macro F1
    f1s = [metrics["per_class"][c]["f1"] for c in classes]
    metrics["macro_f1"] = sum(f1s) / len(f1s)

    return metrics


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="data/subsample.jsonl", help="Input JSONL file")
    parser.add_argument("--max-reviews", type=int, default=None, help="Limit to N reviews")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds")
    args = parser.parse_args()

    cfg = resolve_config(args)
    print(f"Endpoint: {cfg['base_url']} | Model: {cfg['model']}", file=sys.stderr)

    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=args.timeout)

    # Load reviews
    reviews = []
    with open(args.sample, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                reviews.append(json.loads(line))
    if args.max_reviews:
        reviews = reviews[:args.max_reviews]

    print(f"Classifying {len(reviews)} reviews …", file=sys.stderr)

    results = []
    errors = []
    sentiments, ground_truths = [], []
    latencies = []

    for i, rev in enumerate(reviews):
        asin = rev.get("asin", f"row_{i}")
        label = rev["label"]  # "positive", "neutral", "negative"
        pred_sentiment = label.upper()  # ground truth in label format
        title = rev.get("title", "")
        text = rev.get("text", "")

        prediction, latency = classify_review(client, cfg["model"], title, text, args.timeout)
        latencies.append(latency)

        # Ground truth: label is already "positive" or "negative"
        gt_sentiment = label.upper()

        sentiments.append(prediction)
        ground_truths.append(gt_sentiment)

        results.append({
            "index": i,
            "asin": asin,
            "rating": rev.get("rating", 0),
            "title": title,
            "text": text,
            "ground_truth_sentiment": gt_sentiment,
            "predicted_sentiment": prediction,
            "correct": prediction == gt_sentiment,
            "latency_s": round(latency, 2),
            "raw": prediction,
        })

        if i % 25 == 0:
            print(f"  [{i+1}/{len(reviews)}]", file=sys.stderr)

    # Metrics
    metrics = compute_metrics(sentiments, ground_truths)
    print(f"\nAccuracy: {metrics.get('accuracy', 0)*100:.1f}% ({metrics.get('correct', 0)}/{metrics.get('total', 0)})", file=sys.stderr)

    # Write predictions.csv
    os.makedirs("results", exist_ok=True)
    with open("results/predictions.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["index", "asin", "rating", "title", "text",
                                                 "ground_truth_sentiment", "predicted_sentiment",
                                                 "emotion", "correct", "latency_s", "raw"])
        writer.writeheader()
        writer.writerows(results)

    # Write errors.txt
    error_results = [r for r in results if r["predicted_sentiment"] == "ERROR"]
    with open("results/errors.txt", "w", encoding="utf-8") as fh:
        for r in error_results:
            fh.write(f"[{r['index']}] {r['asin']}: {r['raw']}\n")

    # Write metrics.txt
    with open("results/metrics.txt", "w", encoding="utf-8") as fh:
        fh.write(f"=== 3-Class Sentiment Classifier Metrics ===\n")
        fh.write(f"Total: {metrics.get('total', 0)}\n")
        fh.write(f"Correct: {metrics.get('correct', 0)}\n")
        fh.write(f"Accuracy: {metrics.get('accuracy', 0)*100:.1f}%\n")
        fh.write(f"Macro F1: {metrics.get('macro_f1', 0)*100:.1f}%\n\n")
        fh.write("--- Per-Class ---\n")
        for cls in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
            m = metrics.get("per_class", {}).get(cls, {})
            fh.write(f"{cls}: P={m.get('precision',0)*100:.1f} R={m.get('recall',0)*100:.1f} F1={m.get('f1',0)*100:.1f} support={m.get('support',0)}\n")
        fh.write("\n--- Confusion Matrix ---\n")
        fh.write(f"{'':>10s} | POS     NEU     NEG\n")
        confusion = metrics.get("confusion", {})
        for row_cls in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
            cols = confusion.get(row_cls, {})
            fh.write(f"{row_cls:>10s} | {cols.get('POSITIVE',0):>6d}  {cols.get('NEUTRAL',0):>6d}  {cols.get('NEGATIVE',0):>6d}\n")
        fh.write(f"\nAvg latency: {sum(latencies)/len(latencies):.2f}s\n")

    # Save JSON for dashboard consumption
    report = {
        "total": len(results),
        "correct": metrics.get("correct", 0),
        "accuracy": metrics.get("accuracy", 0),
        "macro_f1": metrics.get("macro_f1", 0),
        "per_class": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in metrics.get("per_class", {}).items()},
        "confusion": {k: {kk: vv for kk, vv in v.items()} for k, v in metrics.get("confusion", {}).items()},
        "reviews": results,
    }
    with open("results/report_3class.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"Saved: results/predictions.csv, results/metrics.txt, results/report_3class.json", file=sys.stderr)


if __name__ == "__main__":
    main()
