#!/usr/bin/env python3
"""Run the spot check with a robust per-request timeout, handling hangs gracefully."""
import json, time, sys
from openai import OpenAI

client = OpenAI(api_key="6418", base_url="http://dobolyi.com:9000/v1", timeout=30.0)
sysp = ("You are a sentiment classifier. Given the review title and text, "
        "decide whether the sentiment is positive or negative. "
        "Reply with a single word only: POSITIVE or NEGATIVE. Do not add any explanation.")

rows = [json.loads(l) for l in open("spot_check.jsonl")]
results = []
errors = []

for i, r in enumerate(rows):
    title = r.get("title") or ""
    text = r.get("text") or ""
    user = f"Title: {title}\n\nText: {text}" if title else f"Text: {text}"
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model="DeepSeek-V4-Flash-0731", temperature=0.0,
            messages=[{"role": "system", "content": sysp},
                      {"role": "user", "content": user}])
        raw = resp.choices[0].message.content or ""
        dt = time.time() - t0
        pred = None
        if "POSITIVE" in raw.upper():
            pred = "positive"
        elif "NEGATIVE" in raw.upper():
            pred = "negative"
        results.append({
            "index": i,
            "asin": r["asin"],
            "label": r["label"],
            "title": title,
            "text": text,
            "prediction": pred,
            "raw": raw.strip(),
            "latency_s": round(dt, 1)
        })
        print(f"[OK] {dt:5.1f}s {r['asin']:>3} | label={r['label']:>8} pred={pred} | {raw.strip()[:50]}")
    except Exception as e:
        dt = time.time() - t0
        results.append({
            "index": i,
            "asin": r["asin"],
            "label": r["label"],
            "title": title,
            "text": text,
            "prediction": None,
            "raw": f"{type(e).__name__}: {e}",
            "latency_s": round(dt, 1)
        })
        errors.append(results[-1])
        print(f"[ERR] {dt:5.1f}s {r['asin']:>3} | {type(e).__name__}: {e}")
    time.sleep(0.5)  # brief backoff

# Compute accuracy
scored = [r for r in results if r["prediction"] is not None]
correct = [r for r in scored if r["prediction"] == r["label"]]
acc = len(correct) / len(scored) if scored else 0

tp = sum(1 for r in scored if r["prediction"] == "positive" and r["label"] == "positive")
tn = sum(1 for r in scored if r["prediction"] == "negative" and r["label"] == "negative")
fp = sum(1 for r in scored if r["prediction"] == "positive" and r["label"] == "negative")
fn = sum(1 for r in scored if r["prediction"] == "negative" and r["label"] == "positive")

print(f"\nTotal: {len(results)} | Scored: {len(scored)} | Errors: {len(errors)}")
print(f"Accuracy: {acc:.1%} ({len(correct)}/{len(scored)})")
print(f"TP={tp} TN={tn} FP={fp} FN={fn}")

# Save structured JSON for the HTML dashboard
output = {
    "total": len(results),
    "scored": len(scored),
    "errors": len(errors),
    "accuracy": round(acc, 4),
    "correct": len(correct),
    "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    "by_category": {},
    "reviews": scored,
    "errors": errors
}

# Categorize reviews for the analysis
for r in scored:
    text = r["text"].lower() + " " + r["title"].lower()
    if any(w in text for w in ["love", "awesome", "exceeded", "amazing", "😍", "💩"]):
        cat = "emoji"
    elif any(w in text for w in ["fast!", "fast!", "fast"]):
        cat = "mixed"
    elif any(w in text for w in ["great", "just", "yeah", "really", "honestly", "anything", "love how", "because who"]):
        cat = "sarcasm"
    elif any(w in text for w in ["is what", "whatever", "it's fine", "it works", "nothing more"]):
        cat = "neutral-ish"
    else:
        cat = "standard"
    r["category"] = cat

# Aggregate by category
cats = {}
for r in scored:
    c = r.get("category", "unknown")
    if c not in cats:
        cats[c] = {"total": 0, "correct": 0}
    cats[c]["total"] += 1
    if r["prediction"] == r["label"]:
        cats[c]["correct"] += 1

output["by_category"] = {
    k: {"total": v["total"], "correct": v["correct"],
        "accuracy": round(v["correct"]/v["total"], 4) if v["total"] else 0}
    for k, v in cats.items()
}

# Save as JSON + CSV
with open("results/spot_check/report.json", "w") as f:
    json.dump(output, f, indent=2)

# CSV for each review
import csv
with open("results/spot_check/predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["index","asin","label","title","text","prediction","raw","latency_s","category"])
    for r in scored:
        writer.writerow([r["index"], r["asin"], r["label"], r["title"],
                         r["text"], r["prediction"], r["raw"], r["latency_s"], r.get("category")])

print("\nWrote results/spot_check/report.json + results/spot_check/predictions.csv")
