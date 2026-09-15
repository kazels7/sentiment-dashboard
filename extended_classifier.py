#!/usr/bin/env python3
"""
Run 1: Extended classifier — asks the LLM for BOTH sentiment AND primary emotion.
Output: results/spot_check/extended.json
"""
import json, time
from openai import OpenAI

client = OpenAI(api_key="6418", base_url="http://dobolyi.com:9000/v1", timeout=30.0)

sysp = (
    "You are a sentiment and emotion classifier for Amazon product reviews. "
    "Given the review title and text, determine: "
    "(1) The overall sentiment: POSITIVE or NEGATIVE. "
    "(2) The primary emotion expressed by the reviewer, choosing exactly ONE from this list: "
    "joy, anger, sadness, fear, disgust, surprise, trust, anticipation. "
    "Reply with exactly two tokens separated by a space: SENTIMENT EMOTION. "
    "Example: POSITIVE joy. Do not add any explanation."
)

rows = [json.loads(l) for l in open("spot_check.jsonl")]
results = []

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
        raw = (resp.choices[0].message.content or "").strip()
        dt = time.time() - t0

        parts = raw.split()
        sent = None
        emo = None
        if len(parts) >= 2:
            sent = parts[0].upper()
            emo = parts[1].lower()
        elif len(parts) == 1:
            sent = parts[0].upper()

        valid_sent = sent in ("POSITIVE", "NEGATIVE")
        valid_emo = emo in ("joy", "anger", "sadness", "fear", "disgust", "surprise", "trust", "anticipation")

        results.append({
            "index": i,
            "asin": r["asin"],
            "label": r["label"],
            "title": title,
            "text": text,
            "llm_sentiment": sent if valid_sent else None,
            "llm_emotion": emo if valid_emo else None,
            "raw": raw,
            "latency_s": round(dt, 1)
        })
        print(f"[OK] {dt:5.1f}s {r['asin']:>3} | {r['label']:>8} → sent={sent} emo={emo}  | {raw[:40]}")
    except Exception as e:
        dt = time.time() - t0
        results.append({
            "index": i,
            "asin": r["asin"],
            "label": r["label"],
            "title": title,
            "text": text,
            "llm_sentiment": None,
            "llm_emotion": None,
            "raw": f"{type(e).__name__}: {e}",
            "latency_s": round(dt, 1)
        })
        print(f"[ERR] {dt:5.1f}s {r['asin']:>3} | {type(e).__name__}: {e}")
    time.sleep(0.3)

# Save
with open("results/spot_check/extended.json", "w") as f:
    json.dump(results, f, indent=2)

# Quick stats
ok = [r for r in results if r["llm_sentiment"]]
emo_counts = {}
for r in ok:
    e = r["llm_emotion"] or "unknown"
    emo_counts[e] = emo_counts.get(e, 0) + 1
print(f"\nScored: {len(ok)}/{len(results)}")
print(f"Emotion distribution: {emo_counts}")
