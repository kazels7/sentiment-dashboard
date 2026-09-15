"""
Spot check runner for 3-class classifier with emotion output.
Outputs: results/spot_check_3class/extended.json
"""
import json, time, os, sys
from openai import OpenAI

BASE = os.path.dirname(__file__)
SPOT_CHECK = os.path.join(BASE, "spot_check.jsonl")
OUTPUT_DIR = os.path.join(BASE, "results/spot_check_3class")
OUTPUT = os.path.join(OUTPUT_DIR, "extended.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load config
import yaml
with open(os.path.expanduser("~/.hermes/config.yaml"), "r") as f:
    cfg = yaml.safe_load(f) or {}
providers = cfg.get("custom_providers") or []
provider = None
for p in providers:
    if p.get("base_url") and p.get("model"):
        provider = p
        break
if provider is None:
    mb = cfg.get("model") or {}
    provider = mb

base_url = provider.get("base_url")
model = provider.get("model", "gpt-4o-mini")
api_key = provider.get("api_key")

client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0)

# Load reviews
reviews = []
with open(SPOT_CHECK) as f:
    for line in f:
        line = line.strip()
        if line:
            reviews.append(json.loads(line))

results = []
for i, rev in enumerate(reviews):
    text = f"Title: {rev.get('title', '')}\nText: {rev.get('text', '')}"
    # Derive ground-truth label from rating
    rating = rev.get("rating", 0)
    if rating >= 4:
        gt_label = "positive"
    elif rating == 3:
        gt_label = "neutral"
    elif rating <= 2:
        gt_label = "negative"
    else:
        gt_label = "unknown"

    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Classify sentiment: POSITIVE (4-5 stars, happy), NEUTRAL (3 stars, mixed/okay), or NEGATIVE (1-2 stars, unhappy). Also name primary emotion from: anger, anticipation, disgust, fear, joy, sadness, surprise, trust.\n\nFormat: SENTIMENT|EMOTION\n\n{text}\n\nOutput:"}],
            temperature=0,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content.strip()
        latency = time.time() - start
        parts = raw.split("|")
        if len(parts) == 2:
            sentiment = parts[0].strip().upper()
            emotion = parts[1].strip().lower()
        else:
            sentiment = "UNKNOWN"
            emotion = "unknown"
    except Exception as e:
        latency = time.time() - start
        sentiment = "ERROR"
        emotion = "error"

    results.append({
        "index": i,
        "asin": rev["asin"],
        "label": gt_label,
        "title": rev.get("title", ""),
        "text": rev["text"],
        "llm_sentiment": sentiment,
        "llm_emotion": emotion,
        "raw": f"{sentiment}|{emotion}",
        "latency_s": round(latency, 2),
        "category": rev.get("category", ""),
    })

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Spot check: {len(results)} reviews → {OUTPUT}")
