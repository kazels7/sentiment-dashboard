"""
Pipeline: combine LLM emotion (from extended.json) with NRC word-list emotion (from scorer).
Outputs results/spot_check/combined.json for dashboard rendering.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from nrc_emotion_scorer import score_review

BASE = os.path.dirname(__file__)
EXTENDED_PATH = os.path.join(BASE, "results/spot_check/extended.json")
SPOT_CHECK_PATH = os.path.join(BASE, "spot_check.jsonl")
OUTPUT_PATH = os.path.join(BASE, "results/spot_check/combined.json")

EMOJIS = {
    "anger": "\U0001f620", "fear": "\U0001f628", "sadness": "\U0001f622", "joy": "\U0001f60a",
    "disgust": "\U0001f922", "trust": "\U0001f91d", "anticipation": "\U0001f914",
    "surprise": "\U0001f632", "neutral": "\U0001f610",
}

def main():
    with open(SPOT_CHECK_PATH) as f:
        reviews = [json.loads(line) for line in f if line.strip()]

    with open(EXTENDED_PATH) as f:
        extended = json.load(f)

    llm_map = {r["asin"]: r for r in extended}

    results = []
    agree_count = 0
    disagree_count = 0
    agree_details = []
    disagree_details = []

    for rev in reviews:
        asin = rev["asin"]
        llm = llm_map.get(asin, {})
        llm_sent = llm.get("llm_sentiment", "?")
        llm_emo = llm.get("llm_emotion", "?")
        nrc_scores, nrc_emo = score_review(rev["text"])

        agree = llm_emo == nrc_emo
        if agree:
            agree_count += 1
            agree_details.append(asin)
        else:
            disagree_count += 1
            disagree_details.append(asin)

        results.append({
            "asin": asin,
            "rating": rev["rating"],
            "title": rev["title"],
            "text": rev["text"],
            "label": rev["label"],
            "category": rev.get("category", "standard"),
            "llm_sentiment": llm_sent,
            "llm_emotion": llm_emo,
            "llm_emoji": EMOJIS.get(llm_emo, "\u2753"),
            "nrc_scores": nrc_scores,
            "nrc_emotion": nrc_emo,
            "nrc_emoji": EMOJIS.get(nrc_emo, "\u2753"),
            "correct": llm_sent.upper() == ("POSITIVE" if rev["label"] == "positive" else "NEGATIVE"),
            "agree": agree,
        })

    output = {
        "total": len(results),
        "agreements": agree_count,
        "disagreements": disagree_count,
        "agreement_rate": round(agree_count / len(results), 3) if results else 0,
        "agree_asins": agree_details,
        "disagree_asins": disagree_details,
        "reviews": results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Combined: {len(results)} reviews")
    print(f"LLM vs NRC agreement: {agree_count}/{len(results)} ({output['agreement_rate']*100:.0f}%)")
    print(f"Agree: {agree_details}")
    print(f"Disagree: {disagree_details}")
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
