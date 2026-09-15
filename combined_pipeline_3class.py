"""
Pipeline: combine 3-class LLM emotions (from extended.json) with NRC word-list emotions.
Outputs results/spot_check_3class/combined.json for dashboard rendering.
"""
import json, os, sys

BASE = os.path.dirname(__file__)
EXTENDED = os.path.join(BASE, "results/spot_check_3class/extended.json")
sys.path.insert(0, BASE)
from nrc_emotion_scorer import score_review

# Load LLM results
with open(EXTENDED) as f:
    llm_data = json.load(f)

results = []
for rev in llm_data:
    # NRC scorer
    nrc_scores, nrc_dominant = score_review(rev["text"])
    nrc_emotion = nrc_dominant if nrc_dominant else "neutral"
    nrc_top_scores = dict(sorted(nrc_scores.items(), key=lambda x: -x[1]))
    # Top 3 NRC scores (non-zero)
    top3 = [(e, s) for e, s in nrc_top_scores.items() if s > 0][:3]

    llm_sent = rev["llm_sentiment"]
    llm_emo = rev["llm_emotion"]
    gt_sent = rev["label"].upper()
    correct = llm_sent == gt_sent

    # Convert emotions to emoji
    emo_map = {
        "anger": "😠", "anticipation": "🤔", "disgust": "🤢", "fear": "😨",
        "joy": "😊", "sadness": "😢", "surprise": "😲", "trust": "🤝",
        "neutral": "😐", "unknown": "❓", "error": "❌"
    }
    llm_emo_icon = emo_map.get(llm_emo, llm_emo)
    nrc_emo_icon = emo_map.get(nrc_emotion, nrc_emotion)
    agreement = llm_emo == nrc_emotion

    results.append({
        "index": rev["index"],
        "asin": rev["asin"],
        "rating": rev.get("rating", 0),
        "label": rev["label"],
        "title": rev.get("title", ""),
        "text": rev["text"],
        "category": rev.get("category", ""),
        "llm_sentiment": llm_sent,
        "llm_emotion": llm_emo,
        "llm_emotion_icon": llm_emo_icon,
        "nrc_sentiment": llm_sent,  # NRC inherits LLM sentiment for consistency
        "nrc_emotion": nrc_emotion,
        "nrc_emotion_icon": nrc_emo_icon,
        "nrc_top_emotions": top3,
        "agreement": agreement,
        "correct": correct,
        "latency_s": rev["latency_s"],
    })

# Compute summary stats
total = len(results)
correct = sum(1 for r in results if r["correct"])
agree = sum(1 for r in results if r["agreement"])

# Per-class accuracy
for cls in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
    cls_reviews = [r for r in results if r["label"] == cls.lower()]
    cls_correct = sum(1 for r in cls_reviews if r["correct"])
    print(f"{cls}: {cls_correct}/{len(cls_reviews)} correct")

print(f"\nOverall: {correct}/{total} ({correct/total*100:.0f}%) correct")
print(f"Emotion agreement (LLM vs NRC): {agree}/{total} ({agree/total*100:.0f}%)")

with open(os.path.join(BASE, "results/spot_check_3class/combined.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nSaved: results/spot_check_3class/combined.json")
