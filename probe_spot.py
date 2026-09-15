#!/usr/bin/env python3
"""Probe each spot_check review against the endpoint with a short timeout,
reporting success/failure/time per input, to isolate a hanging review."""
import json, time
from openai import OpenAI

client = OpenAI(api_key="6418", base_url="http://dobolyi.com:9000/v1", timeout=25.0)
sysp = ("You are a sentiment classifier for Amazon product reviews. "
        "Given the review title and text, decide whether the sentiment expressed "
        "is positive or negative. Reply with a single word only: POSITIVE or "
        "NEGATIVE. Do not add any explanation.")

rows = [json.loads(l) for l in open("spot_check.jsonl")]
for r in rows:
    title, text = r.get("title") or "", r.get("text") or ""
    user = f"Title: {title}\n\nText: {text}" if title else f"Text: {text}"
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model="DeepSeek-V4-Flash-0731", temperature=0.0,
            messages=[{"role": "system", "content": sysp},
                      {"role": "user", "content": user}])
        raw = resp.choices[0].message.content or ""
        dt = time.time() - t0
        ok = "OK " if any(w in raw.upper() for w in ("POSITIVE", "NEGATIVE")) else "PARSE"
        print(f"[{ok}] {dt:5.1f}s {r['asin']} -> {raw!r}  | {user[:60]!r}")
    except Exception as e:
        print(f"[ERR] {time.time()-t0:5.1f}s {r['asin']} -> {type(e).__name__}: {e}")
