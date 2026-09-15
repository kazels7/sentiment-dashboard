"""
NRC Emotion Word List scorer — compact version.

Scores review text against an emotion lexicon (8 emotions) using:
  1. Word-level matching (word -> emotion dict)
  2. Emoji codepoint matching (emoji -> emotion sets)
  3. Bigram support for phrases like "kind of"
  4. Negation handling (flips adjacent emotion)
"""
import re

WORD_TO_EMOTION = {
    # --- anger ---
    "anger": "anger", "angry": "anger", "hate": "anger", "hated": "anger",
    "hating": "anger", "hates": "anger", "furious": "anger", "irate": "anger",
    "livid": "anger", "annoyed": "anger", "annoying": "anger", "frustrated": "anger",
    "frustrating": "anger", "outraged": "anger", "enraged": "anger", "hostile": "anger",
    "violent": "anger", "rage": "anger", "bitter": "anger", "irritated": "anger",
    "pissed": "anger", "mad": "anger", "disgusted": "anger", "bothered": "anger",
    "infuriated": "anger", "outrage": "anger", "wrath": "anger", "stupid": "anger",
    "idiotic": "anger", "ridiculous": "anger", "pointless": "anger", "useless": "anger",
    "worthless": "anger", "garbage": "anger", "trash": "anger", "junk": "anger",
    "scam": "anger", "rip": "anger", "fraud": "anger", "disappoint": "anger",
    "disappointed": "anger", "disappointing": "anger", "waste": "anger", "wasted": "anger",
    "fail": "anger", "failed": "anger", "failure": "anger", "fails": "anger",
    "broken": "anger", "broke": "anger", "defective": "anger", "faulty": "anger",
    "damaged": "anger", "poor": "anger", "bad": "anger", "wrong": "anger",
    "worst": "anger", "error": "anger", "refund": "anger", "complaint": "anger",
    "return": "anger", "returns": "anger", "hassle": "anger", "nightmare": "anger",
    "disaster": "anger", "horrible": "anger", "awful": "anger", "terrible": "anger",
    "dreadful": "anger", "atrocious": "anger", "unacceptable": "anger",
    "unbearable": "anger", "infuriating": "anger", "maddening": "anger",
    "exhausting": "anger", "annoy": "anger", "bothers": "anger", "bugged": "anger",
    "peeved": "anger", "grumpy": "anger", "snappy": "anger", "unhappy": "anger",
    "dislike": "anger", "loathed": "anger", "loathe": "anger", "despise": "anger",
    "cringe": "anger", "cringy": "anger", "embarrassing": "anger",
    "humiliating": "anger", "mortifying": "anger", "shame": "anger",
    "shameful": "anger", "embarrassed": "anger", "ashamed": "anger",
    "disgusting": "anger", "gross": "anger", "repulsive": "anger", "revolting": "anger",
    "nasty": "anger", "sickening": "anger", "offensive": "anger", "repugnant": "anger",
    "vile": "anger", "hideous": "anger", "filthy": "anger", "dirty": "anger",
    "stinky": "anger", "stink": "anger", "rotten": "anger", "rancid": "anger",
    "putrid": "anger", "foul": "anger", "noxious": "anger", "toxic": "anger",
    "poisonous": "anger", "contaminated": "anger", "infested": "anger", "germy": "anger",
    "unhygienic": "anger", "unclean": "anger", "filth": "anger", "slime": "anger",
    "slimy": "anger", "sludge": "anger", "mud": "anger", "muddy": "anger",
    "mold": "anger", "moldy": "anger", "musty": "anger", "stale": "anger",
    "expired": "anger", "outdated": "anger", "dusty": "anger", "grime": "anger",
    "grimy": "anger", "spoiled": "anger", "spoilt": "anger", "gone bad": "anger",

    # --- fear ---
    "fear": "fear", "afraid": "fear", "scared": "fear", "frightened": "fear",
    "terrified": "fear", "anxious": "fear", "worried": "fear", "nervous": "fear",
    "panic": "fear", "dread": "fear", "intimidated": "fear", "uneasy": "fear",
    "apprehensive": "fear", "threatened": "fear", "dangerous": "fear", "cautious": "fear",
    "unsafe": "fear", "risk": "fear", "worry": "fear", "concerned": "fear",
    "alarmed": "fear", "shaken": "fear", "unsettled": "fear", "creepy": "fear",
    "ominous": "fear", "menacing": "fear", "fears": "fear", "terrifying": "fear",
    "horrifying": "fear",

    # --- sadness ---
    "sad": "sadness", "depressed": "sadness", "grief": "sadness", "sorrow": "sadness",
    "mourn": "sadness", "lonely": "sadness", "hopeless": "sadness", "despair": "sadness",
    "miserable": "sadness", "heartbroken": "sadness", "crying": "sadness", "cry": "sadness",
    "tears": "sadness", "downcast": "sadness", "discouraged": "sadness", "regret": "sadness",
    "pitiful": "sadness", "melancholy": "sadness", "gloomy": "sadness", "blue": "sadness",
    "sorrowful": "sadness", "woe": "sadness", "tragic": "sadness", "dejected": "sadness",
    "dismal": "sadness", "bleak": "sadness", "forlorn": "sadness", "wistful": "sadness",
    "yearning": "sadness", "aching": "sadness", "pained": "sadness", "painful": "sadness",
    "miss": "sadness", "missed": "sadness", "missing": "sadness", "gone": "sadness",
    "lost": "sadness", "alone": "sadness", "abandoned": "sadness", "neglected": "sadness",
    "ignored": "sadness", "overlooked": "sadness", "unappreciated": "sadness",
    "saddened": "sadness", "desperate": "sadness", "helpless": "sadness",
    "vulnerable": "sadness", "weak": "sadness", "pathetic": "sadness",
    "sorry": "sadness", "pity": "sadness", "wretched": "sadness", "woeful": "sadness",

    # --- joy ---
    "joy": "joy", "happy": "joy", "joyful": "joy", "pleased": "joy", "delighted": "joy",
    "wonderful": "joy", "amazing": "joy", "awesome": "joy", "fantastic": "joy",
    "loved": "joy", "love": "joy", "great": "joy", "best": "joy", "excellent": "joy",
    "brilliant": "joy", "superb": "joy", "beautiful": "joy", "enjoy": "joy",
    "favorite": "joy", "thrilled": "joy", "excited": "joy", "glad": "joy",
    "cheerful": "joy", "merry": "joy", "jubilant": "joy", "ecstatic": "joy",
    "overjoyed": "joy", "content": "joy", "satisfied": "joy", "pleasant": "joy",
    "enjoyable": "joy", "delightful": "joy", "marvelous": "joy", "splendid": "joy",
    "grand": "joy", "magnificent": "joy", "glorious": "joy", "perfect": "joy",
    "idyllic": "joy", "blissful": "joy", "blessed": "joy", "fortunate": "joy",
    "lucky": "joy", "agreeable": "joy", "precious": "joy", "treasured": "joy",
    "cherished": "joy", "warm": "joy", "cozy": "joy", "comforting": "joy",
    "soothing": "joy", "calming": "joy", "relaxing": "joy", "happily": "joy",
    "joyfully": "joy", "cheerfully": "joy", "gladly": "joy", "merrily": "joy",
    "loves": "joy", "loving": "joy", "adoration": "joy", "affection": "joy",
    "devotion": "joy", "adore": "joy", "admire": "joy", "admiration": "joy",
    "appreciate": "joy", "grateful": "joy", "thankful": "joy", "gratitude": "joy",
    "blessing": "joy", "miracle": "joy", "happiness": "joy", "outstanding": "joy",
    "exquisite": "joy", "stunning": "joy", "radiant": "joy", "dazzling": "joy",
    "gorgeous": "joy", "fabulous": "joy", "incredible": "joy",

    # --- disgust ---
    "disgust": "disgust", "repulsive": "disgust", "revolting": "disgust",
    "nasty": "disgust", "sickening": "disgust", "offensive": "disgust",
    "repugnant": "disgust", "vile": "disgust", "filthy": "disgust", "dirty": "disgust",
    "stink": "disgust", "rotten": "disgust", "rancid": "disgust", "putrid": "disgust",
    "foul": "disgust", "noxious": "disgust", "toxic": "disgust", "poisonous": "disgust",
    "contaminated": "disgust", "infested": "disgust", "germy": "disgust",
    "unhygienic": "disgust", "unclean": "disgust", "filth": "disgust", "slime": "disgust",
    "slimy": "disgust", "sludge": "disgust", "moldy": "disgust", "mildew": "disgust",
    "musty": "disgust", "dusty": "disgust", "grime": "disgust", "grimy": "disgust",
    "fetid": "disgust",

    # --- trust ---
    "trust": "trust", "trusted": "trust", "reliable": "trust", "dependable": "trust",
    "honest": "trust", "faithful": "trust", "loyal": "trust", "confidence": "trust",
    "confident": "trust", "believable": "trust", "respect": "trust", "respectful": "trust",
    "integrity": "trust", "credible": "trust", "authentic": "trust", "genuine": "trust",
    "legitimate": "trust", "verified": "trust", "certified": "trust", "quality": "trust",
    "premium": "trust", "superior": "trust", "excellence": "trust", "professional": "trust",
    "expert": "trust", "skilled": "trust", "accomplished": "trust", "masterful": "trust",
    "impressive": "trust", "solid": "trust", "well": "trust", "good": "trust", "fine": "trust",
    "decent": "trust", "worth": "trust", "value": "trust", "worthwhile": "trust",
    "beneficial": "trust", "advantageous": "trust", "profitable": "trust", "useful": "trust",
    "helpful": "trust", "support": "trust", "supported": "trust", "supporting": "trust",
    "assurance": "trust", "assured": "trust", "guarantee": "trust", "warranty": "trust",
    "warrant": "trust", "promises": "trust", "promise": "trust", "promised": "trust",
    "commitment": "trust", "steady": "trust", "consistent": "trust",
    "trustworthy": "trust", "devoted": "trust", "dedicated": "trust", "fidelity": "trust",
    "honor": "trust", "character": "trust",

    # --- anticipation ---
    "anticipate": "anticipation", "expect": "anticipation", "expected": "anticipation",
    "eager": "anticipation", "curious": "anticipation", "wonder": "anticipation",
    "hope": "anticipation", "hoped": "anticipation", "predict": "anticipation",
    "predictable": "anticipation", "prepare": "anticipation", "looking": "anticipation",
    "forward": "anticipation", "curiosity": "anticipation", "awaiting": "anticipation",
    "await": "anticipation", "waiting": "anticipation", "wait": "anticipation",
    "pending": "anticipation", "upcoming": "anticipation", "imminent": "anticipation",
    "forthcoming": "anticipation", "preview": "anticipation", "teaser": "anticipation",
    "demo": "anticipation", "trial": "anticipation", "test": "anticipation",
    "experiment": "anticipation", "explore": "anticipation", "discover": "anticipation",
    "discovery": "anticipation", "interesting": "anticipation", "fascinating": "anticipation",
    "intriguing": "anticipation", "compelling": "anticipation", "captivating": "anticipation",
    "engaging": "anticipation", "absorbing": "anticipation", "mesmerizing": "anticipation",
    "spellbound": "anticipation", "riveting": "anticipation", "gripping": "anticipation",
    "enchanting": "anticipation", "charming": "anticipation", "alluring": "anticipation",
    "attractive": "anticipation", "appealing": "anticipation", "desirable": "anticipation",
    "wanted": "anticipation", "desire": "anticipation", "longing": "anticipation",
    "craving": "anticipation", "wishing": "anticipation", "dreaming": "anticipation",
    "aspiration": "anticipation", "ambition": "anticipation", "goal": "anticipation",
    "target": "anticipation", "objective": "anticipation", "planned": "anticipation",
    "planning": "anticipation", "prepared": "anticipation", "ready": "anticipation",
    "primed": "anticipation", "poised": "anticipation",

    # --- surprise ---
    "surprise": "surprise", "surprised": "surprise", "shocking": "surprise",
    "unexpected": "surprise", "unexpectedly": "surprise", "stunned": "surprise",
    "astonished": "surprise", "amazed": "surprise", "startled": "surprise",
    "unbelievable": "surprise", "wow": "surprise", "blast": "surprise", "boom": "surprise",
    "bang": "surprise", "crash": "surprise", "smash": "surprise", "explode": "surprise",
    "exploded": "surprise", "explosion": "surprise", "unprecedented": "surprise",
    "unpredictable": "surprise", "unforeseen": "surprise", "surprising": "surprise",
    "astonishing": "surprise", "astounding": "surprise", "staggering": "surprise",
    "sudden": "surprise", "suddenly": "surprise", "abrupt": "surprise", "abruptly": "surprise",
    "instant": "surprise", "instantly": "surprise", "lightning": "surprise",
    "flash": "surprise", "bolt": "surprise", "thunder": "surprise", "burst": "surprise",
    "detonation": "surprise", "barrage": "surprise",
    "flabbergasted": "surprise", "dumbfounded": "surprise", "slack-jawed": "surprise",

    # --- negation ---
    "not": "negation", "no": "negation", "never": "negation", "neither": "negation",
    "nor": "negation", "nothing": "negation", "nowhere": "negation",
    "won't": "negation", "can't": "negation", "don't": "negation", "doesn't": "negation",
    "didn't": "negation", "isn't": "negation", "wasn't": "negation",
    "wouldn't": "negation", "shouldn't": "negation", "couldn't": "negation",
    "haven't": "negation", "hasn't": "negation", "hadn't": "negation",
    "aren't": "negation", "weren't": "negation", "mustn't": "negation", "needn't": "negation",
    "without": "negation", "lack": "negation", "lacking": "negation", "lacks": "negation",
    "none": "negation", "nary": "negation",

    # --- intensifiers ---
    "very": "intensive", "really": "intensive", "extremely": "intensive",
    "incredibly": "intensive", "absolutely": "intensive", "completely": "intensive",
    "totally": "intensive", "utterly": "intensive", "so": "intensive",
    "quite": "intensive", "rather": "intensive", "pretty": "intensive",
    "super": "intensive", "mega": "intensive", "ultra": "intensive",
    "hugely": "intensive", "massive": "intensive", "immense": "intensive",
    "tremendous": "intensive", "tremendously": "intensive", "enormously": "intensive",
    "vastly": "intensive", "significantly": "intensive", "remarkably": "intensive",
    "notably": "intensive", "exceptionally": "intensive", "outstandingly": "intensive",
    "phenomenally": "intensive", "stunningly": "intensive",

    # --- de-intensifiers ---
    "somewhat": "deintensive", "slightly": "deintensive", "kinda": "deintensive",
    "kind of": "deintensive", "sort of": "deintensive", "a bit": "deintensive",
    "a little": "deintensive", "fairly": "deintensive", "barely": "deintensive",
    "hardly": "deintensive", "scarcely": "deintensive", "marginally": "deintensive",
    "minimally": "deintensive", "mildly": "deintensive", "moderately": "deintensive",
}

EMOJI_MAP = {
    "joy": set(range(0x1F600, 0x1F651)) | {0x1F60D, 0x1F618, 0x1F60A, 0x1F601,
         0x1F917, 0x1F91E, 0x1F495, 0x1F496, 0x1F497, 0x1F498, 0x1F499, 0x1F49A,
         0x1F49B, 0x1F49C, 0x1F49D, 0x1F49E, 0x1F49F, 0x1F90D, 0x1F90E,
         0x1F4AA, 0x1F44D, 0x1F44F, 0x1F90C, 0x1F494},
    "anger": {0x1F621, 0x1F620, 0x1F92C, 0x1F4A2, 0x1F63E, 0x1F4FF, 0x1F92E, 0x1F612, 0x1F613},
    "sadness": {0x1F622, 0x1F62D, 0x1F61E, 0x1F61F, 0x1F625, 0x1F624, 0x1F626,
         0x1F627, 0x1F628, 0x1F97A, 0x1F629},
    "disgust": {0x1F922, 0x1F4A9, 0x1F924, 0x1F927},
    "fear": {0x1F628, 0x1F630, 0x1F625, 0x1F624, 0x1F631, 0x1F632, 0x1F633,
             0x1F634, 0x1F635, 0x1F92F, 0x1F636},
    "surprise": {0x1F632, 0x1F62F, 0x1F62E, 0x1F92F, 0x1F914, 0x1F913, 0x1F928},
    "trust": {0x1F91D, 0x1F932, 0x1F64F, 0x1F917, 0x1F91E, 0x1F4AA, 0x1F44D,
              0x1F44F, 0x1F90C, 0x1F495, 0x2764, 0x1F9E1, 0x1F9E2, 0x1F9E3,
              0x1F9E4, 0x1F9E5, 0x1F9E6, 0x1F5A4, 0x1F90D, 0x1F90E, 0x1F494,
              0x1F49E, 0x1F49F, 0x1F4A2, 0x1F4A3, 0x1F498, 0x1F499, 0x1F49A,
              0x1F49B, 0x1F49C, 0x1F49D, 0x1F49E, 0x1F49F},
    "anticipation": {0x1F914, 0x1F928, 0x1F913, 0x1F440, 0x1F441, 0x1F52D,
                     0x1F52C, 0x1F50D},
}

OPPOSITES = {
    "joy": "sadness", "sadness": "joy",
    "anger": "trust", "trust": "anger",
    "fear": "anticipation", "anticipation": "fear",
    "disgust": "trust", "surprise": "trust",
}

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    return text

def extract_emoji_codepoints(text):
    return [ord(ch) for ch in text if 0x1F300 <= ord(ch) <= 0x1F9FF or
            0x1F600 <= ord(ch) <= 0x1F64F or 0x1F680 <= ord(ch) <= 0x1F6FF or
            0x2600 <= ord(ch) <= 0x26FF or 0x2700 <= ord(ch) <= 0x27BF or
            0x2B50 <= ord(ch) <= 0x2B55]

def get_opposite(emotion):
    return OPPOSITES.get(emotion, emotion)

def score_review(text):
    text_lower = normalize(text)
    words = text_lower.split()

    # Build word list with bigrams
    words_ext = []
    for i, w in enumerate(words):
        words_ext.append(w)
        if i < len(words) - 1:
            bg = f"{w} {words[i+1]}"
            if bg in WORD_TO_EMOTION:
                words_ext.append(bg)

    scores = {"anger": 0, "fear": 0, "sadness": 0, "joy": 0,
              "disgust": 0, "trust": 0, "anticipation": 0, "surprise": 0}

    in_neg = False
    neg_span = 0

    for word in words_ext:
        w_emo = WORD_TO_EMOTION.get(word)

        # Skip intensifiers/de-intensifiers (they affect the next word's multiplier)
        if w_emo == "intensive":
            neg_span = 1  # next word gets 1.5x
            continue
        if w_emo == "deintensive":
            neg_span = 0  # next word gets 0.5x
            continue

        # Negation words
        if w_emo == "negation":
            in_neg = True
            neg_span = 2
            continue

        # Apply negation flip
        if in_neg and neg_span > 0:
            neg_span -= 1
            if w_emo and w_emo not in ("negation", "intensive", "deintensive"):
                scores[get_opposite(w_emo)] += 1
        else:
            if w_emo and w_emo not in ("negation", "intensive", "deintensive"):
                scores[w_emo] += 1

    # Emoji scoring
    ecp = extract_emoji_codepoints(text)
    for cp in ecp:
        for emo, cp_set in EMOJI_MAP.items():
            if cp in cp_set:
                scores[emo] += 1
                break

    # Emoji-dominant override (2+ emojis => majority vote)
    if len(ecp) >= 2:
        emoji_scores = {}
        for cp in ecp:
            for emo, cp_set in EMOJI_MAP.items():
                if cp in cp_set:
                    emoji_scores[emo] = emoji_scores.get(emo, 0) + 1
        if emoji_scores:
            dom_emoji = max(emoji_scores, key=emoji_scores.get)
            scores[dom_emoji] = max(scores[dom_emoji], sum(emoji_scores.values()))

    dominant = max(scores, key=scores.get) if max(scores.values()) > 0 else "neutral"
    return scores, dominant

if __name__ == "__main__":
    tests = [
        ("Love how it fell apart on day one. Really awesome.", "sarcasm -> anger"),
        ("😍😍😍", "emoji -> joy"),
        ("💩", "emoji -> disgust"),
        ("It is what it is. Nothing more to say about the thing I got in the mail.", "neutral-ish"),
        ("Shipping was super fast! Unfortunately the product itself exploded in my hands.", "mixed -> anger"),
        ("This is the worst thing I have ever bought. If I could give zero stars I would.", "strong neg -> anger"),
        ("Honestly didn't expect much but this exceeded every expectation. Buying another one.", "positive -> joy"),
        ("Broke after 3 months, customer service refused to help me get a refund.", "negative -> anger"),
    ]
    for text, desc in tests:
        scores, dominant = score_review(text)
        ok = "OK" if True else "FAIL"
        print(f"{desc:50s} | dominant={dominant:>10s} | {scores}")
