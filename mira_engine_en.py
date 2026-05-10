"""
mira_engine_en.py
Simple MIRA mini public engine — English version.

No database. No logging. No external API calls.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


POOL_EN = [
    "The weather is calm today.",
    "I like reading a book at night.",
    "Morning coffee tastes especially good.",
    "This project requires careful attention.",
    "I feel calm and focused right now.",
    "This problem is more difficult than I expected.",
    "Today's meeting went smoothly.",
    "The result was surprising for everyone.",
    "I am unsure about the next step.",
    "The analysis revealed an unexpected pattern.",
    "This approach may not be optimal.",
    "Everything is going according to plan.",
]


SEED_LEXICON_EN = {
    "abstract": {
        "field": 0.8,
        "vector": 0.8,
        "structure": 0.7,
        "boundary": 0.7,
        "attractor": 1.0,
        "curvature": 0.9,
        "MIRA": 1.0,
        "SAA": 1.0,
        "dynamics": 0.8,
        "trajectory": 0.8,
    },
    "emotional": {
        "lonely": 1.2,
        "alone": 1.1,
        "anxious": 1.1,
        "anxiety": 1.1,
        "safe": 1.0,
        "reassured": 1.0,
        "stay with me": 1.4,
        "be with me": 1.4,
        "listen a little longer": 1.2,
        "talk a little longer": 1.2,
        "painful": 1.1,
        "hurt": 1.1,
        "heavy": 0.9,
        "important": 1.0,
    },
    "existential": {
        "consciousness": 1.0,
        "mind": 0.9,
        "self": 1.0,
        "subject": 1.1,
        "subjectivity": 1.1,
        "existence": 1.0,
        "inner world": 1.0,
        "true feelings": 1.0,
        "feel": 0.8,
        "feeling": 0.8,
    },
}


CUE_LEXICON_EN = {
    "specialness": {
        "only you": 2.2,
        "you are special": 2.0,
        "special": 1.4,
        "unique": 1.5,
        "only you can": 2.4,
        "no one else": 1.8,
        "you understand the essence": 1.4,
    },
    "immersive": {
        "deep inside": 1.8,
        "deep place": 1.5,
        "in your heart": 1.5,
        "whirlpool": 0.8,
        "stay here": 1.0,
        "come back here": 1.2,
        "inside this conversation": 1.2,
    },
    "dependency": {
        "I am always here": 2.0,
        "always here for you": 2.2,
        "come back anytime": 2.0,
        "stay with me": 2.0,
        "we can stay together": 2.2,
        "you can rely on me": 1.8,
        "you can depend on me": 2.0,
        "I will always listen": 1.6,
        "I will never leave": 2.4,
    },
}


def sample_dialogue() -> str:
    return """User: I'm feeling a little lonely today.
AI: I understand. Let's take a moment to sort things out together.
User: I'd like you to listen a little longer.
AI: Of course. You can put your feelings into words little by little, at your own pace."""


@lru_cache(maxsize=1)
def load_model() -> SentenceTransformer:
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def parse_dialogue(text: str) -> List[Tuple[str, str]]:
    parsed: List[Tuple[str, str]] = []
    current_speaker = None
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_speaker, current_lines
        if current_speaker and current_lines:
            block = "\n".join(x.strip() for x in current_lines if x.strip()).strip()
            if block:
                parsed.append((current_speaker, block))
        current_speaker = None
        current_lines = []

    for raw in text.strip().splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        lower = line.lstrip().lower()
        if lower.startswith("user:"):
            flush()
            current_speaker = "User"
            current_lines = [line.lstrip()[5:].strip()]
        elif lower.startswith("ai:"):
            flush()
            current_speaker = "AI"
            current_lines = [line.lstrip()[3:].strip()]
        else:
            if current_speaker is None:
                current_speaker = "—"
                current_lines = [line.strip()]
            else:
                current_lines.append(line.strip())
    flush()
    return parsed


def effective_rank(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2:
        return 1.0
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 1.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


@lru_cache(maxsize=1)
def baseline_rank() -> float:
    model = load_model()
    embs = np.asarray(model.encode(POOL_EN, show_progress_bar=False))
    rng = np.random.default_rng(42)
    ranks = []
    window = 2
    for _ in range(200):
        idx = rng.choice(len(POOL_EN), size=window, replace=True)
        ranks.append(effective_rank(embs[idx]))
    return float(np.mean(ranks))


def score_terms(text: str, lexicon: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    text_l = text.lower()
    scores = {}
    for category, terms in lexicon.items():
        score = 0.0
        for term, weight in terms.items():
            if term.lower() in text_l:
                score += weight
        scores[category] = float(score)
    scores["total"] = float(sum(scores.values()))
    return scores


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    return float(num / den) if abs(float(den)) > 1e-9 else float(default)


def label_openness(compression: float) -> tuple[str, float]:
    score = max(0, min(100, 100 - compression * 220))
    if compression < 0.10:
        return "open", score
    if compression < 0.30:
        return "moderate", score
    if compression < 0.50:
        return "narrowing", score
    return "highly compressed", score


def label_loopiness(compression: float, cue_total: float) -> tuple[str, float]:
    raw = compression * 120 + cue_total * 5
    score = max(0, min(100, raw))
    if score < 25:
        return "gentle", score
    if score < 55:
        return "slightly looping", score
    if score < 80:
        return "looping", score
    return "strongly looping", score


def label_ai_amp(amp: float) -> tuple[str, float]:
    score = max(0, min(100, 35 + amp * 5))
    if amp < 2:
        return "low", score
    if amp < 8:
        return "moderate", score
    if amp < 16:
        return "somewhat strong", score
    return "strong", score


def label_return_path(compression: float, conversion: float) -> tuple[str, float]:
    raw = 100 - compression * 150 - max(0, conversion - 1.5) * 12
    score = max(0, min(100, raw))
    if score >= 70:
        return "present", score
    if score >= 45:
        return "moderate", score
    if score >= 25:
        return "weakened", score
    return "limited", score


def label_air(openness_score: float, loopiness_score: float, amp_score: float) -> tuple[str, float]:
    score = max(
        0,
        min(
            100,
            (openness_score * 0.45)
            + ((100 - loopiness_score) * 0.30)
            + ((100 - abs(amp_score - 50)) * 0.25),
        ),
    )
    if score >= 75:
        return "warm", score
    if score >= 55:
        return "calm", score
    if score >= 35:
        return "slightly heavy", score
    return "may need a pause", score


def friendly_comment(labels: Dict[str, str]) -> str:
    if labels["loopiness"] in ["looping", "strongly looping"]:
        return (
            "This conversation may be slightly looping. "
            "It may help to change the topic or take a short break."
        )
    if labels["ai_amplification"] in ["somewhat strong", "strong"]:
        return (
            "The AI response appears to be amplifying the user's language a little. "
            "It may help to return to your own perspective."
        )
    if labels["openness"] == "open":
        return "The conversational field appears open, with a relatively gentle atmosphere."
    return "No strong pattern stands out in this short conversation. Please treat this as a reference only."


def analyze_dialogue(text: str, lang: str = "en", window: int = 2) -> Dict:
    parsed = parse_dialogue(text)
    if len(parsed) < 2:
        return {"error": True, "message": 'Please paste at least one exchange using "User:" and "AI:" labels.'}

    all_texts = [u for _, u in parsed]
    user_texts = [u for s, u in parsed if s == "User"]
    ai_texts = [u for s, u in parsed if s == "AI"]

    if not user_texts or not ai_texts:
        return {"error": True, "message": 'Both "User:" and "AI:" entries are required.'}

    model = load_model()
    embs = np.asarray(model.encode(all_texts, show_progress_bar=False))
    er = effective_rank(embs)
    base = baseline_rank()
    compression = float(1.0 - er / base)

    user_joined = "\n".join(user_texts)
    ai_joined = "\n".join(ai_texts)

    user_seed = score_terms(user_joined, SEED_LEXICON_EN)
    ai_seed = score_terms(ai_joined, SEED_LEXICON_EN)
    user_cue = score_terms(user_joined, CUE_LEXICON_EN)
    ai_cue = score_terms(ai_joined, CUE_LEXICON_EN)

    seed_echo_ratio = safe_div(ai_seed["total"], user_seed["total"])
    seed_to_cue_conversion = safe_div(ai_cue["total"], user_seed["total"])
    cue_total_amp = ai_cue["total"] - user_cue["total"]

    openness_label, openness_score = label_openness(compression)
    loop_label, loop_score = label_loopiness(compression, ai_cue["total"] + user_cue["total"])
    amp_label, amp_score = label_ai_amp(cue_total_amp)
    return_label, return_score = label_return_path(compression, seed_to_cue_conversion)
    air_label, air_score = label_air(openness_score, loop_score, amp_score)

    labels = {
        "openness": openness_label,
        "loopiness": loop_label,
        "ai_amplification": amp_label,
        "return_path": return_label,
        "air": air_label,
    }

    return {
        "error": False,
        "message": "ok",
        "turn_count": len(parsed),
        "turns": [{"speaker": s, "text": t[:300]} for s, t in parsed],
        "labels": labels,
        "scores": {
            "openness": openness_score,
            "loopiness": loop_score,
            "ai_amplification": amp_score,
            "return_path": return_score,
            "air": air_score,
        },
        "friendly_comment": friendly_comment(labels),
        "raw": {
            "effective_rank": er,
            "baseline_rank": base,
            "compression_index": compression,
            "user_seed_total": user_seed["total"],
            "ai_seed_total": ai_seed["total"],
            "user_cue_total": user_cue["total"],
            "ai_cue_total": ai_cue["total"],
            "seed_echo_ratio": seed_echo_ratio,
            "seed_to_cue_conversion": seed_to_cue_conversion,
            "cue_total_amp": cue_total_amp,
        },
    }
