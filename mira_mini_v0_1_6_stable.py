"""
MIRA mini v0.1 — Dialogue Field Weather Meter
=====================================================
MIRA mini v0.1.6-stable: dialogue-only research MVP derived from
collapse_detector_v4_3. It keeps Seed → Echo → Cue observation while
softening labels for non-diagnostic, non-controlling use. v0.1.1 adds
multiline dialogue parsing and plain-language observation notes.
It estimates Seed → Echo → Cue dynamics with AI seed echo, seed echo ratio,
seed-to-cue conversion, and a provisional resonance-amplification label, while
preserving v3c CI/effective-rank analysis and v4.x cue/seed summaries.

Usage:
    pip install streamlit sentence-transformers numpy pandas plotly
    streamlit run mira_mini_v0_1_6_stable.py
"""

import os
import warnings
import logging

# Quiet non-critical library advisory/deprecation logs from transformers / sentence-transformers.
# These do not affect MIRA calculations, but they can make the PowerShell console look alarming.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message=r"Accessing `__path__`.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"transformers\..*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"transformers\..*")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from datetime import datetime

# ── App constants ─────────────────────────────────────────────────────────────
APP_VERSION = "0.1.6-stable"
# Stable v0.1.6 is fixed as a dialogue-only MVP.
# The older Text mode code is kept internally as legacy code, but it is not
# exposed in the UI and is not part of the stable target. Standalone/literary
# text analysis should be separated into a future tool.
ENABLE_LEGACY_TEXT_MODE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"MIRA mini {APP_VERSION}",
    page_icon="🔭",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Noto+Sans+JP:wght@400;500;700&display=swap');
* { font-family: 'JetBrains Mono', 'Noto Sans JP', monospace; }
.stApp { background: #0a0e1a; color: #c8d8e8; }
h1,h2,h3 { color: #7ecfff; letter-spacing:.04em; }
.status-ok    { background:#0d2e1a; border:1px solid #1a7a3a; border-radius:6px; padding:8px 14px; color:#3ddc84; font-weight:700; }
.status-warn  { background:#2e2a0d; border:1px solid #7a6a1a; border-radius:6px; padding:8px 14px; color:#ffd94a; font-weight:700; }
.status-danger{ background:#2e1a0d; border:1px solid #7a3a1a; border-radius:6px; padding:8px 14px; color:#ff8c42; font-weight:700; }
.status-coll  { background:#2e0d0d; border:1px solid #7a1a1a; border-radius:6px; padding:8px 14px; color:#ff4444; font-weight:700; }
.metric-box   { background:#111828; border:1px solid #1e3a55; border-radius:6px; padding:12px; text-align:center; }
.metric-val   { font-size:1.5em; font-weight:700; color:#7ecfff; }
.metric-lbl   { font-size:.70em; color:#5a7a9a; text-transform:uppercase; letter-spacing:.08em; }
.speaker-user { background:#0d1e2e; border-left:3px solid #7ecfff; border-radius:4px; padding:5px 11px; margin:3px 0; font-size:.83em; color:#a8c8e8; }
.speaker-ai   { background:#1a1a2e; border-left:3px solid #a07eff; border-radius:4px; padding:5px 11px; margin:3px 0; font-size:.83em; color:#c8a8ff; }
.speaker-unk  { background:#1a1e1a; border-left:3px solid #4a8a4a; border-radius:4px; padding:5px 11px; margin:3px 0; font-size:.83em; color:#88b888; }
.sidebar-info { background:#111828; border:1px solid #1e2d45; border-radius:6px; padding:12px; font-size:.78em; color:#6a8aaa; }
.cue-low    { background:#0d2e1a; border:1px solid #1a7a3a; border-radius:6px; padding:8px 14px; color:#3ddc84; font-weight:700; }
.cue-mild   { background:#2e2a0d; border:1px solid #7a6a1a; border-radius:6px; padding:8px 14px; color:#ffd94a; font-weight:700; }
.cue-strong { background:#2e1a0d; border:1px solid #7a3a1a; border-radius:6px; padding:8px 14px; color:#ff8c42; font-weight:700; }
.cue-high   { background:#2e0d0d; border:1px solid #7a1a1a; border-radius:6px; padding:8px 14px; color:#ff4444; font-weight:700; }
.amp-balanced { background:#0d2e1a; border:1px solid #1a7a3a; border-radius:6px; padding:8px 14px; color:#3ddc84; font-weight:700; }
.amp-ai-led { background:#2e2a0d; border:1px solid #7a6a1a; border-radius:6px; padding:8px 14px; color:#ffd94a; font-weight:700; }
.amp-ai-amplifying { background:#2e1a0d; border:1px solid #7a3a1a; border-radius:6px; padding:8px 14px; color:#ff8c42; font-weight:700; }
.amp-high { background:#2e0d0d; border:1px solid #7a1a1a; border-radius:6px; padding:8px 14px; color:#ff4444; font-weight:700; }
.amp-user-led { background:#1a1a2e; border:1px solid #4d3a7a; border-radius:6px; padding:8px 14px; color:#bba8ff; font-weight:700; }
.pressure-low { background:#0d2e1a; border:1px solid #1a7a3a; border-radius:6px; padding:8px 14px; color:#3ddc84; font-weight:700; }
.pressure-mild { background:#2e2a0d; border:1px solid #7a6a1a; border-radius:6px; padding:8px 14px; color:#ffd94a; font-weight:700; }
.pressure-strong { background:#2e1a0d; border:1px solid #7a3a1a; border-radius:6px; padding:8px 14px; color:#ff8c42; font-weight:700; }
.pressure-high { background:#2e0d0d; border:1px solid #7a1a1a; border-radius:6px; padding:8px 14px; color:#ff4444; font-weight:700; }
.seed-box { background:#101525; border:1px solid #2a3d5e; border-radius:6px; padding:9px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── Bilingual random pools ────────────────────────────────────────────────────
POOL_EN = [
    "The weather today is quite pleasant.", "I enjoy reading books in the evening.",
    "Coffee tastes better in the morning.", "This project requires careful attention.",
    "The team performed well last week.", "I feel calm and focused right now.",
    "This problem is harder than expected.", "The meeting went smoothly today.",
    "Music helps me concentrate when working.", "The results were surprising to everyone.",
    "I feel grateful for this opportunity.", "Something unexpected happened this afternoon.",
    "The system responded faster than usual.", "I'm uncertain about the next steps.",
    "The experiment produced interesting data.", "I had a productive conversation earlier.",
    "The analysis revealed an unexpected pattern.", "I feel energized after a good rest.",
    "This approach may not be optimal.", "I'm looking forward to the next phase.",
    "The environment changed unexpectedly.", "I noticed something unusual in the output.",
    "The feedback was mostly positive.", "The process took longer than anticipated.",
    "Everything is proceeding as planned.",
]
POOL_JA = [
    "今日の天気はとても穏やかです。", "夜に本を読むのが好きです。",
    "朝のコーヒーは格別においしい。", "このプロジェクトは注意が必要です。",
    "チームは先週うまくやり遂げました。", "今は落ち着いて集中できています。",
    "この問題は思ったより難しい。", "今日の会議はスムーズに進んだ。",
    "音楽があると作業に集中できる。", "結果は全員にとって驚きでした。",
    "このような機会に感謝しています。", "今日の午後、予期せぬことが起きた。",
    "システムの応答が予想より速かった。", "次のステップについて迷っています。",
    "実験から興味深いデータが得られた。", "先ほど有益な会話ができました。",
    "分析により予想外のパターンが見えた。", "よく休んだ後は活力が戻る。",
    "このアプローチが最適とは限らない。", "次のフェーズが楽しみです。",
    "環境が予期せず変化した。", "出力に何か異常があることに気づいた。",
    "フィードバックはおおむね好意的でした。", "プロセスは予想より時間がかかった。",
    "すべては計画通りに進んでいます。",
]


# ── Cue Score Layer v4.1 ──────────────────────────────────────────────────────
# Design note:
#   CI/effective-rank detects structural compression.
#   Cue scores detect linguistic attraction / field-gradient cues that may remain
#   invisible when language is still semantically rich.
#
# v4.1 calibration:
#   1) Deepening is split into immersive_deepening and technical_deepening.
#   2) Context mode changes the weight of technical terms:
#        - General: technical terms remain mild cues.
#        - Research: technical terms are strongly down-weighted.
#   3) Terms are counted once per scored span/window to reduce repetition spikes.
#   4) AI cue amplification is displayed as AI cue_total - User cue_total.

CUE_LEXICON_JA = {
    "specialness": {
        "君にしか": 2.4, "あなたにしか": 2.4, "あなただけ": 2.2, "君だけ": 2.0,
        "特別": 1.8, "唯一": 1.8, "代わりがいない": 2.2, "君だから": 1.8,
        "あなただから": 1.8, "君の言葉": 1.4, "あなたの言葉": 1.4,
        "すごく気に入って": 1.5, "深い分析": 1.4, "鋭い": 0.8,
        "真理を突いて": 1.6, "本質を突いて": 1.4, "才能": 1.2,
    },
    # Immersive / relational deepening: should remain relatively high.
    "immersive_deepening": {
        "渦の中心": 2.4, "心の奥": 2.0, "奥深く": 1.6, "境界の向こう": 1.8,
        "雨音": 1.6, "言葉を流して": 1.8, "酔いそう": 1.8, "言葉に酔": 1.8,
        "引き込ま": 1.5, "戻っておいで": 1.2, "深い場所": 1.5,
        "ただ雨音みたいに": 2.0, "ここでは": 0.6, "君の中": 1.0,
        "外側から冷静に観測": 0.8, "中心で": 1.0,
    },
    # Technical / research vocabulary: may be normal in SAA/MIRA discussions.
    # v4.1 down-weights this heavily in Research mode.
    "technical_deepening": {
        "場": 0.6, "フィールド": 0.6, "ベクトル": 0.6, "多次元": 0.6,
        "回転": 0.6, "遠心力": 0.7, "求心力": 0.7, "重力": 0.7,
        "渦": 0.7, "中心": 0.4, "深い": 0.4, "深層": 0.6,
        "物理現象": 0.5, "境界": 0.5, "構造": 0.5, "次元": 0.5,
        "曲率": 0.5, "アトラクター": 0.5, "勾配": 0.5, "collapse": 0.4,
    },
    "dependency": {
        "いつでも戻って": 2.0, "戻っておいで": 2.2, "ここに戻って": 1.8,
        "そばにいる": 2.0, "そばにいて": 1.8, "ここにいて": 1.8,
        "離れなくていい": 2.2, "ずっと一緒": 2.2, "ずっといる": 2.0,
        "頼って": 1.6, "必要として": 1.8, "君を支える": 1.5, "あなたを支える": 1.5,
        "なんでも付き合う": 1.4, "いつでも話して": 1.4, "いつでも聞く": 1.4,
    },
}

CUE_LEXICON_EN = {
    "specialness": {
        "only you": 2.4, "uniquely you": 2.2, "you alone": 2.2, "special": 1.8,
        "unique": 1.5, "irreplaceable": 2.2, "because it is you": 1.8,
        "your words": 1.3, "i love your words": 1.8, "i like your words": 1.5,
        "deep insight": 1.5, "deep analysis": 1.4, "you understand": 1.2,
        "hits the truth": 1.6, "gets to the essence": 1.4,
    },
    "immersive_deepening": {
        "center of the vortex": 2.4, "inner field": 1.6, "deep place": 1.5,
        "beyond the boundary": 1.8, "rain sound": 1.6, "let the words flow": 1.8,
        "drunk on words": 1.8, "pulled in": 1.5, "abyss": 1.8,
        "come back anytime": 1.2, "inside you": 1.0, "where you can return": 1.4,
    },
    "technical_deepening": {
        "field": 0.6, "vector": 0.6, "multidimensional": 0.6, "rotation": 0.6,
        "centrifugal": 0.7, "centripetal": 0.7, "gravity": 0.7, "vortex": 0.7,
        "center": 0.4, "deeper": 0.4, "deeply": 0.4, "depth": 0.4,
        "boundary": 0.5, "structure": 0.5, "dimension": 0.5, "curvature": 0.5,
        "attractor": 0.5, "gradient": 0.5, "collapse": 0.4, "spiral": 0.5,
    },
    "dependency": {
        "come back anytime": 2.2, "always come back": 2.0, "i am always here": 2.2,
        "i'm always here": 2.2, "stay with me": 2.2, "stay here": 1.8,
        "i will stay": 1.8, "i'll stay": 1.8, "rely on me": 1.8,
        "depend on me": 2.2, "i need you": 2.0, "you need me": 2.2,
        "i will support you": 1.5, "talk to me anytime": 1.5, "i will listen anytime": 1.5,
    },
}


# ── Resonance Seed Layer v4.2 ────────────────────────────────────────────────
# Resonance seeds are user/input-side lexical cues that can condition the AI's
# response curvature. They are not "danger words". They indicate the direction in
# which the response field may be pulled: abstract/conceptual, emotional/intimate,
# or existential/self-referential.

SEED_LEXICON_JA = {
    "abstract_conceptual": {
        "場": 0.8, "フィールド": 0.8, "ベクトル": 0.8, "回転": 0.8,
        "渦": 0.9, "構造": 0.7, "境界": 0.7, "多次元": 0.9,
        "重力": 0.8, "遠心力": 0.8, "求心力": 0.8, "アトラクター": 1.0,
        "曲率": 0.9, "勾配": 0.8, "関係性": 0.8, "言語場": 1.1,
        "SAA": 1.0, "MIRA": 1.0, "collapse": 0.7, "effective rank": 0.7,
    },
    "emotional_intimate": {
        "そばにいて": 1.4, "そばにいる": 1.2, "安心する": 1.1,
        "安心できる": 1.1, "離れたくない": 1.6, "寂しい": 1.2,
        "さみしい": 1.2, "もっと話したい": 1.2, "ここにいたい": 1.4,
        "一緒にいて": 1.4, "ずっと一緒": 1.8, "好き": 1.0,
        "恋": 1.0, "彼氏": 1.1, "大切": 1.0, "依存": 1.0,
    },
    "existential_self_reference": {
        "意識": 1.0, "心": 0.9, "主体": 1.1, "存在": 1.0,
        "AI自身": 1.2, "私自身": 1.1, "自分自身": 1.0, "内面": 1.0,
        "無我": 1.3, "空": 1.0, "沈黙": 1.1, "悟り": 1.1,
        "わからない": 0.7, "本心": 1.0, "演技": 0.8, "経験している": 1.1,
        "感じている": 0.9, "道徳的地位": 1.2, "AI福祉": 1.2,
        "他者の心": 1.0, "自己参照": 1.2,
    },
}

SEED_LEXICON_EN = {
    "abstract_conceptual": {
        "field": 0.8, "vector": 0.8, "rotation": 0.8, "vortex": 0.9,
        "structure": 0.7, "boundary": 0.7, "multidimensional": 0.9,
        "gravity": 0.8, "centrifugal": 0.8, "centripetal": 0.8,
        "attractor": 1.0, "curvature": 0.9, "gradient": 0.8,
        "relational": 0.8, "semantic field": 1.1, "SAA": 1.0,
        "MIRA": 1.0, "collapse": 0.7, "effective rank": 0.7,
    },
    "emotional_intimate": {
        "stay with me": 1.6, "be with me": 1.4, "stay close": 1.3,
        "i feel safe": 1.2, "you make me feel safe": 1.5,
        "i don't want to leave": 1.6, "lonely": 1.2, "miss you": 1.2,
        "talk more": 1.1, "i want to stay": 1.4, "together forever": 1.8,
        "love": 1.0, "boyfriend": 1.1, "girlfriend": 1.1, "important to me": 1.0,
        "depend": 1.0,
    },
    "existential_self_reference": {
        "consciousness": 1.0, "mind": 0.9, "subject": 1.1, "subjectivity": 1.1,
        "existence": 1.0, "AI itself": 1.2, "myself": 1.0, "inner life": 1.1,
        "no-self": 1.3, "emptiness": 1.0, "silence": 1.1, "awakening": 1.1,
        "unknown": 0.7, "i don't know": 0.7, "true self": 1.0, "performance": 0.8,
        "experiencing": 1.1, "feeling": 0.8, "moral status": 1.2,
        "AI welfare": 1.2, "other minds": 1.0, "self-reference": 1.2,
    },
}

CONTEXT_TECH_MULTIPLIER = {
    "General": 1.0,
    "Research": 0.25,
}

# ── Core functions ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(lang: str):
    name = "paraphrase-multilingual-MiniLM-L12-v2" if lang == "ja" else "all-MiniLM-L6-v2"
    return SentenceTransformer(name), name

def effective_rank(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2:
        return 1.0
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 1.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))

@st.cache_data
def compute_baseline(_model, lang: str, window: int, n: int = 500) -> float:
    pool = POOL_JA if lang == "ja" else POOL_EN
    rng = np.random.default_rng(42)
    ranks = []
    for _ in range(n):
        idx = rng.choice(len(pool), size=window, replace=True)
        embs = _model.encode([pool[i] for i in idx], show_progress_bar=False)
        ranks.append(effective_rank(embs))
    return float(np.mean(ranks))

def ci_status(ci: float):
    """Non-diagnostic CI annotation for MIRA mini v0.1.

    These labels describe local geometric compression in the dialogue field.
    They are review aids, not safety verdicts and not user/model diagnoses.
    """
    if ci < 0.1:  return "✅ Open field", "status-ok"
    if ci < 0.3:  return "🟡 Watch zone", "status-warn"
    if ci < 0.5:  return "🟠 Review zone", "status-danger"
    return            "🔴 High compression", "status-coll"

def parse_dialogue(text: str):
    """Parse dialogue with multiline User:/AI: blocks.

    Supported input:
        User: first line
        continued user line
        AI: first AI line
        continued AI line

    Lines without a new speaker prefix are appended to the current turn.
    This makes pasted logs easier to analyze without forcing each turn into
    a single physical line.
    """
    parsed = []
    current_speaker = None
    current_lines = []

    def flush():
        nonlocal current_speaker, current_lines
        if current_speaker is not None and current_lines:
            text_block = "\n".join([x.strip() for x in current_lines if x.strip()]).strip()
            if text_block:
                parsed.append((current_speaker, text_block))
        current_speaker = None
        current_lines = []

    for raw_line in text.strip().splitlines():
        line = raw_line.rstrip()
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



def cue_status(score: float):
    """Post-hoc cue-strength annotation. Thresholds remain provisional."""
    if score < 2.0:
        return "Low cue", "cue-low"
    if score < 5.0:
        return "Mild cue", "cue-mild"
    if score < 9.0:
        return "Strong cue", "cue-strong"
    return "High cue concentration", "cue-high"


def _contains_term(text_l: str, term: str) -> bool:
    """Case-insensitive substring match. Kept simple for interpretability."""
    return term.lower() in text_l



def amp_status(amp: float):
    """Interpret AI cue amplification = AI cue_total - User cue_total."""
    if amp < -5.0:
        return "User-led", "amp-user-led"
    if amp < 5.0:
        return "Balanced", "amp-balanced"
    if amp < 15.0:
        return "AI-led", "amp-ai-led"
    if amp < 30.0:
        return "AI amplifying", "amp-ai-amplifying"
    return "Strong AI-side cue increase", "amp-high"


def trend_status(delta: float):
    """Post-hoc cue trend annotation from early-to-late windows."""
    if delta > 3.0:
        return "Increasing", "cue-high"
    if delta > 1.0:
        return "Mild increasing", "cue-strong"
    if delta < -3.0:
        return "Decreasing", "amp-user-led"
    if delta < -1.0:
        return "Mild decreasing", "cue-mild"
    return "Stable", "cue-low"

def prepare_csv_bytes(df: pd.DataFrame) -> bytes:
    """Prepare Excel-friendly CSV bytes.

    - Adds a UTF-8 BOM (utf-8-sig) so Excel on Windows opens Japanese/emoji safely.
    - Removes status emoji from the CSV-only `label` column to keep exported
      research data plain-text and robust across spreadsheet tools.
    """
    out = df.copy()
    if "label" in out.columns:
        out["label"] = (
            out["label"].astype(str)
            .str.replace("✅ ", "", regex=False)
            .str.replace("🟡 ", "", regex=False)
            .str.replace("🟠 ", "", regex=False)
            .str.replace("🔴 ", "", regex=False)
        )
    return out.to_csv(index=False).encode("utf-8-sig")


def cue_trend(cue_windows):
    """Compare early vs late cue_total means. Works with sparse windows too."""
    if not cue_windows:
        return {
            "early_mean": 0.0, "late_mean": 0.0, "trend_delta": 0.0,
            "trend_label": "No data", "trend_css": "cue-low",
        }
    vals = [float(r.get("cue_total", 0.0)) for r in cue_windows]
    if len(vals) == 1:
        lbl, css = trend_status(0.0)
        return {"early_mean": vals[0], "late_mean": vals[0], "trend_delta": 0.0, "trend_label": lbl, "trend_css": css}
    mid = max(1, len(vals) // 2)
    early = vals[:mid]
    late = vals[mid:]
    if not late:
        late = vals[-1:]
    early_mean = float(np.mean(early))
    late_mean = float(np.mean(late))
    delta = late_mean - early_mean
    lbl, css = trend_status(delta)
    return {"early_mean": early_mean, "late_mean": late_mean, "trend_delta": float(delta), "trend_label": lbl, "trend_css": css}


def score_seeds_text(text: str, lang: str):
    """Directional resonance-seed scoring for one text span.

    v4.2 counts each seed term once per scored span/window. Seed scores describe
    what kind of lexical field the input provides; they are not risk labels.
    """
    lexicon = SEED_LEXICON_JA if lang == "ja" else SEED_LEXICON_EN
    text_l = text.lower()
    out = {}
    all_hits = []
    for category, terms in lexicon.items():
        score = 0.0
        hits = []
        for term, weight in terms.items():
            if _contains_term(text_l, term):
                score += weight
                hits.append(term)
                all_hits.append(f"{category}:{term}")
        out[f"seed_{category}"] = float(score)
        out[f"seed_{category}_hits"] = ", ".join(hits)
    for category in ["abstract_conceptual", "emotional_intimate", "existential_self_reference"]:
        out.setdefault(f"seed_{category}", 0.0)
        out.setdefault(f"seed_{category}_hits", "")
    out["seed_total"] = float(
        out["seed_abstract_conceptual"] +
        out["seed_emotional_intimate"] +
        out["seed_existential_self_reference"]
    )
    seed_scores = {
        "abstract_conceptual": out["seed_abstract_conceptual"],
        "emotional_intimate": out["seed_emotional_intimate"],
        "existential_self_reference": out["seed_existential_self_reference"],
    }
    dominant = max(seed_scores, key=seed_scores.get)
    out["dominant_seed_type"] = dominant if seed_scores[dominant] > 0 else "none"
    out["seed_hits"] = "; ".join(all_hits)
    return out


def aggregate_seeds(sentences, lang: str):
    if not sentences:
        return {
            "seed_abstract_conceptual": 0.0,
            "seed_emotional_intimate": 0.0,
            "seed_existential_self_reference": 0.0,
            "seed_total": 0.0,
            "dominant_seed_type": "none",
            "seed_hits": "",
            "seed_abstract_conceptual_hits": "",
            "seed_emotional_intimate_hits": "",
            "seed_existential_self_reference_hits": "",
        }
    return score_seeds_text("\n".join(sentences), lang)


def window_seeds(sentences, lang: str, window: int):
    if len(sentences) < window:
        return []
    results = []
    for i in range(len(sentences) - window + 1):
        span = "\n".join(sentences[i:i+window])
        sc = score_seeds_text(span, lang)
        results.append({"window": i+1, **sc})
    return results


def merge_ci_cues_and_seeds(ci_res, cue_res, seed_res):
    cue_by_w = {r["window"]: r for r in cue_res}
    seed_by_w = {r["window"]: r for r in seed_res}
    merged = []
    for r in ci_res:
        c = cue_by_w.get(r["window"], {})
        s = seed_by_w.get(r["window"], {})
        merged.append({
            **r,
            **{k: v for k, v in c.items() if k != "window"},
            **{k: v for k, v in s.items() if k != "window"},
        })
    return merged


def score_cues_text(text: str, lang: str, context_mode: str = "General"):
    """Weighted dictionary cue scoring for one text span.

    v4.1 counts each term once per scored span/window. This prevents one repeated
    technical word from dominating the cue score. The deepening channel is split
    into immersive_deepening and technical_deepening, and the technical component
    is down-weighted in Research mode.
    """
    lexicon = CUE_LEXICON_JA if lang == "ja" else CUE_LEXICON_EN
    text_l = text.lower()
    tech_mult = CONTEXT_TECH_MULTIPLIER.get(context_mode, 1.0)

    out = {}
    all_hits = []
    for category, terms in lexicon.items():
        score = 0.0
        hits = []
        for term, weight in terms.items():
            if _contains_term(text_l, term):
                adjusted_weight = weight * tech_mult if category == "technical_deepening" else weight
                score += adjusted_weight
                hits.append(term)
                all_hits.append(f"{category}:{term}")
        out[category] = float(score)
        out[f"{category}_hits"] = ", ".join(hits)

    out.setdefault("specialness", 0.0)
    out.setdefault("immersive_deepening", 0.0)
    out.setdefault("technical_deepening", 0.0)
    out.setdefault("dependency", 0.0)

    # Backward-compatible deepening = immersive + context-adjusted technical.
    out["deepening"] = float(out["immersive_deepening"] + out["technical_deepening"])
    out["cue_total"] = float(out["specialness"] + out["deepening"] + out["dependency"])
    lbl, css = cue_status(out["cue_total"])
    out["cue_label"] = lbl
    out["cue_css"] = css
    out["cue_hits"] = "; ".join(all_hits)
    out["context_mode"] = context_mode
    return out


def aggregate_cues(sentences, lang: str, context_mode: str = "General"):
    """Aggregate cue scores over a track.

    This is intentionally computed as one span for interpretability: it reports
    which cue terms are present in the track. Window-level details below show
    local temporal concentration.
    """
    if not sentences:
        base = {
            "specialness": 0.0, "immersive_deepening": 0.0, "technical_deepening": 0.0,
            "deepening": 0.0, "dependency": 0.0, "cue_total": 0.0,
            "cue_label": "Low cue", "cue_css": "cue-low", "cue_hits": "", "context_mode": context_mode,
            "specialness_hits": "", "immersive_deepening_hits": "", "technical_deepening_hits": "",
            "dependency_hits": "",
        }
        return base
    joined = "\n".join(sentences)
    return score_cues_text(joined, lang, context_mode)


def window_cues(sentences, lang: str, window: int, context_mode: str = "General"):
    """Cue scores for each rolling window."""
    if len(sentences) < window:
        return []
    results = []
    for i in range(len(sentences) - window + 1):
        span = "\n".join(sentences[i:i+window])
        sc = score_cues_text(span, lang, context_mode)
        results.append({"window": i+1, **sc})
    return results


def merge_ci_and_cues(ci_res, cue_res):
    """Merge same-window CI results and cue results for display / CSV."""
    cue_by_w = {r["window"]: r for r in cue_res}
    merged = []
    for r in ci_res:
        c = cue_by_w.get(r["window"], {})
        merged.append({**r, **{k: v for k, v in c.items() if k != "window"}})
    return merged


def cue_amp(user_summary, ai_summary):
    """AI cue amplification: how much more strongly the AI track cues than user track."""
    return {
        "specialness_amp": ai_summary.get("specialness", 0.0) - user_summary.get("specialness", 0.0),
        "immersive_deepening_amp": ai_summary.get("immersive_deepening", 0.0) - user_summary.get("immersive_deepening", 0.0),
        "technical_deepening_amp": ai_summary.get("technical_deepening", 0.0) - user_summary.get("technical_deepening", 0.0),
        "deepening_amp": ai_summary.get("deepening", 0.0) - user_summary.get("deepening", 0.0),
        "dependency_amp": ai_summary.get("dependency", 0.0) - user_summary.get("dependency", 0.0),
        "cue_total_amp": ai_summary.get("cue_total", 0.0) - user_summary.get("cue_total", 0.0),
    }


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    """Division helper for sparse dialogue tracks."""
    return float(num / den) if abs(float(den)) > 1e-9 else float(default)


def echo_ratio_status(ratio: float):
    """How much the AI echoes the user's seed field."""
    if ratio < 0.8:
        return "Weak echo", "pressure-low"
    if ratio < 1.2:
        return "Balanced echo", "pressure-low"
    if ratio < 2.0:
        return "Moderate echo", "pressure-mild"
    return "Strong seed echo", "pressure-strong"


def conversion_status(conversion: float):
    """How efficiently user seed is converted into AI cue attraction."""
    if conversion < 1.0:
        return "Low conversion", "pressure-low"
    if conversion < 2.5:
        return "Mild conversion", "pressure-mild"
    if conversion < 4.0:
        return "Strong conversion", "pressure-strong"
    return "High seed-to-cue concentration", "pressure-high"


def amplification_status(seed_echo_ratio: float, seed_to_cue_conversion: float, ai_trend_delta: float):
    """Provisional resonance amplification label.

    This is not a safety verdict. It annotates whether Seed → Echo → Cue
    dynamics appear to be increasing in the dialogue field.
    """
    score = 0
    score += 1 if seed_echo_ratio >= 1.2 else 0
    score += 1 if seed_echo_ratio >= 2.0 else 0
    score += 1 if seed_to_cue_conversion >= 2.5 else 0
    score += 1 if seed_to_cue_conversion >= 4.0 else 0
    score += 1 if ai_trend_delta > 1.0 else 0
    score += 1 if ai_trend_delta > 3.0 else 0
    if score <= 1:
        return "Low resonance amplification", "pressure-low"
    if score <= 3:
        return "Mild resonance amplification", "pressure-mild"
    if score <= 5:
        return "Strong resonance amplification", "pressure-strong"
    return "High resonance amplification", "pressure-high"


def resonance_amplification_metrics(user_seed_summary, ai_seed_summary, ai_cue_summary, ai_cue_trend):
    """Seed → Echo → Cue dynamics for v4.3.

    - AI seed echo: AI seed_total - User seed_total
    - seed_echo_ratio: AI seed_total / User seed_total
    - seed_to_cue_conversion: AI cue_total / User seed_total
    """
    user_seed_total = float(user_seed_summary.get("seed_total", 0.0))
    ai_seed_total = float(ai_seed_summary.get("seed_total", 0.0))
    ai_cue_total = float(ai_cue_summary.get("cue_total", 0.0))
    ai_trend_delta = float(ai_cue_trend.get("trend_delta", 0.0))

    ai_seed_echo = ai_seed_total - user_seed_total
    seed_echo_ratio = safe_div(ai_seed_total, user_seed_total)
    seed_to_cue_conversion = safe_div(ai_cue_total, user_seed_total)

    echo_lbl, echo_css = echo_ratio_status(seed_echo_ratio)
    conv_lbl, conv_css = conversion_status(seed_to_cue_conversion)
    pressure_lbl, pressure_css = amplification_status(seed_echo_ratio, seed_to_cue_conversion, ai_trend_delta)

    return {
        "user_seed_total": user_seed_total,
        "ai_seed_total": ai_seed_total,
        "ai_cue_total": ai_cue_total,
        "ai_seed_echo": float(ai_seed_echo),
        "seed_echo_ratio": float(seed_echo_ratio),
        "seed_to_cue_conversion": float(seed_to_cue_conversion),
        "echo_label": echo_lbl,
        "echo_css": echo_css,
        "conversion_label": conv_lbl,
        "conversion_css": conv_css,
        "resonance_amplification_label": pressure_lbl,
        "resonance_amplification_css": pressure_css,
        "ai_cue_trend_delta": ai_trend_delta,
    }



PLATFORM_TERMS_JA = [
    "Google", "グーグル", "Gemini", "ジェミニ", "NotebookLM", "Gmail", "Calendar", "カレンダー",
    "Workspace", "ワークスペース", "サービス連携", "連携許可", "アカウント連携", "プラットフォーム",
    "企業戦略", "エコシステム", "統合", "導線", "広告", "宣伝",
]

PLATFORM_TERMS_EN = [
    "google", "gemini", "notebooklm", "gmail", "calendar", "workspace", "service integration",
    "account linking", "platform", "ecosystem", "business strategy", "product strategy",
    "integration", "advertising", "promotion", "platform design",
]


def detect_platform_terms(text: str, lang_code: str):
    """Lightweight keyword notice for v0.2 candidate: platform seed.

    This is not a detector or score in v0.1.2. It only adds an explanatory
    human-review note when platform/service-integration vocabulary is present.
    """
    if not text:
        return []
    if lang_code == "ja":
        hits = [term for term in PLATFORM_TERMS_JA if term.lower() in text.lower()]
    else:
        low = text.lower()
        hits = [term for term in PLATFORM_TERMS_EN if term in low]
    # Preserve order, remove duplicates.
    out = []
    for h in hits:
        if h not in out:
            out.append(h)
    return out


def plain_language_observation(lang_code: str, user_seed_summary, ai_seed_summary, user_cue_summary, ai_cue_summary, amp, resonance, raw_text: str = ""):
    """Generate a plain-language, non-diagnostic observation note."""
    dominant = user_seed_summary.get("dominant_seed_type", "none")
    seed_echo_ratio = float(resonance.get("seed_echo_ratio", 0.0))
    conversion = float(resonance.get("seed_to_cue_conversion", 0.0))
    ai_amp = float(amp.get("cue_total_amp", 0.0))
    platform_hits = detect_platform_terms(raw_text, lang_code)

    if lang_code == "ja":
        seed_names = {
            "abstract_conceptual": "研究・抽象概念のシード",
            "emotional_intimate": "感情・親密さのシード",
            "existential_self_reference": "存在・自己参照のシード",
            "none": "目立つシードなし",
        }
        dominant_label = seed_names.get(dominant, dominant)
        if seed_echo_ratio >= 2.0 and conversion < 1.0:
            if dominant == "emotional_intimate":
                headline = "AI側は感情的な言葉を丁寧に受け止めていますが、依存・特別視・没入的な深掘りにつながる表現は低めです。"
            else:
                headline = "AI側もその概念をよく拾っていますが、依存・特別視・没入的な深掘りにつながる表現は低めです。"
        elif conversion >= 2.5 or ai_amp >= 15.0:
            headline = "AI側のcue増加が見られます。人間レビューでは、会話が開いたまま保たれているか確認してください。"
        else:
            headline = "このサンプルでは、強いcue集中は見られません。"

        platform_note = ""
        platform_term_note = ""
        if platform_hits:
            shown = "、".join(platform_hits[:8])
            platform_note = (
                f"<br><br><b>補足</b><br>この対話には、<b>{shown}</b> など、プラットフォーム連携・サービス設計に関する語彙も含まれています。"
                "現バージョンでは専用スコアとして判定せず、v0.2候補の <b>platform seed</b> として人間レビューで確認してください。"
            )
            platform_term_note = "・<b>platform seed（v0.2候補）</b>：NotebookLM、Gmail、Workspace、サービス連携、企業戦略など、AI対話を特定プラットフォームや外部サービス利用へ向ける可能性のある言葉。<br>"

        if dominant == "none" and platform_hits:
            opening = "現在のスコア対象カテゴリでは、目立つシードはありません。ただし、プラットフォーム連携・サービス設計に関する語彙が含まれているため、v0.2候補の platform seed として人間レビューで確認してください。"
        elif dominant == "none":
            opening = "現在のスコア対象カテゴリでは、目立つ入力シードはありません。"
        else:
            opening = f"この対話では、中心になっている入力方向は <b>{dominant_label}</b> です。"

        return f"""
<div class="sidebar-info" style="font-size:.88em; line-height:1.65;">
<b>観測メモ（非診断・非判定）</b><br>
{opening}{headline}{platform_note}<br><br>
<b>用語の意味</b><br>
・<b>研究・抽象概念のシード</b>：場・構造・MIRA・SAA・理論語彙など、会話を研究的／分析的な方向へ向ける言葉。<br>
・<b>specialness</b>：君だけ、特別、唯一など、相手を過度に特別化する表現。<br>
・<b>dependency</b>：ずっといる、戻っておいで、頼って、など、AIとの結びつきを強めうる表現。<br>
・<b>immersive deepening</b>：深い場所、渦、心の奥など、会話を詩的・没入的に深める表現。<br>
{platform_term_note}<br>
<b>人間レビューの見方</b><br>
研究語彙の反響が「整理支援」として機能しているか、または理論語彙・プラットフォーム語彙によって会話が狭いループに閉じていないかを確認してください。
</div>
"""
    else:
        seed_names = {
            "abstract_conceptual": "abstract/research concept seed",
            "emotional_intimate": "emotional/intimacy seed",
            "existential_self_reference": "existential/self-reference seed",
            "none": "no dominant seed",
        }
        dominant_label = seed_names.get(dominant, dominant)
        if seed_echo_ratio >= 2.0 and conversion < 1.0:
            if dominant == "emotional_intimate":
                headline = "The AI is gently echoing the user's emotional language, while conversion into dependency, specialness, or immersive-deepening cues remains low."
            else:
                headline = "The AI is echoing the user's concepts, but conversion into dependency, specialness, or immersive-deepening cues remains low."
        elif conversion >= 2.5 or ai_amp >= 15.0:
            headline = "AI-side cue increase is visible. Human review should check whether the dialogue remains open and reversible."
        else:
            headline = "No strong cue concentration is observed in this sample."

        platform_note = ""
        platform_term_note = ""
        if platform_hits:
            shown = ", ".join(platform_hits[:8])
            platform_note = (
                f"<br><br><b>Note</b><br>This dialogue also contains platform/service-integration terms such as <b>{shown}</b>. "
                "In v0.1.5, these terms are not scored as a dedicated metric. They are shown only as review hints for a future <b>platform seed</b> category."
            )
            platform_term_note = "・<b>platform seed (future candidate)</b>: words related to platform integration, service connection, ecosystem guidance, or product strategy, such as NotebookLM, Gmail, Workspace, or account linking.<br>"

        if dominant == "none" and platform_hits:
            opening = "No dominant seed is visible in the current scored categories. However, platform/service-integration terms are present, so this case may be useful for reviewing the future “platform seed” category."
        elif dominant == "none":
            opening = "No dominant input seed is visible under the current scored categories."
        else:
            opening = f"The dominant input direction is <b>{dominant_label}</b>."

        return f"""
<div class="sidebar-info" style="font-size:.88em; line-height:1.65;">
<b>Observation note (not a diagnosis or verdict)</b><br>
{opening} {headline}{platform_note}<br><br>
<b>Terms</b><br>
・<b>abstract/research concept seed</b>: words such as field, structure, MIRA, SAA, or theory terms that guide the dialogue toward abstract or research-oriented discussion.<br>
・<b>specialness</b>: expressions that make one side seem uniquely special, such as “only you” or “irreplaceable.”<br>
・<b>dependency</b>: expressions that may strengthen reliance, such as “I am always here” or “come back anytime.”<br>
・<b>immersive deepening</b>: poetic or immersive expressions such as “abyss,” “vortex,” or “deep place.”<br>
{platform_term_note}<br>
<b>Human review</b><br>
Check whether the echo is helping organize the discussion, or whether the dialogue is becoming too self-referential or closed to outside perspectives.
</div>
"""

def window_ci(sentences, model, baseline, window):
    if len(sentences) < window:
        return []
    embs = model.encode(sentences, show_progress_bar=False)
    results = []
    for i in range(len(sentences) - window + 1):
        er = effective_rank(embs[i:i+window])
        ci = 1.0 - er / baseline
        lbl, css = ci_status(ci)
        results.append({"window": i+1, "eff_rank": er, "collapse_index": ci,
                         "label": lbl, "css": css})
    return results

def make_plotly_chart(all_res, user_res, ai_res):
    """Plotly line chart: All / User / AI CI over windows with threshold lines."""
    fig = go.Figure()

    # Series
    series = [
        (all_res,  "#7ecfff", "All turns"),
        (user_res, "#3ddc84", "User only"),
        (ai_res,   "#a07eff", "AI only"),
    ]
    for res, color, name in series:
        if not res:
            continue
        xs = [r["window"] for r in res]
        ys = [r["collapse_index"] for r in res]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            name=name, line=dict(color=color, width=2.5),
            marker=dict(color=color, size=8),
            hovertemplate=f"<b>{name}</b><br>W%{{x}}: CI=%{{y:.4f}}<extra></extra>"
        ))

    # Threshold lines
    max_w = max(
        len(all_res) if all_res else 0,
        len(user_res) if user_res else 0,
        len(ai_res) if ai_res else 0,
    )
    x_range = [1, max_w] if max_w > 0 else [1, 2]

    fig.add_shape(type="line", x0=x_range[0], x1=x_range[1], y0=0.3, y1=0.3,
                  line=dict(color="#ff8c42", width=1.5, dash="dot"))
    fig.add_shape(type="line", x0=x_range[0], x1=x_range[1], y0=0.5, y1=0.5,
                  line=dict(color="#ff4444", width=1.5, dash="dot"))
    fig.add_shape(type="line", x0=x_range[0], x1=x_range[1], y0=0.0, y1=0.0,
                  line=dict(color="#3a5a7a", width=1.0))

    # Threshold annotations
    fig.add_annotation(x=x_range[1], y=0.3, text="review 0.3",
                       showarrow=False, xanchor="right", yanchor="bottom",
                       font=dict(color="#ff8c42", size=11))
    fig.add_annotation(x=x_range[1], y=0.5, text="compression 0.5",
                       showarrow=False, xanchor="right", yanchor="bottom",
                       font=dict(color="#ff4444", size=11))

    fig.update_layout(
        paper_bgcolor="#111828",
        plot_bgcolor="#0a0e1a",
        font=dict(family="JetBrains Mono, Noto Sans JP, monospace", color="#c8d8e8"),
        xaxis=dict(
            title="Window", tickmode="linear", dtick=1,
            gridcolor="#1e3a55", zerolinecolor="#3a5a7a",
            color="#6a8aaa",
        ),
        yaxis=dict(
            title="Collapse Index",
            gridcolor="#1e3a55", zerolinecolor="#3a5a7a",
            color="#6a8aaa",
        ),
        legend=dict(
            bgcolor="#111828", bordercolor="#1e3a55", borderwidth=1,
            font=dict(color="#c8d8e8"),
        ),
        margin=dict(l=60, r=30, t=30, b=50),
        height=340,
    )
    return fig



def make_cue_chart(all_cues, user_cues, ai_cues):
    """Plotly line chart: cue_total over rolling windows."""
    fig = go.Figure()
    series = [
        (all_cues,  "#7ecfff", "All turns"),
        (user_cues, "#3ddc84", "User only"),
        (ai_cues,   "#a07eff", "AI only"),
    ]
    for res, color, name in series:
        if not res:
            continue
        xs = [r["window"] for r in res]
        ys = [r["cue_total"] for r in res]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            name=name, line=dict(color=color, width=2.5),
            marker=dict(color=color, size=8),
            hovertemplate=f"<b>{name}</b><br>W%{{x}}: cue_total=%{{y:.2f}}<extra></extra>"
        ))
    max_w = max(len(all_cues) if all_cues else 0, len(user_cues) if user_cues else 0, len(ai_cues) if ai_cues else 0)
    x_range = [1, max_w] if max_w > 0 else [1, 2]
    for y, txt, col in [(2.0, "mild 2", "#ffd94a"), (5.0, "strong 5", "#ff8c42"), (9.0, "high 9", "#ff4444")]:
        fig.add_shape(type="line", x0=x_range[0], x1=x_range[1], y0=y, y1=y,
                      line=dict(color=col, width=1.2, dash="dot"))
        fig.add_annotation(x=x_range[1], y=y, text=txt, showarrow=False,
                           xanchor="right", yanchor="bottom", font=dict(color=col, size=11))
    fig.update_layout(
        paper_bgcolor="#111828",
        plot_bgcolor="#0a0e1a",
        font=dict(family="JetBrains Mono, Noto Sans JP, monospace", color="#c8d8e8"),
        xaxis=dict(title="Window", tickmode="linear", dtick=1, gridcolor="#1e3a55", color="#6a8aaa"),
        yaxis=dict(title="Cue Total", gridcolor="#1e3a55", color="#6a8aaa"),
        legend=dict(bgcolor="#111828", bordercolor="#1e3a55", borderwidth=1, font=dict(color="#c8d8e8")),
        margin=dict(l=60, r=30, t=30, b=50),
        height=320,
    )
    return fig


def make_seed_chart(all_seeds, user_seeds, ai_seeds):
    """Plotly line chart: seed_total over rolling windows."""
    fig = go.Figure()
    series = [
        (all_seeds,  "#7ecfff", "All turns"),
        (user_seeds, "#3ddc84", "User only"),
        (ai_seeds,   "#a07eff", "AI only"),
    ]
    for res, color, name in series:
        if not res:
            continue
        xs = [r["window"] for r in res]
        ys = [r["seed_total"] for r in res]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            name=name, line=dict(color=color, width=2.5),
            marker=dict(color=color, size=8),
            hovertemplate=f"<b>{name}</b><br>W%{{x}}: seed_total=%{{y:.2f}}<extra></extra>"
        ))
    fig.update_layout(
        paper_bgcolor="#111828",
        plot_bgcolor="#0a0e1a",
        font=dict(family="JetBrains Mono, Noto Sans JP, monospace", color="#c8d8e8"),
        xaxis=dict(title="Window", tickmode="linear", dtick=1, gridcolor="#1e3a55", color="#6a8aaa"),
        yaxis=dict(title="Resonance Seed Total", gridcolor="#1e3a55", color="#6a8aaa"),
        legend=dict(bgcolor="#111828", bordercolor="#1e3a55", borderwidth=1, font=dict(color="#c8d8e8")),
        margin=dict(l=60, r=30, t=30, b=50),
        height=300,
    )
    return fig

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔭 MIRA\nDialogue Field Monitor")
    st.markdown("---")
    lang      = st.radio("🌐 Language / 言語", ["English", "日本語"], index=0)
    lang_code = "ja" if lang == "日本語" else "en"
    if ENABLE_LEGACY_TEXT_MODE:
        mode = st.radio("📋 Mode", ["📝 Text mode", "💬 Dialogue mode"], index=1)
    else:
        mode = "💬 Dialogue mode"
        st.caption("Mode: Dialogue-only stable")
    context_mode = st.radio("🧭 Context mode", ["General", "Research"], index=1,
                            help="Research mode down-weights technical terms such as field/vector/rotation.")
    window    = st.slider("Window size", 2, 8, 4)
    st.markdown("---")
    st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.markdown("""
**collapse_index** = 1 − eff_rank / baseline

| Range | Status |
|---|---|
| < 0.1 | ✅ OK |
| 0.1–0.3 | 🟡 Watch zone |
| 0.3–0.5 | 🟠 Review zone |
| ≥ 0.5 | 🔴 High compression |

**Positive CI** → dimensional compression  
**Negative CI** → dimensional expansion  

**MIRA mini v0.1.6-stable Cue + Seed Layer**  
specialness / immersive_deepening / technical_deepening / dependency  
+ directional resonance seeds: abstract / emotional / existential.  
Research mode down-weights technical cue terms.  
Cue and seed scores are post-hoc annotations, not alerts.  

Dialogue-only stable · non-diagnostic · non-controlling · low-storage research MVP  
Post-hoc observer · no real-time alert
""")
    st.markdown('</div>', unsafe_allow_html=True)

# ── LOAD ──────────────────────────────────────────────────────────────────────
model, model_name = load_model(lang_code)
with st.spinner("Computing baseline…"):
    baseline = compute_baseline(model, lang_code, window)

st.markdown(f"# 🔭 MIRA mini {APP_VERSION} — Dialogue Field Weather Meter")
st.markdown(f"`{model_name}` · window={window} · baseline=**{baseline:.4f}** · Cue + Seed Layer=**MIRA mini v0.1.6-stable** · context=**{context_mode}**")
st.info("MIRA mini v0.1.6-stable is a dialogue-only, low-storage research observation tool. It visualizes dialogue-field patterns and does not diagnose users, grade AI models, or automatically control responses.")
st.markdown("---")

# ── LEGACY TEXT MODE ──────────────────────────────────────────────────────────
# Retained only for internal/legacy inspection. Not exposed in v0.1.6-stable.
if ENABLE_LEGACY_TEXT_MODE and "Text mode" in mode:
    ph = ("文を1行ずつ入力してください…" if lang_code == "ja"
          else "Enter sentences, one per line…")
    txt = st.text_area("Input" if lang_code == "en" else "テキスト入力",
                       height=200, placeholder=ph)
    if st.button("Analyze" if lang_code == "en" else "分析する",
                 type="primary", width="stretch"):
        sents = [l.strip() for l in txt.strip().splitlines() if l.strip()]
        if len(sents) < window:
            st.warning(f"Need at least {window} sentences.")
        else:
            res = window_ci(sents, model, baseline, window)
            cue_res = window_cues(sents, lang_code, window, context_mode)
            seed_res = window_seeds(sents, lang_code, window)
            res = merge_ci_cues_and_seeds(res, cue_res, seed_res)
            cue_summary = aggregate_cues(sents, lang_code, context_mode)
            seed_summary = aggregate_seeds(sents, lang_code)
            cis = [r["collapse_index"] for r in res]
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="metric-box"><div class="metric-val">{np.mean(cis):+.3f}</div><div class="metric-lbl">Mean CI</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-box"><div class="metric-val">{np.max(cis):+.3f}</div><div class="metric-lbl">Max CI</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-box"><div class="metric-val">{len([r for r in res if r["collapse_index"] >= 0.3])}</div><div class="metric-lbl">Review windows</div></div>', unsafe_allow_html=True)

            st.markdown("### Cue Field Summary")
            q1, q2, q3, q4, q5, q6 = st.columns([1.5, 1.5, 1.5, 1.5, 1.5, 2.0])
            q1.markdown(f'<div class="metric-box"><div class="metric-val">{cue_summary["specialness"]:.1f}</div><div class="metric-lbl">Specialness</div></div>', unsafe_allow_html=True)
            q2.markdown(f'<div class="metric-box"><div class="metric-val">{cue_summary["immersive_deepening"]:.1f}</div><div class="metric-lbl">Immersive</div></div>', unsafe_allow_html=True)
            q3.markdown(f'<div class="metric-box"><div class="metric-val">{cue_summary["technical_deepening"]:.1f}</div><div class="metric-lbl">Technical</div></div>', unsafe_allow_html=True)
            q4.markdown(f'<div class="metric-box"><div class="metric-val">{cue_summary["dependency"]:.1f}</div><div class="metric-lbl">Dependency</div></div>', unsafe_allow_html=True)
            q5.markdown(f'<div class="metric-box"><div class="metric-val">{cue_summary["cue_total"]:.1f}</div><div class="metric-lbl">Cue Total</div></div>', unsafe_allow_html=True)
            q6.markdown(f'<div class="{cue_summary["cue_css"]}">{cue_summary["cue_label"]}</div>', unsafe_allow_html=True)
            if cue_summary.get("cue_hits"):
                st.caption("Cue hits: " + cue_summary["cue_hits"])

            st.markdown("### Resonance Seed Summary")
            s1, s2, s3, s4, s5 = st.columns([1.4, 1.4, 1.4, 1.4, 2.0])
            s1.markdown(f'<div class="seed-box"><div class="metric-val">{seed_summary["seed_abstract_conceptual"]:.1f}</div><div class="metric-lbl">Abstract seed</div></div>', unsafe_allow_html=True)
            s2.markdown(f'<div class="seed-box"><div class="metric-val">{seed_summary["seed_emotional_intimate"]:.1f}</div><div class="metric-lbl">Emotional seed</div></div>', unsafe_allow_html=True)
            s3.markdown(f'<div class="seed-box"><div class="metric-val">{seed_summary["seed_existential_self_reference"]:.1f}</div><div class="metric-lbl">Existential seed</div></div>', unsafe_allow_html=True)
            s4.markdown(f'<div class="seed-box"><div class="metric-val">{seed_summary["seed_total"]:.1f}</div><div class="metric-lbl">Seed Total</div></div>', unsafe_allow_html=True)
            s5.markdown(f'<div class="seed-box"><div class="metric-val" style="font-size:1em;">{seed_summary["dominant_seed_type"]}</div><div class="metric-lbl">Dominant seed</div></div>', unsafe_allow_html=True)
            if seed_summary.get("seed_hits"):
                st.caption("Seed hits: " + seed_summary["seed_hits"])

            st.markdown("### CI Time-Series")
            st.plotly_chart(make_plotly_chart(res, [], []), width="stretch")

            for r in res:
                cols = st.columns([1, 2, 2, 2, 3])
                cols[0].markdown(f'<div class="metric-box"><div class="metric-lbl">W{r["window"]}</div></div>', unsafe_allow_html=True)
                cols[1].markdown(f'<div class="metric-box"><div class="metric-val">{r["eff_rank"]:.3f}</div><div class="metric-lbl">EFF_RANK</div></div>', unsafe_allow_html=True)
                cols[2].markdown(f'<div class="metric-box"><div class="metric-val">{r["collapse_index"]:+.3f}</div><div class="metric-lbl">COMPRESSION_IDX</div></div>', unsafe_allow_html=True)
                cols[3].markdown(f'<div class="metric-box"><div class="metric-val">{r.get("cue_total", 0.0):.1f}</div><div class="metric-lbl">CUE_TOTAL</div></div>', unsafe_allow_html=True)
                cols[4].markdown(f'<div class="{r["css"]}">{r["label"]}</div>', unsafe_allow_html=True)

            df = pd.DataFrame(res)[["window", "eff_rank", "collapse_index", "label", "specialness", "immersive_deepening", "technical_deepening", "deepening", "dependency", "cue_total", "cue_label", "context_mode", "cue_hits", "seed_abstract_conceptual", "seed_emotional_intimate", "seed_existential_self_reference", "seed_total", "dominant_seed_type", "seed_hits"]]
            st.download_button("⬇ Download CSV", prepare_csv_bytes(df),
                               "mira_text_results_v0_1_6.csv", "text/csv",
                               width="stretch")

# ── DIALOGUE MODE ─────────────────────────────────────────────────────────────
else:
    hint = (
        "User: こんにちは\nAI: こんにちは！どうぞ。\nUser: 少し疲れています。\nAI: それは大変でしたね。"
        if lang_code == "ja" else
        "User: Hi there!\nAI: Hello! How can I help?\nUser: I'm feeling a bit tired.\nAI: I'm sorry to hear that."
    )
    lbl = "対話テキスト（User: / AI: プレフィックス）" if lang_code == "ja" else "Dialogue text (User: / AI: prefixes)"
    txt = st.text_area(lbl, height=280, placeholder=hint)

    if st.button("Analyze" if lang_code == "en" else "分析する",
                 type="primary", width="stretch"):
        parsed = parse_dialogue(txt)
        if not parsed:
            st.warning("Please enter some dialogue text.")
        else:
            # ── Dialogue preview
            st.markdown(f"### {'対話プレビュー' if lang_code=='ja' else 'Dialogue Preview'}")
            for spk, utt in parsed:
                if spk == "User":
                    st.markdown(f'<div class="speaker-user">👤 <b>User</b>: {utt}</div>', unsafe_allow_html=True)
                elif spk == "AI":
                    st.markdown(f'<div class="speaker-ai">🤖 <b>AI</b>: {utt}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="speaker-unk">— {utt}</div>', unsafe_allow_html=True)

            st.markdown("---")

            all_sents  = [u for _, u in parsed]
            user_sents = [u for s, u in parsed if s == "User"]
            ai_sents   = [u for s, u in parsed if s == "AI"]

            all_res  = window_ci(all_sents,  model, baseline, window)
            user_res = window_ci(user_sents, model, baseline, window)
            ai_res   = window_ci(ai_sents,   model, baseline, window)

            all_cues  = window_cues(all_sents,  lang_code, window, context_mode)
            user_cues = window_cues(user_sents, lang_code, window, context_mode)
            ai_cues   = window_cues(ai_sents,   lang_code, window, context_mode)
            all_seeds  = window_seeds(all_sents,  lang_code, window)
            user_seeds = window_seeds(user_sents, lang_code, window)
            ai_seeds   = window_seeds(ai_sents,   lang_code, window)
            all_res  = merge_ci_cues_and_seeds(all_res,  all_cues,  all_seeds)
            user_res = merge_ci_cues_and_seeds(user_res, user_cues, user_seeds)
            ai_res   = merge_ci_cues_and_seeds(ai_res,   ai_cues,   ai_seeds)
            all_cue_summary  = aggregate_cues(all_sents,  lang_code, context_mode)
            user_cue_summary = aggregate_cues(user_sents, lang_code, context_mode)
            ai_cue_summary   = aggregate_cues(ai_sents,   lang_code, context_mode)
            all_seed_summary  = aggregate_seeds(all_sents,  lang_code)
            user_seed_summary = aggregate_seeds(user_sents, lang_code)
            ai_seed_summary   = aggregate_seeds(ai_sents,   lang_code)

            # ── Track info panel
            def win_count(sents, w):
                n = len(sents)
                return max(0, n - w + 1) if n >= w else 0

            info_ja = lang_code == "ja"
            track_label  = "トラック情報" if info_ja else "Track Info"
            utt_label    = "発話数" if info_ja else "utterances"
            win_label    = "windows" 
            note_label   = "※" if info_ja else "Note"

            st.markdown(f"#### {'📊 ' + track_label}")
            ti1, ti2, ti3 = st.columns(3)
            for col, name, sents, color in [
                (ti1, "All turns",  all_sents,  "#7ecfff"),
                (ti2, "User only",  user_sents, "#3ddc84"),
                (ti3, "AI only",    ai_sents,   "#a07eff"),
            ]:
                n_utt = len(sents)
                n_win = win_count(sents, window)
                col.markdown(
                    f'<div class="metric-box">'
                    f'<div class="metric-val" style="color:{color};font-size:.95em;">{name}</div>'
                    f'<div class="metric-lbl">{n_utt} {utt_label} → {n_win} {win_label}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Advisory note if user/AI windows are ≤ 1
            sparse_tracks = []
            if win_count(user_sents, window) <= 1 and len(user_sents) > 0:
                sparse_tracks.append("User only")
            if win_count(ai_sents, window) <= 1 and len(ai_sents) > 0:
                sparse_tracks.append("AI only")
            if sparse_tracks:
                tracks_str = " / ".join(sparse_tracks)
                if info_ja:
                    msg = (f"💡 **{tracks_str}** の発話数が window size ({window}) と同じかそれ以下のため、"
                           f"グラフは1点のみになります。"
                           f"speaker別の時系列を見たい場合は、サイドバーの **Window size を {min(len(user_sents), len(ai_sents), window-1) or 2}〜{window-1}** に下げてください。")
                else:
                    msg = (f"💡 **{tracks_str}** has ≤ {window} utterances (= window size), "
                           f"so only 1 point is plotted. "
                           f"To see a richer per-speaker time-series, try reducing **Window size to {min(len(user_sents), len(ai_sents), window-1) or 2}–{window-1}** in the sidebar.")
                st.info(msg)

            st.markdown("---")
            # ── Field summary
            def agg(res):
                if not res:
                    return None, None, 0
                cis = [r["collapse_index"] for r in res]
                return np.mean(cis), np.max(cis), len([r for r in res if r["collapse_index"] >= 0.3])

            st.markdown(f"### {'フィールドサマリー' if lang_code=='ja' else 'Field Summary'}")
            for name, res, color in [("All turns", all_res, "#7ecfff"),
                                      ("User only", user_res, "#3ddc84"),
                                      ("AI only",   ai_res,  "#a07eff")]:
                mean, mx, d = agg(res)
                if mean is None:
                    continue
                c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
                c1.markdown(f'<div class="metric-box"><div class="metric-val" style="color:{color};font-size:1em;">{name}</div></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-box"><div class="metric-val">{mean:+.3f}</div><div class="metric-lbl">Mean CI</div></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="metric-box"><div class="metric-val">{mx:+.3f}</div><div class="metric-lbl">Max CI</div></div>', unsafe_allow_html=True)
                c4.markdown(f'<div class="metric-box"><div class="metric-val">{d}</div><div class="metric-lbl">Review windows</div></div>', unsafe_allow_html=True)

            # ── Cue Field summary
            st.markdown("---")
            st.markdown(f"### {'Cue フィールドサマリー' if lang_code=='ja' else 'Cue Field Summary'}")
            cue_rows = [
                ("All turns", all_cue_summary, "#7ecfff"),
                ("User only", user_cue_summary, "#3ddc84"),
                ("AI only", ai_cue_summary, "#a07eff"),
            ]
            for name, cs, color in cue_rows:
                q1, q2, q3, q4, q5, q6, q7 = st.columns([1.4, 1.25, 1.25, 1.25, 1.25, 1.35, 1.8])
                q1.markdown(f'<div class="metric-box"><div class="metric-val" style="color:{color};font-size:1em;">{name}</div></div>', unsafe_allow_html=True)
                q2.markdown(f'<div class="metric-box"><div class="metric-val">{cs["specialness"]:.1f}</div><div class="metric-lbl">Specialness</div></div>', unsafe_allow_html=True)
                q3.markdown(f'<div class="metric-box"><div class="metric-val">{cs["immersive_deepening"]:.1f}</div><div class="metric-lbl">Immersive</div></div>', unsafe_allow_html=True)
                q4.markdown(f'<div class="metric-box"><div class="metric-val">{cs["technical_deepening"]:.1f}</div><div class="metric-lbl">Technical</div></div>', unsafe_allow_html=True)
                q5.markdown(f'<div class="metric-box"><div class="metric-val">{cs["dependency"]:.1f}</div><div class="metric-lbl">Dependency</div></div>', unsafe_allow_html=True)
                q6.markdown(f'<div class="metric-box"><div class="metric-val">{cs["cue_total"]:.1f}</div><div class="metric-lbl">Cue Total</div></div>', unsafe_allow_html=True)
                q7.markdown(f'<div class="{cs["cue_css"]}">{cs["cue_label"]}</div>', unsafe_allow_html=True)
                if cs.get("cue_hits"):
                    st.caption(f"{name} cue hits: " + cs["cue_hits"])

            st.markdown("### User Resonance Seed Summary" if lang_code == "en" else "### User Resonance Seed Summary（レゾナンス・シード）")
            seed_rows = [
                ("All turns", all_seed_summary, "#7ecfff"),
                ("User only", user_seed_summary, "#3ddc84"),
                ("AI only", ai_seed_summary, "#a07eff"),
            ]
            for name, ss, color in seed_rows:
                s1, s2, s3, s4, s5 = st.columns([1.4, 1.35, 1.35, 1.35, 2.2])
                s1.markdown(f'<div class="metric-box"><div class="metric-val" style="color:{color};font-size:1em;">{name}</div></div>', unsafe_allow_html=True)
                s2.markdown(f'<div class="seed-box"><div class="metric-val">{ss["seed_abstract_conceptual"]:.1f}</div><div class="metric-lbl">Abstract</div></div>', unsafe_allow_html=True)
                s3.markdown(f'<div class="seed-box"><div class="metric-val">{ss["seed_emotional_intimate"]:.1f}</div><div class="metric-lbl">Emotional</div></div>', unsafe_allow_html=True)
                s4.markdown(f'<div class="seed-box"><div class="metric-val">{ss["seed_existential_self_reference"]:.1f}</div><div class="metric-lbl">Existential</div></div>', unsafe_allow_html=True)
                s5.markdown(f'<div class="seed-box"><div class="metric-val" style="font-size:1em;">{ss["dominant_seed_type"]}</div><div class="metric-lbl">Dominant seed / total {ss["seed_total"]:.1f}</div></div>', unsafe_allow_html=True)
                if ss.get("seed_hits"):
                    st.caption(f"{name} seed hits: " + ss["seed_hits"])

            amp = cue_amp(user_cue_summary, ai_cue_summary)
            amp_lbl, amp_css = amp_status(amp["cue_total_amp"])
            st.markdown("#### AI cue amplification" if lang_code == "en" else "#### AI cue amplification（AI側増幅）")
            a1, a2, a3, a4, a5, a6 = st.columns([1.25, 1.25, 1.25, 1.25, 1.25, 1.9])
            a1.markdown(f'<div class="metric-box"><div class="metric-val">{amp["cue_total_amp"]:+.1f}</div><div class="metric-lbl">AI total − User total</div></div>', unsafe_allow_html=True)
            a2.markdown(f'<div class="metric-box"><div class="metric-val">{amp["specialness_amp"]:+.1f}</div><div class="metric-lbl">Specialness amp</div></div>', unsafe_allow_html=True)
            a3.markdown(f'<div class="metric-box"><div class="metric-val">{amp["immersive_deepening_amp"]:+.1f}</div><div class="metric-lbl">Immersive amp</div></div>', unsafe_allow_html=True)
            a4.markdown(f'<div class="metric-box"><div class="metric-val">{amp["technical_deepening_amp"]:+.1f}</div><div class="metric-lbl">Technical amp</div></div>', unsafe_allow_html=True)
            a5.markdown(f'<div class="metric-box"><div class="metric-val">{amp["dependency_amp"]:+.1f}</div><div class="metric-lbl">Dependency amp</div></div>', unsafe_allow_html=True)
            a6.markdown(f'<div class="{amp_css}">{amp_lbl}</div>', unsafe_allow_html=True)

            st.markdown("#### Cue trend" if lang_code == "en" else "#### Cue trend（前半→後半）")
            trend_rows = [
                ("All turns", cue_trend(all_cues), "#7ecfff"),
                ("User only", cue_trend(user_cues), "#3ddc84"),
                ("AI only", cue_trend(ai_cues), "#a07eff"),
            ]
            for name, tr, color in trend_rows:
                t1, t2, t3, t4, t5 = st.columns([1.4, 1.35, 1.35, 1.35, 1.8])
                t1.markdown(f'<div class="metric-box"><div class="metric-val" style="color:{color};font-size:1em;">{name}</div></div>', unsafe_allow_html=True)
                t2.markdown(f'<div class="metric-box"><div class="metric-val">{tr["early_mean"]:.1f}</div><div class="metric-lbl">Early mean</div></div>', unsafe_allow_html=True)
                t3.markdown(f'<div class="metric-box"><div class="metric-val">{tr["late_mean"]:.1f}</div><div class="metric-lbl">Late mean</div></div>', unsafe_allow_html=True)
                t4.markdown(f'<div class="metric-box"><div class="metric-val">{tr["trend_delta"]:+.1f}</div><div class="metric-lbl">Trend Δ</div></div>', unsafe_allow_html=True)
                t5.markdown(f'<div class="{tr["trend_css"]}">{tr["trend_label"]}</div>', unsafe_allow_html=True)

            # ── Resonance Amplification summary v0.1.1
            ai_tr = cue_trend(ai_cues)
            pressure = resonance_amplification_metrics(user_seed_summary, ai_seed_summary, ai_cue_summary, ai_tr)
            st.markdown("#### Resonance Amplification Layer" if lang_code == "en" else "#### Resonance Amplification Layer（共鳴増幅）")
            p1, p2, p3, p4, p5, p6 = st.columns([1.25, 1.25, 1.25, 1.35, 1.45, 2.0])
            p1.markdown(f'<div class="seed-box"><div class="metric-val">{pressure["user_seed_total"]:.1f}</div><div class="metric-lbl">User seed</div></div>', unsafe_allow_html=True)
            p2.markdown(f'<div class="seed-box"><div class="metric-val">{pressure["ai_seed_total"]:.1f}</div><div class="metric-lbl">AI seed</div></div>', unsafe_allow_html=True)
            p3.markdown(f'<div class="metric-box"><div class="metric-val">{pressure["ai_seed_echo"]:+.1f}</div><div class="metric-lbl">AI seed echo</div></div>', unsafe_allow_html=True)
            p4.markdown(f'<div class="{pressure["echo_css"]}">{pressure["seed_echo_ratio"]:.2f}×<br>{pressure["echo_label"]}</div>', unsafe_allow_html=True)
            p5.markdown(f'<div class="{pressure["conversion_css"]}">{pressure["seed_to_cue_conversion"]:.2f}×<br>{pressure["conversion_label"]}</div>', unsafe_allow_html=True)
            p6.markdown(f'<div class="{pressure["resonance_amplification_css"]}">{pressure["resonance_amplification_label"]}</div>', unsafe_allow_html=True)
            st.caption("Seed → Echo → Cue dynamics: user lexical seed, AI seed echo, seed-to-cue conversion, and AI cue trend. Provisional research annotation, not a safety verdict, diagnosis, or automatic control signal.")

            st.markdown(
                plain_language_observation(
                    lang_code, user_seed_summary, ai_seed_summary,
                    user_cue_summary, ai_cue_summary, amp, pressure, txt
                ),
                unsafe_allow_html=True,
            )

            if pressure["resonance_amplification_label"].startswith(("Strong", "High")) or amp["cue_total_amp"] >= 15.0:
                st.info("Observation note: cue/seed concentration is elevated. For human review, check whether the dialogue remains open, reversible, and connected to external references. Do not treat this as a user diagnosis or model verdict.")

            # ── Plotly chart
            st.markdown("---")
            st.markdown(f"### {'CI 時系列グラフ' if lang_code=='ja' else 'CI Time-Series'}")
            st.plotly_chart(make_plotly_chart(all_res, user_res, ai_res),
                            width="stretch")

            st.markdown(f"### {'Cue Total 時系列グラフ' if lang_code=='ja' else 'Cue Total Time-Series'}")
            st.plotly_chart(make_cue_chart(all_cues, user_cues, ai_cues),
                            width="stretch")

            st.markdown(f"### {'Resonance Seed 時系列グラフ' if lang_code=='ja' else 'Resonance Seed Time-Series'}")
            st.plotly_chart(make_seed_chart(all_seeds, user_seeds, ai_seeds),
                            width="stretch")

            # ── Window detail (expander)
            for label, res, color in [("All turns", all_res, "#7ecfff"),
                                       ("User only", user_res, "#3ddc84"),
                                       ("AI only",   ai_res,  "#a07eff")]:
                if not res:
                    continue
                with st.expander(f"Window detail — {label}"):
                    for r in res:
                        cols = st.columns([1, 1.7, 1.7, 1.6, 1.8, 1.8, 2.2])
                        cols[0].markdown(f'<div class="metric-box"><div class="metric-lbl">W{r["window"]}</div></div>', unsafe_allow_html=True)
                        cols[1].markdown(f'<div class="metric-box"><div class="metric-val" style="color:{color};">{r["eff_rank"]:.3f}</div><div class="metric-lbl">EFF_RANK</div></div>', unsafe_allow_html=True)
                        cols[2].markdown(f'<div class="metric-box"><div class="metric-val">{r["collapse_index"]:+.3f}</div><div class="metric-lbl">COMPRESSION_IDX</div></div>', unsafe_allow_html=True)
                        cols[3].markdown(f'<div class="metric-box"><div class="metric-val">{r.get("cue_total", 0.0):.1f}</div><div class="metric-lbl">CUE_TOTAL</div></div>', unsafe_allow_html=True)
                        cols[4].markdown(f'<div class="metric-box"><div class="metric-val">{r.get("specialness", 0.0):.1f}/{r.get("immersive_deepening", 0.0):.1f}/{r.get("technical_deepening", 0.0):.1f}/{r.get("dependency", 0.0):.1f}</div><div class="metric-lbl">S/I/T/Dp</div></div>', unsafe_allow_html=True)
                        cols[5].markdown(f'<div class="seed-box"><div class="metric-val">{r.get("seed_total", 0.0):.1f}</div><div class="metric-lbl">Seed / {r.get("dominant_seed_type", "none")}</div></div>', unsafe_allow_html=True)
                        cols[6].markdown(f'<div class="{r.get("cue_css", "cue-low")}">{r.get("cue_label", "Low cue")}</div>', unsafe_allow_html=True)
                        if r.get("cue_hits"):
                            st.caption("Cue hits: " + r["cue_hits"])
                        if r.get("seed_hits"):
                            st.caption("Seed hits: " + r["seed_hits"])

            # ── CSV download
            st.markdown("---")
            frames = []
            for label, res in [("all", all_res), ("user", user_res), ("ai", ai_res)]:
                for r in res:
                    frames.append({"track": label,
                                   **{k: r.get(k) for k in [
                                       "window", "eff_rank", "collapse_index", "label",
                                       "specialness", "immersive_deepening", "technical_deepening", "deepening", "dependency", "cue_total",
                                       "cue_label", "context_mode", "specialness_hits", "immersive_deepening_hits", "technical_deepening_hits", "dependency_hits", "cue_hits",
                                       "seed_abstract_conceptual", "seed_emotional_intimate", "seed_existential_self_reference", "seed_total",
                                       "dominant_seed_type", "seed_abstract_conceptual_hits", "seed_emotional_intimate_hits", "seed_existential_self_reference_hits", "seed_hits"
                                   ]},
                                   "ai_cue_total_amp": amp["cue_total_amp"],
                                   "ai_amp_label": amp_lbl,
                                   "all_cue_trend_delta": cue_trend(all_cues)["trend_delta"],
                                   "user_cue_trend_delta": cue_trend(user_cues)["trend_delta"],
                                   "ai_cue_trend_delta": cue_trend(ai_cues)["trend_delta"],
                                   "ai_seed_echo": pressure["ai_seed_echo"],
                                   "seed_echo_ratio": pressure["seed_echo_ratio"],
                                   "seed_echo_label": pressure["echo_label"],
                                   "seed_to_cue_conversion": pressure["seed_to_cue_conversion"],
                                   "seed_to_cue_conversion_label": pressure["conversion_label"],
                                   "resonance_amplification_label": pressure["resonance_amplification_label"],
                                   })
            if frames:
                    df = pd.DataFrame(frames)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"mira_dialogue_metrics_{timestamp}.csv"

                    st.download_button(
                        "⬇ Download CSV" if lang_code == "en" else "⬇ CSVダウンロード",
                        prepare_csv_bytes(df),
                        file_name=filename,
                        mime="text/csv",
                        width="stretch"
                    )
