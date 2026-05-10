
"""
MIRA mini public v0.1.2 — Conversation Air Meter

Cute, simple Streamlit public UI prototype.

- dialogue-only
- no login
- no database
- no conversation-log storage
- non-diagnostic observation
- short "chokotto" measurement for 2–4 exchanges
- conversation CSV + summary CSV
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from mira_engine_en import analyze_dialogue, sample_dialogue


APP_VERSION = "public-v0.1.3-buttonstyle"
ASSET_DIR = Path(__file__).parent / "assets"


st.set_page_config(
    page_title="MIRA mini",
    page_icon="☁️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700;800&display=swap');

:root {
  --cream: #fff8ec;
  --cream2: #fffdf7;
  --pink: #ffb6c8;
  --pink2: #ffe5ed;
  --mint: #dff6ec;
  --blue: #dff1ff;
  --yellow: #fff2b8;
  --brown: #6f5144;
  --brown2: #9b7a6d;
  --card: rgba(255,255,255,0.82);
}

html, body, [class*="css"] {
  font-family: 'Noto Sans JP', sans-serif !important;
}

.stApp {
  background:
    radial-gradient(circle at 14% 7%, #ffeaf2 0 10%, transparent 11%),
    radial-gradient(circle at 91% 9%, #e5f6ff 0 9%, transparent 10%),
    radial-gradient(circle at 18% 89%, #e8f8ee 0 12%, transparent 13%),
    linear-gradient(180deg, #fff8ec 0%, #fffdf7 60%, #fff4e8 100%);
  color: var(--brown);
}

.block-container {
  max-width: 760px;
  padding-top: 1.8rem;
  padding-bottom: 3rem;
}

.mira-title {
  font-size: 2.7rem;
  line-height: 1;
  font-weight: 800;
  letter-spacing: .02em;
  color: var(--brown);
  margin: 1rem 0 0 0;
}

.mira-subtitle {
  font-size: 1.25rem;
  font-weight: 700;
  color: #7e6255;
  margin-top: .75rem;
}

.mira-note {
  display: inline-block;
  background: var(--pink2);
  color: #7d564d;
  border-radius: 999px;
  padding: .55rem .95rem;
  margin-top: .9rem;
  font-weight: 700;
  font-size: .92rem;
}

.mira-card {
  background: var(--card);
  border: 2px solid rgba(255, 214, 226, 0.9);
  border-radius: 28px;
  padding: 20px;
  box-shadow: 0 10px 24px rgba(168, 110, 120, 0.10);
  margin: 18px 0;
}

.mira-section-title {
  font-size: 1.28rem;
  font-weight: 800;
  color: var(--brown);
  margin: 1.1rem 0 .4rem 0;
}

.small-muted {
  color: var(--brown2);
  font-size: .92rem;
  line-height: 1.75;
}

.privacy-box {
  background: #effaf3;
  border: 2px solid #c8edd8;
  color: #50705e;
  border-radius: 22px;
  padding: 14px 16px;
  font-weight: 700;
  line-height: 1.65;
}

.result-card {
  background: #ffffffcc;
  border: 2px solid #ffe0ea;
  border-radius: 24px;
  padding: 16px 18px;
  margin: 10px 0;
}

.result-row {
  display: grid;
  grid-template-columns: 1.2fr 1fr;
  gap: 12px;
  align-items: center;
}

.result-label {
  font-weight: 800;
  color: var(--brown);
}

.result-value {
  text-align: right;
  font-weight: 800;
  color: #8a5d75;
}

.meter {
  margin-top: 8px;
  width: 100%;
  height: 12px;
  background: #f5eadf;
  border-radius: 999px;
  overflow: hidden;
}

.meter > div {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #aee8d0, #ffd67a, #ffadc5);
}

.comment-bubble {
  background: #fff;
  border: 2px solid #ffe0ea;
  border-radius: 24px;
  padding: 18px 20px;
  color: var(--brown);
  font-weight: 700;
  line-height: 1.75;
  box-shadow: 0 10px 22px rgba(168, 110, 120, 0.10);
}

.stTextArea textarea {
  border-radius: 22px !important;
  border: 2px solid #ffdbe6 !important;
  background: rgba(255,255,255,0.92) !important;
  color: #5d463c !important;
  font-size: .95rem !important;
  line-height: 1.7 !important;
}

.stButton button {
  border-radius: 999px !important;
  border: none !important;
  background: linear-gradient(90deg, #ff9fba, #ffc0cf) !important;
  color: white !important;
  font-weight: 800 !important;
  font-size: 1.06rem !important;
  padding: .78rem 1.4rem !important;
  box-shadow: 0 8px 18px rgba(234, 120, 150, .25) !important;
}

div[data-testid="stMetric"] {
  background: #fffdf8 !important;
  border: 2px solid #ffd9e6 !important;
  border-radius: 22px !important;
  padding: 14px !important;
  box-shadow: 0 8px 18px rgba(168, 110, 120, 0.08);
}

div[data-testid="stMetric"] label,
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
  color: #8a6a5d !important;
  opacity: 1 !important;
  font-weight: 700 !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div,
div[data-testid="stMetricValue"] p {
  color: #6f5144 !important;
  opacity: 1 !important;
  font-weight: 800 !important;
}

div[data-testid="stExpander"] {
  background: rgba(255, 255, 255, 0.78) !important;
  border: 2px solid #ffd9e6 !important;
  border-radius: 18px !important;
}

div[data-testid="stExpander"] details summary,
div[data-testid="stExpander"] details summary p {
  color: #6f5144 !important;
  font-weight: 800 !important;
}

div[data-testid="stDataFrame"] {
  background: #fffdf8 !important;
  border-radius: 16px !important;
}

hr {
  border-color: #ffe0ea;
}

/* Download buttons: make CSV buttons readable and hover-friendly */
div[data-testid="stDownloadButton"] button {
  background: #fffdf8 !important;
  color: #6f5144 !important;
  border: 2px solid #ffd9e6 !important;
  border-radius: 999px !important;
  font-weight: 800 !important;
  box-shadow: 0 8px 18px rgba(168, 110, 120, 0.10) !important;
  transition: all 0.18s ease-in-out !important;
}

div[data-testid="stDownloadButton"] button p,
div[data-testid="stDownloadButton"] button span {
  color: #6f5144 !important;
  font-weight: 800 !important;
}

div[data-testid="stDownloadButton"] button:hover {
  background: linear-gradient(90deg, #ff9fba, #ffc0cf) !important;
  color: #ffffff !important;
  border-color: #ff9fba !important;
  transform: translateY(-1px);
  box-shadow: 0 10px 22px rgba(234, 120, 150, 0.24) !important;
}

div[data-testid="stDownloadButton"] button:hover p,
div[data-testid="stDownloadButton"] button:hover span {
  color: #ffffff !important;
}

/* Primary/normal button hover */
.stButton button {
  transition: all 0.18s ease-in-out !important;
}

.stButton button:hover {
  background: linear-gradient(90deg, #ff87aa, #ffb0c5) !important;
  color: #ffffff !important;
  transform: translateY(-1px);
  box-shadow: 0 10px 22px rgba(234, 120, 150, 0.28) !important;
}

</style>
""",
    unsafe_allow_html=True,
)


def show_mascot(width: int = 750) -> None:
    candidates = [
        ASSET_DIR / "mira_main.png",
        ASSET_DIR / "mira_smile.png",
        ASSET_DIR / "mira_observe.png",
    ]
    for p in candidates:
        if p.exists():
            st.image(str(p), width=width)
            return
    st.markdown("<div style='font-size:5rem;text-align:center;'>🐻‍❄️🔭</div>", unsafe_allow_html=True)


def meter(label: str, value_label: str, score_0_100: float, icon: str) -> None:
    score = max(0, min(100, float(score_0_100)))
    st.markdown(
        f"""
<div class="result-card">
  <div class="result-row">
    <div class="result-label">{icon} {html.escape(label)}</div>
    <div class="result-value">{html.escape(value_label)}</div>
  </div>
  <div class="meter"><div style="width:{score:.0f}%"></div></div>
</div>
""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────
show_mascot(width=750)
st.markdown('<h1 class="mira-title">MIRA mini</h1>', unsafe_allow_html=True)
st.markdown('<div class="mira-subtitle">A gentle meter for AI conversation dynamics</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="mira-note">Paste a short AI conversation.<br>MIRA mini gently visualizes openness, loopiness, AI amplification, and return paths.<br>This is not a diagnosis, score, warning, or safety judgment.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="mira-card">
  <div class="mira-section-title">☁️  What is MIRA mini?</div>
  <div class="small-muted">
    MIRA mini is not a diagnosis, score, or warning.<br>
    It is a lightweight observation tool for the dynamics and openness of AI conversations.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="privacy-box">
  🔒 Just 2–4 recent exchanges · No login · No storage<br>Your conversation data is not stored.
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# Input
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="mira-section-title">📝 Paste your conversation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="small-muted">Format: Add "User:" and "AI:" labels. Even if the AI response is long, you can paste it as is.<br><br></div>',
    unsafe_allow_html=True,
)

col_a, col_b = st.columns([1, 1])
with col_a:
    use_sample = st.button("Use sample")
with col_b:
    clear = st.button("Clear")

if "dialogue_text" not in st.session_state:
    st.session_state.dialogue_text = ""

if use_sample:
    st.session_state.dialogue_text = sample_dialogue()
if clear:
    st.session_state.dialogue_text = ""
    st.session_state.last_result = None

dialogue_text = st.text_area(
    "Conversation with AI",
    key="dialogue_text",
    height=230,
    placeholder=sample_dialogue(),
    label_visibility="collapsed",
)
analyze = st.button("☁️ Check conversation tone", type="primary", use_container_width=True)




# ─────────────────────────────────────────────────────────────
# Analyze
# ─────────────────────────────────────────────────────────────
if "last_result" not in st.session_state:
    st.session_state.last_result = None

if analyze:
    with st.spinner("MIRA is observing.…"):
        result = analyze_dialogue(dialogue_text, lang="ja", window=2)

    if result["error"]:
        st.session_state.last_result = None
        st.warning(result["message"])
    else:
        st.session_state.last_result = result

result = st.session_state.last_result

if result is not None:
    st.markdown("---")
    st.markdown("## 🌤️ Results")

    st.markdown(
        f"""
<div class="comment-bubble">
 ☁☁ <b>From MIRA</b><br>
  {html.escape(result["friendly_comment"])}
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### Today’s conversation air")

    meter("Openness", result["labels"]["openness"], result["scores"]["openness"], "🌿")
    meter("Loopiness", result["labels"]["loopiness"], result["scores"]["loopiness"], "🌀")
    meter("AI amplification", result["labels"]["ai_amplification"], result["scores"]["ai_amplification"], "📣")
    meter("Return path", result["labels"]["return_path"], result["scores"]["return_path"], "🔁")
    meter("Conversation air", result["labels"]["air"], result["scores"]["air"], "☁️")

    with st.expander("View detailed metrics (for research)", expanded=True):
        st.markdown(
            """
<div class="small-muted">
  The values ​below are for research and verification purposes.<br>
  For everyday use, you only need to look at the "Atmosphere of Today's Conversation" above.<br>
  The detailed values ​​are reference values ​for research purposes. <br>
  They are not for diagnosis, judgment, or safety assessment.<br>
  You can download the information you viewed on the screen as a CSV file.<br><br>
</div>
""",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("turns", result["turn_count"])
        c2.metric("effective_rank", f'{result["raw"]["effective_rank"]:.3f}')
        c3.metric("compression_index", f'{result["raw"]["compression_index"]:+.3f}')

        c4, c5, c6 = st.columns(3)
        c4.metric("user_seed_total", f'{result["raw"]["user_seed_total"]:.1f}')
        c5.metric("ai_seed_total", f'{result["raw"]["ai_seed_total"]:.1f}')
        c6.metric("ai_cue_total", f'{result["raw"]["ai_cue_total"]:.1f}')

        c7, c8, c9 = st.columns(3)
        c7.metric("seed_echo_ratio", f'{result["raw"]["seed_echo_ratio"]:.2f}×')
        c8.metric("seed_to_cue_conversion", f'{result["raw"]["seed_to_cue_conversion"]:.2f}×')
        c9.metric("cue_total_amp", f'{result["raw"]["cue_total_amp"]:+.1f}')

        st.caption("The values are for research purposes only and are not for diagnosis, assessment, or safety evaluation.")

        df_turns = pd.DataFrame(result["turns"])
        st.markdown("#### Conversation log")
        st.dataframe(df_turns, use_container_width=True)

        st.download_button(
            "Download conversation CSV",
            df_turns.to_csv(index=False).encode("utf-8-sig"),
            "mira_mini_conversation.csv",
            "text/csv",
            use_container_width=True,
            on_click="ignore",
        )

        summary_row = {
            "turn_count": result["turn_count"],
            "openness": result["labels"]["openness"],
            "loopiness": result["labels"]["loopiness"],
            "ai_amplification": result["labels"]["ai_amplification"],
            "return_path": result["labels"]["return_path"],
            "air": result["labels"]["air"],
            "friendly_comment": result["friendly_comment"],
            "overall_note": "This result provides a reference view of the 'conversation atmosphere' based on the short conversation log pasted here.",
            "metric_note": "The metrics are observational values derived from the vector structure of the conversation text and lexical cues. They are not intended as a diagnosis, safety determination, or evaluation of the AI.",
            "privacy_note": "This app does not store conversation logs. The CSV includes only the information displayed on the screen for the user's own review and download.",
            "disclaimer": "Results may be less stable for short conversations. If needed, you may check multiple times or compare across different conversations.",
            **result["raw"],
        }
        df_summary = pd.DataFrame([summary_row])

        st.markdown("#### Summary")
        st.dataframe(df_summary, use_container_width=True)

        st.download_button(
            "Download the measurement results CSV",
            df_summary.to_csv(index=False).encode("utf-8-sig"),
            "mira_mini_summary.csv",
            "text/csv",
            use_container_width=True,
            on_click="ignore",
        )

    st.markdown(
        """
<div class="mira-card">
  <div class="small-muted">
    🌱 <b>Usage tips</b><br>
    This result is an observation of the short conversation just posted. It does not represent the entire longer conversation.<br>
    If something feels uncomfortable, you can change the topic, take a break, or jot down your own thoughts.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# Footer mascots
# ─────────────────────────────────────────────────────────────


footer_img = ASSET_DIR / "mira_main.png"
if footer_img.exists():
    st.image(str(footer_img), use_container_width=True)

st.markdown(
    '<div class="small-muted" style="text-align:center; margin-top:1.5rem;">....</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="
    text-align:center;
    color:#9b7a6d;
    font-size:0.82rem;
    margin-top:1.5rem;
    margin-bottom:1rem;
">
  © 2026 MIRA mini / Jun Sakai. All rights reserved.
</div>
""",
    unsafe_allow_html=True,
)