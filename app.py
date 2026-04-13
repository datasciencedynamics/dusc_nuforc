#!/usr/bin/env python3
"""
app.py
======
Streamlit LIME explainer app for the NUFORC UAP dramatic sighting classifier.

Usage
-----
    streamlit run app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from lime.lime_text import LimeTextExplainer

sys.path.append(str(Path(__file__).resolve().parent))

from core.functions import mlflow_load_model

st.set_page_config(
    page_title="UAP Sighting Classifier",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

html, body {
    background: #fafaf8 !important;
    color: #111 !important;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 19px;
    color: #0a0a0a !important;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
[data-testid="stMain"],
[data-testid="block-container"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    background: #fafaf8 !important;
}

[data-testid="stSidebar"] { background: #f2f2ef !important; }

/* ── Typography ── */
.app-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #333;
    margin-bottom: 0.6rem;
}

.app-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    color: #0d0d0d;
    line-height: 1.05;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
}

.app-tagline {
    font-size: 1.25rem;
    color: #1a1a1a;
    font-weight: 400;
    line-height: 1.6;
    max-width: 640px;
    margin-bottom: 0;
}

/* ── Divider ── */
.ruled {
    border: none;
    border-top: 1.5px solid #e4e4e0;
    margin: 2rem 0;
}

/* ── Section labels ── */
.field-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #111;
    margin-bottom: 0.5rem;
}

/* ── Inputs ── */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] > div > div,
[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input {
    background: #ffffff !important;
    color: #111 !important;
    border: 1.5px solid #e0e0da !important;
    border-radius: 10px !important;
    font-size: 1.05rem !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #111 !important;
    box-shadow: none !important;
}

[data-baseweb="popover"] > div,
[data-baseweb="menu"] {
    background: #ffffff !important;
    border: 1.5px solid #e0e0da !important;
    border-radius: 10px !important;
}

[data-baseweb="option"]:hover { background: #f5f5f0 !important; }

label, .stSelectbox label, .stTextArea label,
.stTextInput label, .stSlider label, .stNumberInput label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: #111 !important;
}

/* ── Score card ── */
.score-block {
    background: #f8f9fd;
    border: 2px solid #d0d8ee;
    border-radius: 16px;
    padding: 2.2rem 2rem;
    text-align: center;
}

.score-num {
    font-family: 'Syne', sans-serif;
    font-size: 4.5rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: 0;
    margin: 0;
    animation: score-reveal 0.9s cubic-bezier(0.22, 1, 0.36, 1) both;
}

.score-tier {
    animation: score-reveal 0.9s 0.1s cubic-bezier(0.22, 1, 0.36, 1) both;
}

.score-sub {
    animation: score-reveal 0.9s 0.2s cubic-bezier(0.22, 1, 0.36, 1) both;
}

@keyframes score-reveal {
    from { opacity: 0; transform: translateY(14px) scale(0.96); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

.score-tier {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 0.4rem;
    letter-spacing: -0.01em;
    animation: score-reveal 0.9s 0.1s cubic-bezier(0.22, 1, 0.36, 1) both;
}

.score-sub {
    font-size: 1rem;
    color: #444;
    margin-top: 0.4rem;
    font-weight: 500;
    animation: score-reveal 0.9s 0.2s cubic-bezier(0.22, 1, 0.36, 1) both;
}

/* Gauge */
.gauge-wrap {
    margin-top: 1.6rem;
    margin-bottom: 0.3rem;
}

.gauge-track {
    height: 8px;
    border-radius: 99px;
    background: linear-gradient(to right, #b8d4f8, #fde68a, #fca5a5, #ef4444);
    position: relative;
    width: 100%;
}

.gauge-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.88rem;
    color: #444;
    font-weight: 600;
    margin-top: 0.5rem;
    font-family: 'Inter', sans-serif;
}

/* ── Interpretation box ── */
.interp {
    background: #ffffff;
    border: 1.5px solid #e4e4e0;
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    font-size: 1.05rem;
    line-height: 1.8;
    color: #0d0d0d;
    margin-top: 1.5rem;
}

.interp strong { color: #0d0d0d; }

/* ── Notice banner ── */
.notice {
    background: #fffdf0;
    border-left: 3px solid #f0b429;
    padding: 1.1rem 1.6rem;
    border-radius: 0 10px 10px 0;
    font-size: 1.05rem;
    color: #2a1a00;
    line-height: 1.75;
    margin-bottom: 2.5rem;
}

/* ── Button ── */
.stButton > button {
    background: #0d0d0d !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    padding: 0.8rem 2rem !important;
    width: 100% !important;
    transition: background 0.15s !important;
    margin-top: 1rem !important;
}

.stButton > button:hover {
    background: #2a2a2a !important;
}

/* ── LIME section ── */
.lime-intro {
    font-size: 1.05rem;
    color: #0d0d0d;
    line-height: 1.75;
    margin-bottom: 1.2rem;
}

[data-testid="stDataFrame"] { background: #ffffff !important; }
[data-testid="stAlert"] {
    background: #ffffff !important;
    border: 1.5px solid #e0e0da !important;
    color: #333 !important;
    font-size: 1rem !important;
}

[data-testid="stExpander"] summary {
    font-size: 0.95rem !important;
    color: #888 !important;
}

/* placeholder empty state */
.empty-state {
    padding: 3rem 0;
    text-align: center;
    color: #bbb;
}

.empty-state .icon { font-size: 3.5rem; margin-bottom: 0.8rem; }
.empty-state p { font-family: 'Syne', sans-serif; font-size: 1.1rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PRED_DIR = Path(__file__).resolve().parent / "models" / "predictions"
ENRICHED_PATH = (
    Path(__file__).resolve().parent / "data" / "processed" / "NUFORC_enriched.parquet"
)
TEXT_COL = "summary_clean"
OUTCOME = "dramatic"
MODEL_KEY = "CatBoost Feats + Text"

SHAPES = [
    "Orb",
    "Light",
    "Fireball",
    "Flash",
    "Circle",
    "Sphere",
    "Oval",
    "Disk",
    "Saucer",
    "Triangle",
    "Rectangle",
    "Diamond",
    "Chevron",
    "Boomerang",
    "Cigar",
    "Cylinder",
    "Torpedo",
    "Cone",
    "Formation",
    "Changing",
    "Unknown",
    "Other",
]

SHAPE_MAP = {
    "Orb": "luminous",
    "Light": "luminous",
    "Fireball": "luminous",
    "Flash": "luminous",
    "Circle": "circular",
    "Sphere": "circular",
    "Oval": "circular",
    "Disk": "classic_ufo",
    "Saucer": "classic_ufo",
    "Triangle": "angular",
    "Rectangle": "angular",
    "Diamond": "angular",
    "Chevron": "angular",
    "Boomerang": "angular",
    "Cigar": "elongated",
    "Cylinder": "elongated",
    "Torpedo": "elongated",
    "Cone": "elongated",
    "Formation": "multi_object",
    "Changing": "other",
    "Unknown": "other",
    "Other": "other",
}

UAP_EVENTS = [
    pd.Timestamp("2021-06-25"),
    pd.Timestamp("2022-05-17"),
    pd.Timestamp("2022-07-15"),
    pd.Timestamp("2023-01-12"),
    pd.Timestamp("2023-04-19"),
    pd.Timestamp("2023-07-26"),
    pd.Timestamp("2023-08-30"),
    pd.Timestamp("2023-09-14"),
    pd.Timestamp("2024-03-06"),
    pd.Timestamp("2024-11-13"),
    pd.Timestamp("2024-11-14"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return mlflow_load_model(
        experiment_name=f"{OUTCOME}_text_model",
        run_name="cat_feats_and_text_orig_training",
        model_name=f"cat_feats_and_text_{OUTCOME}",
    )


@st.cache_data(show_spinner="Loading reference data...")
def load_reference_data():
    X_test = pd.read_parquet(PRED_DIR / "X_test_cat_feats_and_text.parquet")
    enriched = pd.read_parquet(ENRICHED_PATH)
    y_prob = pd.read_parquet(PRED_DIR / "y_prob_cat_feats_and_text.parquet").squeeze()
    return X_test, enriched, y_prob


@st.cache_data
def load_threshold():
    with open(PRED_DIR / "model_thresholds.json") as f:
        return json.load(f)[MODEL_KEY]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def build_feature_row(
    summary_text, shape, state, country, hour, X_test_template, enriched_df
):
    import re

    STOPWORDS = {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "see",
        "saw",
        "seen",
        "look",
        "looked",
        "looking",
        "thing",
        "things",
        "something",
        "went",
        "go",
        "gone",
        "got",
        "come",
        "came",
    }
    TOKEN_RE = re.compile(r"[a-z0-9']+")
    tokens = [
        t.strip("'")
        for t in TOKEN_RE.findall(summary_text.lower())
        if len(t.strip("'")) >= 2
        and not t.strip("'").isdigit()
        and t.strip("'") not in STOPWORDS
    ]
    summary_clean = " ".join(tokens)

    ref = X_test_template.median(numeric_only=True).to_dict()
    for col in X_test_template.select_dtypes(include="object").columns:
        ref[col] = (
            X_test_template[col].mode().iloc[0]
            if not X_test_template[col].mode().empty
            else ""
        )

    loc_count = int(
        enriched_df[
            enriched_df["State"].str.upper().eq(str(state).upper())
            & enriched_df["Country"].eq(country)
        ].shape[0]
    )

    today = pd.Timestamp.today()
    prior = [e for e in UAP_EVENTS if e.date() <= today.date()]
    days_since = int((today.date() - max(prior).date()).days) if prior else 0

    ref.update(
        {
            TEXT_COL: summary_clean,
            "occurred_year": int(today.year),
            "occurred_month": int(today.month),
            "occurred_day": int(today.day),
            "occurred_hour": int(hour),
            "report_lag_days": 1,
            "is_night": int(hour >= 20 or hour <= 5),
            "is_weekend": int(today.weekday() in (5, 6)),
            "shape_group": SHAPE_MAP.get(shape, "other"),
            "summary_token_count": len(tokens),
            "has_media": 0,
            "location_count_total": loc_count,
            "cluster_id": np.nan,
            "in_cluster": 0,
            "days_since_uap_event": days_since,
            "exp_drone": 0,
            "exp_rocket": 0,
            "exp_balloon": 0,
            "exp_aircraft": 0,
            "exp_starlink": 0,
            "exp_lantern": 0,
            "exp_satellite": 0,
            "exp_certain": 0,
        }
    )
    return pd.DataFrame([ref])[X_test_template.columns]


def make_predict_fn(model, row_dict, col_order):
    def predict_fn(texts):
        rows = [{**row_dict, TEXT_COL: t} for t in texts]
        return model.predict_proba(pd.DataFrame(rows)[col_order])

    return predict_fn


def get_percentile(prob, ref_probs):
    return int(np.mean(ref_probs <= prob) * 100)


def score_tier(pct):
    if pct >= 90:
        return "Extremely Unusual", "#dc2626"
    if pct >= 75:
        return "Very Unusual", "#ea580c"
    if pct >= 50:
        return "Somewhat Unusual", "#ca8a04"
    if pct >= 25:
        return "Fairly Typical", "#2563eb"
    return "Very Typical", "#1d4ed8"


# ─────────────────────────────────────────────────────────────────────────────
# Page layout
# ─────────────────────────────────────────────────────────────────────────────

# ── Title ──
st.markdown(
    '<p class="app-eyebrow">NUFORC · CatBoost Feats + Text · Posard et al. (2023)</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<h1 class="app-title">UAP Sighting Classifier</h1>', unsafe_allow_html=True
)
st.markdown(
    '<p class="app-tagline">Describe your sighting and see how dramatically it reads '
    "compared to 20,000+ real NUFORC reports.</p>",
    unsafe_allow_html=True,
)

st.markdown('<hr class="ruled">', unsafe_allow_html=True)

# ── Notice ──
st.markdown(
    '<div class="notice"><strong>What does "dramatic" mean?</strong> '
    "It refers to <em>language patterns</em> — not whether your experience was real or significant. "
    "NUFORC editors flag reports with unusual vocabulary, vivid technical phrasing, or descriptions "
    "that deviate from typical misidentification reports. A plain account of an extraordinary event "
    "can still score low.</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Two-column layout
# ─────────────────────────────────────────────────────────────────────────────

left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown(
        '<p class="field-label">Sighting Description</p>', unsafe_allow_html=True
    )
    summary_text = st.text_area(
        "desc",
        placeholder="A triangular craft hovered silently above the treeline, three pulsing lights at each corner, then accelerated instantly out of sight...",
        height=200,
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="field-label">Details</p>', unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        shape = st.selectbox("Shape", SHAPES, index=0)
        country = st.text_input("Country", value="USA")
    with d2:
        state = st.text_input("State / Province", value="CA")
        hour = st.slider("Hour of sighting (24h)", 0, 23, 21)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("⚙️ LIME Settings"):
        n_perturb = st.number_input("Permutations", value=300, step=100, min_value=100)
        n_words = st.slider("Words to highlight", 5, 30, 12)

    predict_clicked = st.button("Classify Sighting →")

# ── Right: score ──
with right:
    if predict_clicked and summary_text.strip():
        X_test_template, enriched_df, ref_probs = load_reference_data()
        threshold = load_threshold()
        model = load_model()

        feature_row = build_feature_row(
            summary_text,
            shape,
            state,
            country,
            hour,
            X_test_template,
            enriched_df,
        )

        prob_drama = float(model.predict_proba(feature_row)[0][1])
        pct = get_percentile(prob_drama, ref_probs.values)
        tier, color = score_tier(pct)
        marker_left = f"{prob_drama * 100:.1f}%"

        if pct >= 75:
            interp = (
                f"Your report is in the <strong>top {100-pct}%</strong> of NUFORC sightings — "
                f"it uses specific vocabulary, vivid phrasing, or structural characteristics "
                f"that NUFORC editors associate with anomalous reports."
            )
        elif pct >= 50:
            interp = (
                f"Your report scores above the median, higher than <strong>{pct}%</strong> "
                f"of NUFORC sightings. Some unusual language characteristics are present."
            )
        else:
            interp = (
                f"Your report reads like the majority of routine NUFORC sightings "
                f"(<strong>{pct}th percentile</strong>). Try adding specific detail about "
                f"the object's movement, appearance, speed, or behavior."
            )

        st.components.v1.html(
            f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@800&family=Inter:wght@500;600&display=swap');
@keyframes fadein {{
    from {{ opacity:0; transform:translateY(10px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}
* {{ margin:0;padding:0;box-sizing:border-box; }}
body {{ background:#fafaf8;font-family:'Inter',sans-serif; }}
.wrap    {{ background:#f8f9fd;border:2px solid #d0d8ee;border-radius:16px;padding:2rem 1.8rem;text-align:center; }}
.num     {{ font-family:'Syne',sans-serif;font-size:4rem;font-weight:800;color:{color};line-height:1;margin-top:.3rem; }}
.tier    {{ font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:700;color:{color};line-height:1.1;animation:fadein .4s both; }}
.sub     {{ font-size:1rem;color:#555;font-weight:500;margin-top:.2rem;animation:fadein .5s .9s both; }}
.gauge   {{ height:8px;border-radius:99px;background:linear-gradient(to right,#b8d4f8,#fde68a,#fca5a5,#ef4444);position:relative;margin:1.4rem 0 .4rem; }}
.marker  {{ position:absolute;top:-4px;width:10px;height:16px;background:#0d0d0d;border-radius:3px;border:2px solid #fafaf8;transition:left 1s ease; }}
.glabs   {{ display:flex;justify-content:space-between;font-size:.85rem;color:#555;font-weight:600; }}
.divider {{ border:none;border-top:1.5px solid #e4e4e0;margin:1.3rem 0; }}
.interp  {{ font-size:1rem;line-height:1.75;color:#0d0d0d;text-align:left;animation:fadein .5s 1s both; }}
.meta    {{ font-size:.82rem;color:#666;margin-top:.8rem;text-align:left;animation:fadein .5s 1.1s both; }}
.meta strong {{ color:#0d0d0d; }}
</style>
<div class="wrap">
  <div class="tier">{tier}</div>
  <div class="num" id="counter">0%</div>
  <div class="sub">of NUFORC reports</div>
  <div class="gauge">
    <div class="marker" id="marker" style="left:0px;"></div>
  </div>
  <div class="glabs"><span>Typical</span><span>Unusual</span></div>
  <hr class="divider">
  <div class="interp">{interp}</div>
  <div class="meta">Score: <strong>{prob_drama:.4f}</strong> &nbsp;·&nbsp; {pct}th percentile &nbsp;·&nbsp; Threshold: <strong>{threshold:.3f}</strong></div>
</div>
<script>
(function() {{
  var target = {prob_drama};
  var duration = 900;
  var start = null;
  var el = document.getElementById('counter');
  var marker = document.getElementById('marker');
  var gaugeWidth = marker.parentElement.offsetWidth;

  function easeOut(t) {{ return 1 - Math.pow(1 - t, 3); }}

  function step(ts) {{
    if (!start) start = ts;
    var progress = Math.min((ts - start) / duration, 1);
    var eased = easeOut(progress);
    var current = Math.round(eased * {pct});
    el.textContent = current + '%';
    marker.style.left = (eased * target * 100 * (gaugeWidth / gaugeWidth) - 5) + 'px';
    if (progress < 1) requestAnimationFrame(step);
    else {{
      el.textContent = '{pct}%';
      marker.style.left = 'calc({marker_left} - 5px)';
    }}
  }}
  requestAnimationFrame(step);
}})();
</script>
""",
            height=420,
        )

    elif predict_clicked and not summary_text.strip():
        st.warning("Please enter a sighting description first.")

    else:
        st.markdown(
            '<div class="empty-state">'
            '<div class="icon">🛸</div>'
            "<p>Your score will<br>appear here</p>"
            "</div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# LIME
# ─────────────────────────────────────────────────────────────────────────────

if predict_clicked and summary_text.strip():
    st.markdown('<hr class="ruled">', unsafe_allow_html=True)
    st.markdown(
        '<p class="field-label">LIME Text Explanation</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="lime-intro">Words in '
        '<span style="color:#dc2626;font-weight:600;">red/orange</span> pushed the score up. '
        'Words in <span style="color:#2563eb;font-weight:600;">blue</span> pushed it down.</p>',
        unsafe_allow_html=True,
    )

    row_dict = feature_row.iloc[0].to_dict()
    col_order = list(feature_row.columns)
    predict_fn = make_predict_fn(model, row_dict, col_order)
    explainer = LimeTextExplainer(class_names=["Not Dramatic", "Dramatic"])

    with st.spinner("Running LIME..."):
        exp = explainer.explain_instance(
            str(row_dict.get(TEXT_COL, "")),
            predict_fn,
            num_features=int(n_words),
            num_samples=int(n_perturb),
        )

    st.components.v1.html(exp.as_html(), height=220, scrolling=False)

    st.markdown(
        '<p class="field-label" style="margin-top:2rem;">Word Contributions</p>',
        unsafe_allow_html=True,
    )
    lime_df = pd.DataFrame(exp.as_list(), columns=["Word", "LIME Weight"]).sort_values(
        "LIME Weight", key=abs, ascending=False
    )
    lime_df["Direction"] = lime_df["LIME Weight"].apply(
        lambda x: "🔴 More Dramatic" if x > 0 else "🔵 Less Dramatic"
    )
    st.dataframe(
        lime_df.style.format({"LIME Weight": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
