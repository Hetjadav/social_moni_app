"""
Machine Learning & Data Analytics Dashboard — Streamlit version.

Mirrors the React dashboard (Overview / Data Explorer / Predictor) and loads
the trained RandomForest model from `rf_model.pkl` when available. If the
pickle is missing, a light heuristic fallback keeps the UI functional.

Run:
    pip install streamlit pandas numpy plotly scikit-learn joblib
    streamlit run streamlit_app.py

Expected files next to this script:
    - dataset.csv       (your full dataset — same columns as the sample)
    - rf_model.pkl      (optional, trained sklearn pipeline / classifier)
"""

from __future__ import annotations

import io
import time
from pathlib import Path
import base64

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------- #
# Page config & theme
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Digital Wellbeing • ML Dashboard",
    page_icon="wellbeing.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
  .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}
  h1, h2, h3 {letter-spacing: -0.02em;}
  .metric-card {
      background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(14,165,233,0.08));
      border: 1px solid rgba(148,163,184,0.15);
      border-radius: 14px; padding: 18px 20px;
  }
  .result-card {
      border-radius: 18px; padding: 24px 28px;
      background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(59,130,246,0.12));
      border: 1px solid rgba(148,163,184,0.2);
  }
  .pill {display:inline-block;padding:4px 12px;border-radius:999px;
         font-size:12px;font-weight:600;letter-spacing:.03em;}
  .pill-good {background:rgba(16,185,129,.15);color:#10b981;}
  .pill-mod  {background:rgba(234,179,8,.15);color:#eab308;}
  .pill-risk {background:rgba(239,68,68,.15);color:#ef4444;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Constants / schema
# --------------------------------------------------------------------------- #
CATEGORIES = {
    "gender": ["Female", "Male", "Non-binary", "Prefer not to say", "Unknown"],
    "occupation": ["Full-time employed", "Part-time employed", "Retired",
                   "Self-employed", "Student", "Unemployed"],
    "region": ["Africa", "Asia", "Europe", "Latin America", "North America", "Oceania"],
    "most_used_platform": ["Facebook", "Instagram", "LinkedIn", "Reddit",
                            "Snapchat", "TikTok", "X/Twitter", "YouTube"],
    "night_time_use": ["Every night", "Never", "Often", "Sometimes"],
    "primary_purpose": ["Connection with friends", "Content creation", "Entertainment",
                         "News/information", "Passing time/boredom", "Work/career"],
    "uses_screen_time_limits": ["No", "Yes"],
    "attempted_digital_detox": ["No", "Yes, failed", "Yes, succeeded"],
    "seeks_mental_health_support": ["Considering it", "No", "Yes"],
}
WELLBEING_CLASSES = ["Good", "Moderate", "At-risk"]
BAND_COLORS = {"Good": "#10b981", "Moderate": "#eab308", "At-risk": "#ef4444"}

FEATURE_ORDER = [
    "age", "gender", "occupation", "region", "most_used_platform",
    "platforms_used_count", "daily_screen_hours", "daily_notifications",
    "night_time_use", "minutes_to_first_check_after_waking", "primary_purpose",
    "avg_sleep_hours", "anxiety_score_0to27", "low_mood_score_0to27",
    "life_satisfaction_1to10", "loneliness_1to10", "self_esteem_1to10",
    "fomo_1to10", "social_comparison_1to10", "physical_activity_days_per_week",
    "uses_screen_time_limits", "attempted_digital_detox", "seeks_mental_health_support",
]

# --------------------------------------------------------------------------- #
# Data & model loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_dataset(path: str = "social_media_screentime_mental_health_2026.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        st.warning(f"`{path}` not found — using an empty frame. "
                   "Place your CSV next to this script.")
        return pd.DataFrame(columns=FEATURE_ORDER + ["wellbeing_band", "participant_id"])
    return pd.read_csv(p)


@st.cache_resource(show_spinner=False)
def load_model(path: str = "rf_model.pkl"):
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        return None


def heuristic_predict(row: dict) -> tuple[str, dict]:
    risk = (
        row["anxiety_score_0to27"] * 1.2
        + row["low_mood_score_0to27"] * 1.1
        + row["fomo_1to10"] * 0.9
        + row["social_comparison_1to10"] * 0.8
        + row["loneliness_1to10"] * 0.9
        + max(0, row["daily_screen_hours"] - 3) * 2.5
        + max(0, 7 - row["avg_sleep_hours"]) * 2
        - row["life_satisfaction_1to10"] * 1.2
        - row["self_esteem_1to10"] * 1.1
        - row["physical_activity_days_per_week"] * 0.8
    )
    good = np.exp(-max(0, risk) / 8)
    bad = np.exp(max(0, risk - 15) / 8)
    mod = 1.0
    s = good + mod + bad
    probs = {"Good": good / s, "Moderate": mod / s, "At-risk": bad / s}
    pred = max(probs, key=probs.get)
    return pred, probs


def predict(model, payload: dict) -> tuple[str, dict, float]:
    t0 = time.time()
    if model is None:
        pred, probs = heuristic_predict(payload)
    else:
        X = pd.DataFrame([payload])[FEATURE_ORDER]
        try:
            proba = model.predict_proba(X)[0]
            classes = list(model.classes_)
            probs = {c: float(p) for c, p in zip(classes, proba)}
            for c in WELLBEING_CLASSES:
                probs.setdefault(c, 0.0)
            pred = max(probs, key=probs.get)
        except Exception as e:
            st.error(f"Model inference failed, falling back to heuristic: {e}")
            pred, probs = heuristic_predict(payload)
    return pred, probs, (time.time() - t0) * 1000


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
df = load_dataset()
model = load_model()

BASE_DIR = Path(__file__).parent
icon_path = BASE_DIR / "wellbeing.png"

with open(icon_path, "rb") as f:
    data = base64.b64encode(f.read()).decode()

with st.sidebar:
    st.markdown(
        f"""
        <div style="display:flex; align-items:center;">
            <img src="data:image/png;base64,{data}" width="50" style="margin-right:8px;">
            <h2 style="margin:0;">Wellbeing ML</h2>
        </div>
        """,
        unsafe_allow_html=True
    )



    st.caption("RandomForest • rf_model.pkl")
    st.markdown("---")
    st.metric("Rows loaded", f"{len(df):,}")
    st.metric("Features", len(FEATURE_ORDER))
    st.metric("Model", "Loaded ✅" if model is not None else "Heuristic fallback")
    st.markdown("---")
    st.caption("Reported accuracy: **0.87**")

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Digital Wellbeing Analytics")
st.caption("Explore social-media behaviour, mental-health signals and predict wellbeing bands.")

tab_overview, tab_explorer, tab_predict = st.tabs(
    ["📊 Overview", "🔎 Data Explorer", "🤖 Predictor"]
)

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tab_overview:
    if df.empty:
        st.info("Load a dataset to see analytics.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Participants", f"{len(df):,}")
        c2.metric("Avg screen hours", f"{df['daily_screen_hours'].mean():.1f} h")
        c3.metric("Avg sleep hours", f"{df['avg_sleep_hours'].mean():.1f} h")
        at_risk = (df["wellbeing_band"] == "At-risk").mean() * 100
        c4.metric("At-risk share", f"{at_risk:.1f} %")

        st.markdown("### Wellbeing distribution")
        left, right = st.columns([1, 1])
        with left:
            counts = df["wellbeing_band"].value_counts().reindex(WELLBEING_CLASSES).fillna(0)
            fig = px.pie(values=counts.values, names=counts.index, hole=0.55,
                         color=counts.index, color_discrete_map=BAND_COLORS)
            fig.update_layout(height=340, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.histogram(df, x="daily_screen_hours", color="wellbeing_band",
                               nbins=20, color_discrete_map=BAND_COLORS,
                               title="Daily screen hours by wellbeing")
            fig.update_layout(height=340, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Mental-health signals")
        l, r = st.columns(2)
        with l:
            fig = px.box(df, x="wellbeing_band", y="anxiety_score_0to27",
                         color="wellbeing_band", color_discrete_map=BAND_COLORS,
                         category_orders={"wellbeing_band": WELLBEING_CLASSES})
            fig.update_layout(height=340, margin=dict(t=10, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with r:
            fig = px.scatter(df, x="avg_sleep_hours", y="life_satisfaction_1to10",
                             color="wellbeing_band", color_discrete_map=BAND_COLORS,
                             opacity=0.65)
            fig.update_layout(height=340, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Platform usage")
        plat = (df.groupby("most_used_platform").size()
                  .reset_index(name="count").sort_values("count", ascending=True))
        fig = px.bar(plat, x="count", y="most_used_platform", orientation="h")
        fig.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Data Explorer
# --------------------------------------------------------------------------- #
with tab_explorer:
    if df.empty:
        st.info("Load a dataset to explore.")
    else:
        st.markdown("### Filters")
        f1, f2, f3, f4 = st.columns(4)
        region_f = f1.multiselect("Region", CATEGORIES["region"])
        platform_f = f2.multiselect("Platform", CATEGORIES["most_used_platform"])
        band_f = f3.multiselect("Wellbeing", WELLBEING_CLASSES)
        age_min, age_max = int(df["age"].min()), int(df["age"].max())
        age_f = f4.slider("Age range", age_min, age_max, (age_min, age_max))

        view = df.copy()
        if region_f:   view = view[view["region"].isin(region_f)]
        if platform_f: view = view[view["most_used_platform"].isin(platform_f)]
        if band_f:     view = view[view["wellbeing_band"].isin(band_f)]
        view = view[(view["age"] >= age_f[0]) & (view["age"] <= age_f[1])]

        st.caption(f"{len(view):,} of {len(df):,} rows")
        st.dataframe(view, use_container_width=True, height=520)

        buf = io.StringIO(); view.to_csv(buf, index=False)
        st.download_button("⬇️ Download filtered CSV", buf.getvalue(),
                           file_name="filtered.csv", mime="text/csv")

# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
with tab_predict:
    st.markdown("### Enter participant features")
    st.caption("Fill the form and run the RandomForest classifier.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 13, 90, 28)
            gender = st.selectbox("Gender", CATEGORIES["gender"])
            occupation = st.selectbox("Occupation", CATEGORIES["occupation"])
            region = st.selectbox("Region", CATEGORIES["region"])
            most_used_platform = st.selectbox("Most-used platform", CATEGORIES["most_used_platform"])
            platforms_used_count = st.slider("Platforms used", 1, 10, 3)
            daily_screen_hours = st.slider("Daily screen hours", 0.0, 16.0, 4.5, 0.1)
            daily_notifications = st.slider("Daily notifications", 0, 500, 80)
        with c2:
            night_time_use = st.selectbox("Night-time use", CATEGORIES["night_time_use"])
            minutes_to_first_check = st.slider("Minutes to first check after waking", 0, 240, 10)
            primary_purpose = st.selectbox("Primary purpose", CATEGORIES["primary_purpose"])
            avg_sleep_hours = st.slider("Avg sleep hours", 0.0, 12.0, 7.0, 0.1)
            anxiety = st.slider("Anxiety score (0–27)", 0, 27, 8)
            low_mood = st.slider("Low-mood score (0–27)", 0, 27, 7)
            life_sat = st.slider("Life satisfaction (1–10)", 1, 10, 6)
            loneliness = st.slider("Loneliness (1–10)", 1, 10, 4)
        with c3:
            self_esteem = st.slider("Self-esteem (1–10)", 1, 10, 6)
            fomo = st.slider("FOMO (1–10)", 1, 10, 5)
            social_comparison = st.slider("Social comparison (1–10)", 1, 10, 5)
            physical_activity = st.slider("Physical activity days/week", 0, 7, 3)
            uses_limits = st.selectbox("Uses screen-time limits", CATEGORIES["uses_screen_time_limits"])
            detox = st.selectbox("Attempted digital detox", CATEGORIES["attempted_digital_detox"])
            support = st.selectbox("Seeks mental-health support", CATEGORIES["seeks_mental_health_support"])

        submitted = st.form_submit_button("🚀 Run model prediction", use_container_width=True)

    if submitted:
        payload = {
            "age": age, "gender": gender, "occupation": occupation, "region": region,
            "most_used_platform": most_used_platform,
            "platforms_used_count": platforms_used_count,
            "daily_screen_hours": daily_screen_hours,
            "daily_notifications": daily_notifications,
            "night_time_use": night_time_use,
            "minutes_to_first_check_after_waking": minutes_to_first_check,
            "primary_purpose": primary_purpose,
            "avg_sleep_hours": avg_sleep_hours,
            "anxiety_score_0to27": anxiety,
            "low_mood_score_0to27": low_mood,
            "life_satisfaction_1to10": life_sat,
            "loneliness_1to10": loneliness,
            "self_esteem_1to10": self_esteem,
            "fomo_1to10": fomo,
            "social_comparison_1to10": social_comparison,
            "physical_activity_days_per_week": physical_activity,
            "uses_screen_time_limits": uses_limits,
            "attempted_digital_detox": detox,
            "seeks_mental_health_support": support,
        }
        with st.spinner("Scoring with RandomForest…"):
            pred, probs, latency = predict(model, payload)

        pill = {"Good": "pill-good", "Moderate": "pill-mod", "At-risk": "pill-risk"}[pred]
        st.markdown(
            f"""
            <div class='result-card'>
              <div class='pill {pill}'>Predicted wellbeing band</div>
              <h1 style='margin:.4rem 0 0;font-size:2.6rem;'>{pred}</h1>
              <p style='opacity:.75;margin:.25rem 0 1rem;'>
                Confidence: <b>{probs[pred]*100:.1f}%</b> · Latency: {latency:.0f} ms
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pc1, pc2 = st.columns([1, 1])
        with pc1:
            st.markdown("#### Class probabilities")
            prob_df = pd.DataFrame(
                {"class": WELLBEING_CLASSES,
                 "probability": [probs.get(c, 0.0) for c in WELLBEING_CLASSES]}
            )
            fig = px.bar(prob_df, x="probability", y="class", orientation="h",
                         color="class", color_discrete_map=BAND_COLORS,
                         range_x=[0, 1])
            fig.update_layout(height=260, showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with pc2:
            st.markdown("#### Confidence gauge")
            import plotly.graph_objects as go
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probs[pred] * 100,
                number={"suffix": "%"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": BAND_COLORS[pred]},
                       "steps": [
                           {"range": [0, 40],  "color": "rgba(239,68,68,.15)"},
                           {"range": [40, 70], "color": "rgba(234,179,8,.15)"},
                           {"range": [70, 100],"color": "rgba(16,185,129,.15)"},
                       ]},
            ))
            gauge.update_layout(height=260, margin=dict(t=10, b=10))
            st.plotly_chart(gauge, use_container_width=True)

        export_df = pd.DataFrame([{**payload, "prediction": pred,
                                    **{f"proba_{k}": v for k, v in probs.items()}}])
        st.download_button("⬇️ Export result (CSV)",
                            export_df.to_csv(index=False),
                            file_name="prediction.csv", mime="text/csv")