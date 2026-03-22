"""
app.py — ProfessorMatch  Streamlit Frontend
============================================

Hackathon-ready UI that wraps professor_recommendation_engine.py.

Run with:
    streamlit run app.py

No changes to the engine are made here — this file only imports and calls it.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

# ── Import recommendation engine (all logic lives there) ─────────────────────
from professor_recommendation_engine import (
    fuse_features,
    make_mock_linkedin_data,
    make_mock_official_data,
    make_mock_rmp_data,
    recommend_professors,
)


# ---------------------------------------------------------------------------
# Page config  (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ProfessorMatch",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

# Maps the UI selectbox key → engine mode string
GOAL_OPTIONS: dict[str, str] = {
    "gpa_focused":      "GPA Focused — prioritise easy grading and low difficulty",
    "balanced":         "Balanced — quality, clarity, and moderate workload",
    "learning_focused": "Learning Focused — depth, engagement, intellectual challenge",
    "career_focused":   "Career Focused — industry experience and networking value",
}

# Human-readable source names used in the transparency panel
SOURCE_META: dict[str, dict] = {
    "rmp": {
        "name":   "RateMyProfessors",
        "icon":   "⭐",
        "desc":   "Primary source. Student reviews, ratings, difficulty scores, and tags.",
        "weight": "50% of source confidence",
    },
    "linkedin": {
        "name":   "LinkedIn",
        "icon":   "💼",
        "desc":   "Secondary source. Career background, industry experience, research profile.",
        "weight": "30% of source confidence",
    },
    "official": {
        "name":   "Official Course Info",
        "icon":   "📚",
        "desc":   "Optional source. Syllabus, grading policy, assignment structure.",
        "weight": "20% of source confidence",
    },
}

# Ordered list of dimension keys used in charts and comparison table
DIMENSION_KEYS: list[str] = [
    "quality",
    "clarity",
    "engagement",
    "fairness",
    "difficulty",
    "workload",
    "career_relevance",
    "industry_experience",
    "academic_strength",
    "connect_opportunity",
]


def _pct(value: float) -> str:
    """Format a 0–1 float as a percentage string."""
    return f"{value:.0%}"


def _confidence_label(score: float) -> str:
    """Convert a 0–1 confidence score to a human-readable tier."""
    if score >= 0.70:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


def _dim_label(key: str) -> str:
    """Convert a snake_case dimension key to a Title Case display label."""
    return key.replace("_", " ").title()


def _connect_band(score: float) -> str:
    """Map a 0–1 connect_opportunity score to a human-readable band label."""
    if score >= 0.65:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Limited"


def _build_dim_df(rec: dict) -> pd.DataFrame:
    """
    Return a single-column DataFrame of dimension scores for one professor,
    suitable for st.bar_chart.  Index = dimension labels, column = "Score".
    """
    ds = rec["dimension_scores"]
    data = {_dim_label(k): ds.get(k, 0.0) for k in DIMENSION_KEYS}
    return pd.DataFrame({"Score": data})


@st.cache_data(show_spinner="Loading professor data…")
def _load_all_features() -> list[dict]:
    """
    Fuse mock data from all three sources into UnifiedProfessorFeatureSchema dicts.
    In production, replace the make_mock_*() calls with real database fetches.
    """
    all_rmp      = make_mock_rmp_data()
    all_linkedin = make_mock_linkedin_data()
    all_official = make_mock_official_data()

    li_by_name  = {d["professor_name"]: d for d in all_linkedin}
    off_by_name = {d["professor_name"]: d for d in all_official}

    return [
        fuse_features(
            rmp,
            linkedin = li_by_name.get(rmp["professor_name"]),
            official = off_by_name.get(rmp["professor_name"]),
        )
        for rmp in all_rmp
    ]


# ---------------------------------------------------------------------------
# SECTION 1 — Header
# ---------------------------------------------------------------------------

st.markdown(
    "<h1 style='margin-bottom:0'>🎓 ProfessorMatch</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#888; margin-top:4px; font-size:1.05rem'>"
    "Find the best professor for your learning goals, workload preference, "
    "and career development."
    "</p>",
    unsafe_allow_html=True,
)
st.caption("ProfessorMatch — Improved Demo UI")
st.divider()


# ---------------------------------------------------------------------------
# SECTION 2 — Sidebar input panel
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("🔍 Student Preferences")

    # Course filter — informational only in demo mode; used to filter real data
    st.subheader("Course Filter")
    school      = st.text_input("School",       value="State University")
    department  = st.text_input("Department",   value="Computer Science")
    course_code = st.text_input("Course Code",  value="CS301")

    st.subheader("Primary Objective")
    goal_key = st.selectbox(
        "What matters most to you?",
        options=list(GOAL_OPTIONS.keys()),
        format_func=lambda k: GOAL_OPTIONS[k],
        index=1,  # default: balanced
    )

    st.subheader("Workload Tolerance")
    workload_tolerance = st.select_slider(
        "How much workload can you handle?",
        options=["low", "medium", "high"],
        value="medium",
    )

    st.subheader("Preferences")
    prefer_clear_grading     = st.checkbox("Prefer clear grading criteria",  value=True)
    prefer_engaging_lectures = st.checkbox("Prefer engaging lectures",        value=True)
    career_oriented          = st.checkbox("Career-oriented student",         value=False)
    open_to_networking       = st.checkbox("Open to LinkedIn networking",      value=False)

    st.divider()

    # Lightweight visual demo toggle — controls card density and networking display
    # Classic: per-card dimension charts + raw score %; Improved: cleaner, band labels
    ui_mode      = st.radio("UI Mode", ["Improved", "Classic"], horizontal=True)
    improved_mode = ui_mode == "Improved"

    st.divider()
    run_button = st.button(
        "🚀  Generate Recommendations",
        use_container_width=True,
        type="primary",
    )


# ---------------------------------------------------------------------------
# SECTION 3 — Run recommendation engine when button is clicked
# ---------------------------------------------------------------------------

# Persist results across reruns so the UI doesn't blank when a checkbox changes
if "results" not in st.session_state:
    st.session_state["results"] = None
if "active_mode" not in st.session_state:
    st.session_state["active_mode"] = "balanced"

if run_button:
    preferences = {
        "student_goal":             goal_key,
        "workload_tolerance":       workload_tolerance,
        "prefer_clear_grading":     prefer_clear_grading,
        "prefer_engaging_lectures": prefer_engaging_lectures,
        "career_oriented":          career_oriented,
        "open_to_networking":       open_to_networking,
    }
    all_features = _load_all_features()
    results      = recommend_professors(all_features, preferences, mode=goal_key, top_n=5)

    st.session_state["results"]     = results
    st.session_state["active_mode"] = goal_key


# ---------------------------------------------------------------------------
# Results area — only rendered after the engine has run at least once
# ---------------------------------------------------------------------------

results = st.session_state.get("results")

if not results:
    # ── Pre-run landing state ─────────────────────────────────────────────
    st.info(
        "👈  Set your preferences in the sidebar and click **Get Recommendations** "
        "to find your best professor match.",
        icon="🎓",
    )
    with st.expander("What will I see?"):
        st.markdown("""
- **Top professor match** with match score and confidence level
- **Professor cards** with reasons, concerns, and dimension breakdown
- **LinkedIn networking panel** with connect reasons and a direct profile link
- **Side-by-side comparison** of your top 3 matches
- **Full explanation** of why each professor fits your goals
- **Data source transparency** showing which sources were used and how
        """)
    st.stop()


# ─── Convenience aliases ─────────────────────────────────────────────────────
top  = results[0]
top3 = results[:3]


# ---------------------------------------------------------------------------
# SECTION 4 — Top match hero banner
# ---------------------------------------------------------------------------

st.subheader("🏆 Best Professor for Your Preferences")

hero_col1, hero_col2, hero_col3, hero_col4 = st.columns([3, 2, 2, 2])

with hero_col1:
    st.markdown(f"### {top['professor_name']}")
    if top.get("course_code"):
        st.caption(f"Course: {top['course_code']}")
    st.caption(f"Mode: **{top['match_type']}**")

hero_col2.metric("Fit Score",        _pct(top["match_score"]))
hero_col3.metric("Data Confidence",  _confidence_label(top["confidence_score"]))
hero_col4.metric("Controversy",      _pct(top["controversy_score"]))

if top.get("mixed_feedback_warning"):
    st.warning(f"⚠️  {top['mixed_feedback_warning']}")

st.divider()


# ---------------------------------------------------------------------------
# Main tab area
# ---------------------------------------------------------------------------

tab_cards, tab_compare, tab_insights = st.tabs(
    ["📋  Recommended Professors", "📊  Compare Professors", "💡  Data Sources & Insights"]
)


# ===========================================================================
# TAB 1 — Professor Cards  (Sections 5, 6, and per-professor dimension chart)
# ===========================================================================

with tab_cards:
    for rank_idx, rec in enumerate(top3):
        rank_emoji = ["🥇", "🥈", "🥉"][rank_idx]
        name       = rec["professor_name"]

        with st.container(border=True):

            # ── Card header ──────────────────────────────────────────────
            hcol1, hcol2, hcol3, hcol4 = st.columns([4, 1, 1, 1])
            with hcol1:
                st.markdown(f"### {rank_emoji}  {name}")
                if rec.get("course_code"):
                    st.caption(f"Course: {rec['course_code']}")
                st.caption(f"**{rec['match_type']}**")
            hcol2.metric("Fit Score",       _pct(rec["match_score"]))
            hcol3.metric("Data Confidence", _confidence_label(rec["confidence_score"]))
            hcol4.metric("Controversy",     _pct(rec["controversy_score"]))

            if rec.get("mixed_feedback_warning"):
                st.warning(rec["mixed_feedback_warning"], icon="⚠️")

            st.divider()

            # ── Why / Concerns ───────────────────────────────────────────
            wcol1, wcol2 = st.columns(2)

            with wcol1:
                why = rec.get("why_recommended") or []
                if why:
                    st.markdown("**✅  Why Recommended**")
                    for reason in why:
                        st.markdown(f"- {reason}")
                else:
                    st.markdown("*No specific match reasons generated.*")

            with wcol2:
                concerns = rec.get("potential_concerns") or []
                if concerns:
                    st.markdown("**⚠️  Potential Concerns**")
                    for concern in concerns:
                        st.markdown(f"- {concern}")
                else:
                    st.markdown("*No concerns flagged.*")

            # ── Sources used ─────────────────────────────────────────────
            if rec.get("sources_used"):
                source_display = [
                    SOURCE_META.get(s, {}).get("name", s)
                    for s in rec["sources_used"]
                ]
                st.caption(f"📂  Sources: {' · '.join(source_display)}")

            # ── Dimension bar chart — Classic mode only ───────────────────
            # Improved mode omits per-card charts; see the Compare tab instead.
            if not improved_mode:
                st.markdown("**Dimension Scores**")
                st.bar_chart(_build_dim_df(rec), height=220, use_container_width=True)

            # ── LinkedIn / networking panel ───────────────────────────────
            # Only shown when a LinkedIn profile URL is available
            if rec.get("linkedin_profile_url"):
                st.divider()
                st.markdown("**🔗  Networking Opportunity**")

                co_score = rec["dimension_scores"].get("connect_opportunity", 0.0)
                lcol1, lcol2 = st.columns([3, 1])

                with lcol1:
                    if improved_mode:
                        # Improved: show band label instead of raw percentage
                        band = _connect_band(co_score)
                        band_colour = {"High": "🟢", "Medium": "🟡", "Limited": "🔴"}
                        st.markdown(
                            f"**Networking Potential:** {band_colour[band]}  {band}"
                        )
                    else:
                        # Classic: show raw progress bar with score
                        st.progress(
                            co_score,
                            text=f"Connect Opportunity Score: {_pct(co_score)}",
                        )

                    connect_reasons = rec.get("connect_reasons") or []
                    if connect_reasons:
                        for cr in connect_reasons:
                            st.markdown(f"★  {cr}")
                    else:
                        st.caption(
                            "LinkedIn profile available — "
                            "networking potential below threshold for specific reasons."
                        )

                with lcol2:
                    st.link_button(
                        "View LinkedIn Profile",
                        rec["linkedin_profile_url"],
                        use_container_width=True,
                    )

        # Spacer between cards
        st.write("")


# ===========================================================================
# TAB 2 — Comparison View  (Section 8)
# ===========================================================================

with tab_compare:
    st.subheader("📊  Side-by-Side Dimension Comparison")
    st.caption("Top 3 professors compared across all scoring dimensions.")

    # Build DataFrame: rows = dimension labels, columns = professor names
    compare_data = {
        rec["professor_name"]: [
            rec["dimension_scores"].get(k, 0.0) for k in DIMENSION_KEYS
        ]
        for rec in top3
    }
    compare_df = pd.DataFrame(
        compare_data,
        index=[_dim_label(k) for k in DIMENSION_KEYS],
    )

    # Grouped bar chart — one colour per professor
    st.bar_chart(compare_df, height=420, use_container_width=True)

    # Colour-coded raw scores table beneath the chart
    with st.expander("📄  View raw scores table"):
        try:
            styled = compare_df.style.format("{:.2f}").background_gradient(
                cmap="RdYlGn", vmin=0.0, vmax=1.0
            )
            st.dataframe(styled, use_container_width=True)
        except Exception:
            # Fallback if Pandas Styler gradient not available in this environment
            st.dataframe(compare_df.style.format("{:.2f}"), use_container_width=True)

    # Quick-read summary table: match scores + key metadata
    st.subheader("Quick Summary")
    summary_rows = []
    for rec in top3:
        summary_rows.append({
            "Professor":        rec["professor_name"],
            "Fit Score":        _pct(rec["match_score"]),
            "Match Type":       rec["match_type"],
            "Data Confidence":  _confidence_label(rec["confidence_score"]),
            "Controversy":      _pct(rec["controversy_score"]),
            "Sources":          ", ".join(rec.get("sources_used") or []),
            "LinkedIn":         "✅" if rec.get("linkedin_profile_url") else "—",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 3 — Insights & Sources  (Sections 9 + 10)
# ===========================================================================

with tab_insights:

    # ── SECTION 9 — Explanation panel ────────────────────────────────────────
    st.subheader("💡  Why Your Top Match Fits Your Goals")

    all_top_reasons = (top.get("why_recommended") or []) + (top.get("connect_reasons") or [])
    if all_top_reasons:
        st.markdown(
            f"**{top['professor_name']}** matches your **{top['match_type']}** goal because:"
        )
        for reason in all_top_reasons:
            st.markdown(f"- {reason}")
    else:
        st.info("No specific reasons were generated for this match.")

    if top.get("mixed_feedback_warning"):
        st.warning(top["mixed_feedback_warning"])

    st.divider()

    # Full explanations for all top 3
    st.subheader("📝  Full Explanation — Top 3 Professors")
    for rank_idx, rec in enumerate(top3):
        rank_emoji = ["🥇", "🥈", "🥉"][rank_idx]
        with st.expander(
            f"{rank_emoji}  {rec['professor_name']}  —  {_pct(rec['match_score'])} match"
        ):
            ecol1, ecol2 = st.columns(2)

            with ecol1:
                why = rec.get("why_recommended") or []
                if why:
                    st.markdown("**Why Recommended**")
                    for r in why:
                        st.markdown(f"✅  {r}")

                connect_reasons = rec.get("connect_reasons") or []
                if connect_reasons:
                    st.markdown("**Networking Value**")
                    for r in connect_reasons:
                        st.markdown(f"★  {r}")
                    if rec.get("linkedin_profile_url"):
                        st.link_button(
                            "View LinkedIn Profile",
                            rec["linkedin_profile_url"],
                        )

            with ecol2:
                concerns = rec.get("potential_concerns") or []
                if concerns:
                    st.markdown("**Potential Concerns**")
                    for c in concerns:
                        st.markdown(f"⚠️  {c}")

                if rec.get("mixed_feedback_warning"):
                    st.warning(rec["mixed_feedback_warning"])

            # Dimension scores as a mini-table
            ds = rec["dimension_scores"]
            dim_table = pd.DataFrame(
                {
                    "Dimension": [_dim_label(k) for k in DIMENSION_KEYS],
                    "Score":     [f"{ds.get(k, 0.0):.2f}" for k in DIMENSION_KEYS],
                }
            )
            st.dataframe(dim_table, use_container_width=True, hide_index=True)

    st.divider()

    # ── SECTION 10 — Data source transparency ────────────────────────────────
    st.subheader("🗂️  Data Source Transparency")
    st.caption("Which sources contributed to the recommendations above.")

    # Collect all sources used across top 3
    all_sources_used: set[str] = set()
    for rec in top3:
        all_sources_used.update(rec.get("sources_used") or [])

    scol1, scol2, scol3 = st.columns(3)
    for col, src_key in zip([scol1, scol2, scol3], ["rmp", "linkedin", "official"]):
        info      = SOURCE_META[src_key]
        available = src_key in all_sources_used
        status    = "✅  Available" if available else "❌  Not used"
        with col:
            with st.container(border=True):
                st.markdown(f"{info['icon']}  **{info['name']}**")
                st.caption(info["desc"])
                st.caption(f"Weight: {info['weight']}")
                st.markdown(status)

    # Methodology explainer
    with st.expander("ℹ️  How scores are calculated"):
        st.markdown("""
**Recommendation methodology — rule-based weighted scoring (no ML)**

| Step | What happens |
|------|-------------|
| 1. Feature extraction | Raw RMP, LinkedIn, and syllabus fields are converted to normalised 0–1 scores per dimension |
| 2. Multi-source fusion | Dimensions are blended across sources using source-specific weights (RMP=primary, LinkedIn=secondary, Official=optional) |
| 3. Mode scoring | A weighted sum of dimension scores is computed under the chosen goal mode |
| 4. Preference nudges | Small ±3–5% adjustments are applied for individual preferences (clear grading, networking, etc.) |
| 5. Output | Professors are sorted by final match score; explanation strings are generated from thresholds |

**Key scores:**
- **Confidence** — driven by review volume, data recency, and number of available sources
- **Controversy** — flags professors with polarised student feedback (extreme positive + negative words co-occurring)
- **Connect Opportunity** — `0.4 × career_relevance + 0.3 × industry_experience + 0.2 × academic_strength + 0.1 × confidence`
        """)
