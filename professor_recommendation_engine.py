"""
Professor Recommendation Engine
================================

Rule-based, multi-source professor recommendation system built for hackathon speed
and demo-readiness. No collaborative filtering or deep learning — just weighted
scoring, keyword signals, and multi-source fusion.

Sources (by priority):
  RMP (RateMyProfessors) — primary:  teaching quality, student experience
  LinkedIn               — secondary: career background, industry experience
  Official course data   — optional:  workload, structure, grading policy

Ranking modes:
  gpa_focused      — prioritises easy grading, clarity, low workload
  balanced         — even-handed across quality, clarity, and engagement
  learning_focused — prioritises clarity, engagement, academic depth
  career_focused   — prioritises industry experience and career relevance

Usage:
    python professor_recommendation_engine.py
"""

from __future__ import annotations

import math
import textwrap
from typing import Any


# ---------------------------------------------------------------------------
# Tag → feature signal map  (RateMyProfessors tags)
# ---------------------------------------------------------------------------
# Each tag nudges one or more dimension scores by a small delta (+/-).
# Deltas are accumulated across all tags before clamping to [0, 1].

TAG_SIGNALS: dict[str, dict[str, float]] = {
    # Engagement / charisma
    "Inspirational":             {"engagement": +0.35},
    "Hilarious":                 {"engagement": +0.30},
    "Caring":                    {"engagement": +0.20, "fairness": +0.10},
    "Accessible outside class":  {"engagement": +0.25, "clarity":  +0.10},
    "Participation matters":     {"engagement": +0.20},
    # Clarity / communication
    "Amazing lectures":          {"clarity": +0.30, "engagement": +0.25},
    "Gives good feedback":       {"clarity": +0.25, "fairness":   +0.20},
    "Clear grading criteria":    {"fairness": +0.30, "clarity":   +0.20},
    # Teaching quality
    "Respected":                 {"quality": +0.20},
    "Would take again":          {"quality": +0.25},
    # Difficulty / workload
    "Lots of homework":          {"workload":   +0.30},
    "Get ready to read":         {"workload":   +0.25},
    "So many papers":            {"workload":   +0.30},
    "Test heavy":                {"workload":   +0.20, "difficulty": +0.15},
    "Heavy grader":              {"difficulty": +0.25, "fairness":  -0.10},
    "Tough grader":              {"difficulty": +0.20, "fairness":  -0.10},
    "Graded by few things":      {"workload":   -0.20},
    # Career / industry signal
    "Industry experience":       {"career_relevance":    +0.40, "industry_experience": +0.30},
    "Real-world examples":       {"career_relevance":    +0.30},
}

# Review text keywords → dimension signals (case-insensitive substring match)
TEXT_SIGNALS: dict[str, dict[str, float]] = {
    "explains clearly":    {"clarity":          +0.20},
    "easy to understand":  {"clarity":          +0.20},
    "very clear":          {"clarity":          +0.15},
    "well organized":      {"clarity":          +0.15, "fairness": +0.10},
    "engaging":            {"engagement":       +0.20},
    "passionate":          {"engagement":       +0.20},
    "boring":              {"engagement":       -0.20},
    "monotone":            {"engagement":       -0.15},
    "lots of work":        {"workload":         +0.20},
    "heavy workload":      {"workload":         +0.25},
    "fair grader":         {"fairness":         +0.20},
    "unfair":              {"fairness":         -0.25},
    "industry":            {"career_relevance": +0.15},
    "real world":          {"career_relevance": +0.20},
    "research":            {"academic_strength":+0.15},
    "published":           {"academic_strength":+0.20},
}

# Extreme sentiment words used to detect polarised / controversial reviews
_POSITIVE_EXTREMES = {"amazing", "best", "excellent", "love", "perfect", "outstanding"}
_NEGATIVE_EXTREMES = {"worst", "terrible", "awful", "avoid", "useless", "waste", "horrible"}


# ---------------------------------------------------------------------------
# Scoring weights by ranking mode
# ---------------------------------------------------------------------------
# Positive weight  → higher dimension score raises the final match_score.
# Negative weight  → higher dimension score LOWERS the final match_score.
#                    Used for difficulty and workload in GPA-focused mode.
#
# Normalisation later ensures all modes produce scores in [0, 1].

RANKING_MODE_WEIGHTS: dict[str, dict[str, float]] = {
    "gpa_focused": {
        # Lower difficulty and workload = better for GPA
        "quality":              +0.15,
        "difficulty":           -0.25,
        "workload":             -0.20,
        "clarity":              +0.20,
        "fairness":             +0.25,
        "engagement":           +0.10,
        "career_relevance":     +0.05,
        "industry_experience":  +0.00,
        "academic_strength":    +0.00,
        # Networking irrelevant when goal is purely GPA
        "connect_opportunity":  +0.00,
    },
    "balanced": {
        "quality":              +0.20,
        "difficulty":           -0.08,
        "workload":             -0.07,
        "clarity":              +0.18,
        "fairness":             +0.15,
        "engagement":           +0.18,
        "career_relevance":     +0.12,
        "industry_experience":  +0.05,
        "academic_strength":    +0.05,
        # Small connect boost — networking is a mild bonus in balanced mode
        "connect_opportunity":  +0.03,
    },
    "learning_focused": {
        # Moderate difficulty is a POSITIVE signal — challenge deepens learning
        "quality":              +0.20,
        "difficulty":           +0.08,
        "workload":             -0.05,
        "clarity":              +0.25,
        "fairness":             +0.08,
        "engagement":           +0.25,
        "career_relevance":     +0.05,
        "industry_experience":  +0.02,
        "academic_strength":    +0.12,
        # Slight connect signal — research-active professors worth knowing
        "connect_opportunity":  +0.02,
    },
    "career_focused": {
        "quality":              +0.12,
        "difficulty":           -0.05,
        "workload":             -0.05,
        "clarity":              +0.10,
        "fairness":             +0.08,
        "engagement":           +0.10,
        "career_relevance":     +0.25,
        "industry_experience":  +0.18,
        "academic_strength":    +0.12,
        # Connect opportunity matters most in career mode — professor as network node
        "connect_opportunity":  +0.10,
    },
}

# Human-readable labels shown in FrontendRecommendationOutputSchema
_MATCH_TYPES: dict[str, str] = {
    "gpa_focused":      "GPA-Friendly",
    "balanced":         "Well-Rounded",
    "learning_focused": "High Learning Value",
    "career_focused":   "Career Booster",
}

# Score thresholds used to generate human-readable explanations
_THRESHOLDS = {
    "high_quality":          0.75,
    "high_engagement":       0.70,
    "high_clarity":          0.70,
    "high_fairness":         0.70,
    "high_career_relevance":    0.65,
    "high_industry_exp":        0.60,
    "high_academic":            0.65,
    "high_connect_opportunity": 0.60,   # threshold to surface connect reasons
    "low_reviews":              5,       # fewer than N reviews → warn about sparse data
    "concern_workload":         0.65,
    "concern_difficulty":       0.70,
    "concern_fairness":         0.45,
    "controversy_warning":      0.50,
}


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

def make_mock_rmp_data() -> list[dict[str, Any]]:
    """Return a list of RMPRawSchema-shaped dicts for 5 mock professors."""
    return [
        {
            # Alice Chen — strong teaching quality, clear grader, mild research background
            "professor_name": "Alice Chen",
            "school":         "State University",
            "department":     "Computer Science",
            "course_code":    "CS301",
            "overall_rating": 4.5,
            "difficulty":     3.2,
            "would_take_again": 88.0,
            "num_reviews":    120,
            "tags": [
                "Amazing lectures",
                "Clear grading criteria",
                "Gives good feedback",
                "Accessible outside class",
            ],
            "review_texts": [
                "Dr. Chen explains clearly every concept. Highly recommend.",
                "Very clear grading — I always knew exactly where I stood.",
                "Well organized and engaging. One of the best CS professors here.",
            ],
            "review_dates": ["2025-11-01", "2025-09-15", "2025-08-20"],
        },
        {
            # Bob Smith — industry veteran, practical focus, mixed organisation
            "professor_name": "Bob Smith",
            "school":         "State University",
            "department":     "Computer Science",
            "course_code":    "CS301",
            "overall_rating": 3.8,
            "difficulty":     3.5,
            "would_take_again": 72.0,
            "num_reviews":    55,
            "tags": ["Real-world examples", "Industry experience", "Lots of homework"],
            "review_texts": [
                "Bob brings real world examples from his startup days. Amazing professor!",
                "Worst organised course I've taken. Complete waste of time for theory.",
                "Heavy workload but the industry insights make it worth it.",
            ],
            "review_dates": ["2025-10-10", "2025-07-05", "2025-06-30"],
        },
        {
            # Carol Davis — easy grader, clear expectations, low difficulty
            "professor_name": "Carol Davis",
            "school":         "State University",
            "department":     "Computer Science",
            "course_code":    "CS301",
            "overall_rating": 4.2,
            "difficulty":     1.8,
            "would_take_again": 91.0,
            "num_reviews":    85,
            "tags": ["Clear grading criteria", "Graded by few things", "Caring"],
            "review_texts": [
                "Easy to pass, fair grader, very caring. Best GPA booster.",
                "Low difficulty, clear expectations. I always knew what was expected.",
                "Not the most engaging but always fair and very clear.",
            ],
            "review_dates": ["2025-12-01", "2025-10-20", "2025-09-01"],
        },
        {
            # David Park — inspirational but demanding, top academic researcher
            "professor_name": "David Park",
            "school":         "State University",
            "department":     "Computer Science",
            "course_code":    "CS301",
            "overall_rating": 4.6,
            "difficulty":     4.5,
            "would_take_again": 80.0,
            "num_reviews":    98,
            "tags": ["Inspirational", "Test heavy", "Lots of homework", "Respected"],
            "review_texts": [
                "Hardest class I've taken but I learned so much. Absolutely passionate.",
                "Professor Park is inspiring but the workload is intense.",
                "Best professor for deep learning. Very engaging, extremely tough.",
            ],
            "review_dates": ["2025-11-15", "2025-10-01", "2025-08-10"],
        },
        {
            # Emily Johnson — solid but unremarkable, CS education researcher
            "professor_name": "Emily Johnson",
            "school":         "State University",
            "department":     "Computer Science",
            "course_code":    "CS301",
            "overall_rating": 4.1,
            "difficulty":     2.9,
            "would_take_again": 78.0,
            "num_reviews":    42,
            "tags": ["Caring", "Gives good feedback", "Participation matters"],
            "review_texts": [
                "Decent professor, caring and gives good feedback.",
                "Average difficulty, fair grader. Nothing exceptional but solid.",
            ],
            "review_dates": ["2025-09-20", "2025-07-15"],
        },
    ]


def make_mock_linkedin_data() -> list[dict[str, Any]]:
    """Return LinkedInRawSchema-shaped mock data for the same 5 professors."""
    return [
        {
            "professor_name":               "Alice Chen",
            "school":                       "State University",
            "department":                   "Computer Science",
            "course_code":                  "CS301",
            "headline":                     "Professor of Computer Science | NLP Researcher",
            "current_title":                "Associate Professor",
            "current_company_or_university":"State University",
            "about":                        "Researcher in NLP and ML with 10 years of academic experience.",
            "past_experiences": [
                {"title": "Research Scientist", "company": "Google Brain", "years": 2},
            ],
            "education": [
                {"degree": "PhD", "field": "Computer Science", "school": "MIT"},
            ],
            "skills":               ["Python", "NLP", "Machine Learning", "Teaching"],
            "certifications":       [],
            "research_interests":   ["Natural Language Processing", "AI Ethics"],
            "industry_experience_years": 2.0,
            "has_industry_experience":   True,
            "has_research_background":   True,
            "profile_last_updated":  "2025-10-01",
            "linkedin_profile_url":  "https://linkedin.com/in/alicechen",
        },
        {
            "professor_name":               "Bob Smith",
            "school":                       "State University",
            "department":                   "Computer Science",
            "course_code":                  "CS301",
            "headline":                     "Adjunct Prof | Serial Entrepreneur | Ex-CTO at TechStartup",
            "current_title":                "Adjunct Professor",
            "current_company_or_university":"State University",
            "about":                        "Built and sold two software startups. Now teaching to give back.",
            "past_experiences": [
                {"title": "CTO",              "company": "TechStartup Inc.", "years": 5},
                {"title": "Software Engineer","company": "Amazon",           "years": 3},
            ],
            "education": [
                {"degree": "BS", "field": "Computer Science", "school": "UC Berkeley"},
            ],
            "skills":               ["Software Architecture", "Entrepreneurship", "Cloud Computing", "Leadership"],
            "certifications":       ["AWS Solutions Architect"],
            "research_interests":   [],
            "industry_experience_years": 12.0,
            "has_industry_experience":   True,
            "has_research_background":   False,
            "profile_last_updated":  "2025-11-15",
            "linkedin_profile_url":  "https://linkedin.com/in/bobsmith",
        },
        {
            "professor_name":               "Carol Davis",
            "school":                       "State University",
            "department":                   "Computer Science",
            "course_code":                  "CS301",
            "headline":                     "Lecturer in CS | Passionate about Undergraduate Education",
            "current_title":                "Lecturer",
            "current_company_or_university":"State University",
            "about":                        "Dedicated to making CS approachable for all students.",
            "past_experiences": [
                {"title": "Teaching Assistant", "company": "State University", "years": 4},
            ],
            "education": [
                {"degree": "MS", "field": "Computer Science", "school": "State University"},
            ],
            "skills":               ["Teaching", "Curriculum Design", "Python", "Java"],
            "certifications":       [],
            "research_interests":   [],
            "industry_experience_years": 0.0,
            "has_industry_experience":   False,
            "has_research_background":   False,
            "profile_last_updated":  "2025-06-01",
            "linkedin_profile_url":  None,
        },
        {
            "professor_name":               "David Park",
            "school":                       "State University",
            "department":                   "Computer Science",
            "course_code":                  "CS301",
            "headline":                     "Professor | Systems Researcher | Author of 40+ papers",
            "current_title":                "Full Professor",
            "current_company_or_university":"State University",
            "about":                        "My research focuses on distributed systems and scalability.",
            "past_experiences": [
                {"title": "Visiting Researcher", "company": "Microsoft Research", "years": 1},
            ],
            "education": [
                {"degree": "PhD", "field": "Computer Science",  "school": "Stanford"},
                {"degree": "BS",  "field": "Mathematics",       "school": "Caltech"},
            ],
            "skills":               ["Distributed Systems", "Research", "C++", "Algorithms"],
            "certifications":       [],
            "research_interests":   ["Distributed Systems", "Fault Tolerance", "Cloud Computing"],
            "industry_experience_years": 1.0,
            "has_industry_experience":   True,
            "has_research_background":   True,
            "profile_last_updated":  "2025-12-01",
            "linkedin_profile_url":  "https://linkedin.com/in/davidpark",
        },
        {
            "professor_name":               "Emily Johnson",
            "school":                       "State University",
            "department":                   "Computer Science",
            "course_code":                  "CS301",
            "headline":                     "Assistant Professor | CS Education Researcher",
            "current_title":                "Assistant Professor",
            "current_company_or_university":"State University",
            "about":                        "Research in CS education and student engagement.",
            "past_experiences":             [],
            "education": [
                {"degree": "PhD", "field": "CS Education", "school": "Carnegie Mellon"},
            ],
            "skills":               ["Teaching", "Research", "Python", "Data Analysis"],
            "certifications":       [],
            "research_interests":   ["CS Education", "Pedagogy"],
            "industry_experience_years": None,
            "has_industry_experience":   False,
            "has_research_background":   True,
            "profile_last_updated":  "2025-08-10",
            "linkedin_profile_url":  "https://linkedin.com/in/emilyjohnson",
        },
    ]


def make_mock_official_data() -> list[dict[str, Any]]:
    """Return OfficialCourseRawSchema-shaped mock data for 3 of the 5 professors.
    Not all professors have official course data — this is intentional to test
    graceful degradation when a source is missing.
    """
    return [
        {
            # Alice Chen — structured, fair, moderate load
            "school":              "State University",
            "department":          "Computer Science",
            "course_code":         "CS301",
            "course_title":        "Algorithms and Data Structures",
            "professor_name":      "Alice Chen",
            "section":             "001",
            "modality":            "in-person",
            "schedule":            "MWF 10:00–11:00",
            "syllabus_text":       "3 programming assignments, 1 midterm, 1 final. Weekly readings.",
            "grading_policy_text": "Assignments 40%, Midterm 25%, Final 35%. Rubric provided for every deliverable.",
            "assignment_info":     "3 programming assignments — pair work allowed.",
            "exam_info":           "Open-note midterm, closed-book final.",
        },
        {
            # David Park — heavy load, rigorous, no curves
            "school":              "State University",
            "department":          "Computer Science",
            "course_code":         "CS301",
            "course_title":        "Algorithms and Data Structures",
            "professor_name":      "David Park",
            "section":             "002",
            "modality":            "in-person",
            "schedule":            "TTh 13:00–14:30",
            "syllabus_text":       "8 weekly assignments, 2 exams, 1 research project. Heavy reading list.",
            "grading_policy_text": "Assignments 30%, Exams 40%, Project 30%. No curves applied.",
            "assignment_info":     "8 weekly assignments, individual work only.",
            "exam_info":           "Two in-class closed-book exams plus a research paper.",
        },
        {
            # Carol Davis — light load, flexible, easy exam format
            "school":              "State University",
            "department":          "Computer Science",
            "course_code":         "CS301",
            "course_title":        "Algorithms and Data Structures",
            "professor_name":      "Carol Davis",
            "section":             "003",
            "modality":            "hybrid",
            "schedule":            "MW 14:00–15:30",
            "syllabus_text":       "2 assignments, 1 group project, 1 final. Flexible deadlines.",
            "grading_policy_text": "Assignments 30%, Project 30%, Final 40%. Partial credit given generously.",
            "assignment_info":     "2 light assignments plus a group project.",
            "exam_info":           "Take-home final, open-book.",
        },
    ]


def make_mock_preferences() -> dict[str, Any]:
    """Return a sample UserPreferenceSchema dict."""
    return {
        "student_goal":             "learning",  # "gpa" | "learning" | "career" | "balanced"
        "workload_tolerance":       "medium",     # "low" | "medium" | "high"
        "prefer_clear_grading":     True,
        "prefer_engaging_lectures": True,
        "career_oriented":          False,
        "open_to_networking":       False,        # True → connect_opportunity gives a small score boost
    }


def make_mock_career_preferences() -> dict[str, Any]:
    """Return a career-oriented student with networking open preferences for demo."""
    return {
        "student_goal":             "career",
        "workload_tolerance":       "high",
        "prefer_clear_grading":     False,
        "prefer_engaging_lectures": True,
        "career_oriented":          True,
        "open_to_networking":       True,   # enables connect_opportunity score nudge
    }


# ---------------------------------------------------------------------------
# Feature extraction — RMP (primary source)
# ---------------------------------------------------------------------------

def normalize_rmp_signals(rmp: dict[str, Any]) -> dict[str, float]:
    """
    Convert raw RMP fields to normalised [0, 1] floats.

      overall_rating   (1–5)   → quality_score
      difficulty       (1–5)   → difficulty_score   (higher = harder)
      would_take_again (0–100) → wta_score
      num_reviews              → review_volume_score (log-scaled, saturates ~200)
    """
    quality   = rmp.get("overall_rating") or 3.0
    diff      = rmp.get("difficulty")     or 3.0
    wta       = rmp.get("would_take_again")
    n_reviews = rmp.get("num_reviews")    or 0

    quality_score       = (quality - 1.0) / 4.0                       # [0, 1]
    difficulty_score    = (diff   - 1.0) / 4.0                        # [0, 1]
    wta_score           = (wta / 100.0) if wta is not None else 0.50  # default neutral
    review_volume_score = min(1.0, math.log1p(n_reviews) / math.log1p(200))

    return {
        "quality_score":       round(quality_score,       4),
        "difficulty_score":    round(difficulty_score,    4),
        "wta_score":           round(wta_score,           4),
        "review_volume_score": round(review_volume_score, 4),
    }


def extract_rmp_tag_features(tags: list[str]) -> dict[str, float]:
    """
    Aggregate per-tag signal deltas into a combined dimension adjustment dict.
    Deltas are NOT yet clamped here — clamping happens during fusion.
    """
    accumulator: dict[str, float] = {}
    for tag in (tags or []):
        for dim, delta in TAG_SIGNALS.get(tag, {}).items():
            accumulator[dim] = accumulator.get(dim, 0.0) + delta
    return accumulator


def extract_rmp_text_features(review_texts: list[str]) -> dict[str, float]:
    """
    Scan review texts for keyword signals. Averages contributions across all
    reviews so that a professor with more reviews isn't over-amplified.
    """
    if not review_texts:
        return {}

    accumulator: dict[str, float] = {}
    for text in review_texts:
        lower = text.lower()
        for keyword, signals in TEXT_SIGNALS.items():
            if keyword in lower:
                for dim, delta in signals.items():
                    accumulator[dim] = accumulator.get(dim, 0.0) + delta

    n = len(review_texts)
    return {dim: round(val / n, 4) for dim, val in accumulator.items()}


# ---------------------------------------------------------------------------
# Feature extraction — LinkedIn (secondary source)
# ---------------------------------------------------------------------------

def extract_linkedin_features(linkedin: dict[str, Any] | None) -> dict[str, float]:
    """
    Derive career and academic-strength scores from LinkedIn data.

    LinkedIn is *contextual evidence*, not a replacement for student reviews.
    Its signals boost career_relevance, industry_experience, and academic_strength
    at a secondary weight. Direct teaching quality still comes from RMP.
    """
    if linkedin is None:
        return {}

    result: dict[str, float] = {}

    # ── Industry experience ───────────────────────────────────────────────
    has_industry = linkedin.get("has_industry_experience") or False
    years        = linkedin.get("industry_experience_years") or 0.0

    if has_industry:
        # Score saturates at 15 years of industry experience
        industry_score = min(1.0, years / 15.0) if years > 0 else 0.40
        result["industry_experience"] = round(industry_score, 4)
        # Industry experience also drives career relevance upward
        result["career_relevance"]    = round(min(1.0, industry_score + 0.20), 4)

    # ── Academic / research strength ─────────────────────────────────────
    has_research       = linkedin.get("has_research_background") or False
    research_interests = linkedin.get("research_interests")      or []

    if has_research or research_interests:
        academic_score = 0.60
        if len(research_interests) >= 2:
            academic_score += 0.20  # breadth of research interests adds confidence
        result["academic_strength"] = round(min(1.0, academic_score), 4)

    # PhD listed in education → additional academic depth signal
    for edu in (linkedin.get("education") or []):
        degree = (edu.get("degree") or "").upper()
        if "PHD" in degree or "PH.D" in degree:
            existing = result.get("academic_strength", 0.0)
            result["academic_strength"] = round(min(1.0, existing + 0.20), 4)
            break

    # Skills breadth as a mild career-relevance proxy
    skills = linkedin.get("skills") or []
    if len(skills) >= 5:
        existing = result.get("career_relevance", 0.0)
        result["career_relevance"] = round(min(1.0, existing + 0.10), 4)

    return result


# ---------------------------------------------------------------------------
# LinkedIn connection / networking scores
# ---------------------------------------------------------------------------

def compute_connect_opportunity_score(
    career_relevance:    float,
    industry_experience: float,
    academic_strength:   float,
    confidence:          float,
) -> float:
    """
    Estimate how worthwhile it is to connect with this professor on LinkedIn.
    Simple weighted combination — intentionally explainable for the demo.

      career_relevance    0.40  — is their background relevant to the student's field?
      industry_experience 0.30  — do they have real-world experience to share?
      academic_strength   0.20  — are they research-active / well-credentialed?
      confidence          0.10  — how reliable is the underlying data?

    Returns a score in [0, 1].
    """
    score = (
        0.40 * career_relevance
        + 0.30 * industry_experience
        + 0.20 * academic_strength
        + 0.10 * confidence
    )
    return round(max(0.0, min(1.0, score)), 4)


def generate_connect_reasons(features: dict[str, Any]) -> list[str]:
    """
    Return a short list of human-readable reasons why this professor
    is worth connecting with on LinkedIn.  Driven by the same thresholds
    used elsewhere in the engine — keeps the logic consistent.

    Returns an empty list when the connect_opportunity_score is low,
    so the frontend can simply hide the section rather than show an
    empty placeholder.
    """
    t = _THRESHOLDS
    reasons: list[str] = []

    co = features.get("connect_opportunity_score", 0.0)
    cr = features.get("career_relevance_score",    0.0)
    ie = features.get("industry_experience_score", 0.0)
    ac = features.get("academic_strength_score",   0.0)

    # Only generate reasons when the overall connect score clears the bar
    if co < t["high_connect_opportunity"]:
        return reasons

    if ie >= t["high_industry_exp"]:
        reasons.append(
            "Strong industry background — connecting may open doors to referrals or internships"
        )

    if cr >= t["high_career_relevance"]:
        reasons.append(
            "Relevant professional experience aligned with your career direction"
        )

    if ac >= t["high_academic"]:
        reasons.append(
            "Active researcher — good to know for grad school letters or research opportunities"
        )

    # Generic fallback when score is high but individual dims are moderate
    if not reasons and co >= t["high_connect_opportunity"]:
        reasons.append(
            "Active professional profile suggests good networking potential"
        )

    return reasons


# ---------------------------------------------------------------------------
# Feature extraction — Official course data (optional source)
# ---------------------------------------------------------------------------

def extract_official_course_features(official: dict[str, Any] | None) -> dict[str, float]:
    """
    Infer workload, clarity, and fairness signals from syllabus and grading text.
    These supplement RMP-derived scores rather than replacing them.
    """
    if official is None:
        return {}

    result: dict[str, float] = {}

    syllabus  = (official.get("syllabus_text")        or "").lower()
    grading   = (official.get("grading_policy_text")  or "").lower()
    assign    = (official.get("assignment_info")       or "").lower()
    exam_info = (official.get("exam_info")             or "").lower()

    # ── Workload signals ──────────────────────────────────────────────────
    heavy_kws = ("8 assignments", "weekly assignments", "heavy reading", "research project")
    light_kws = ("2 assignments", "flexible deadlines", "group project")

    workload = 0.50  # start neutral
    for kw in heavy_kws:
        if kw in syllabus or kw in assign:
            workload += 0.15
    for kw in light_kws:
        if kw in syllabus or kw in assign:
            workload -= 0.10
    result["workload"] = round(max(0.0, min(1.0, workload)), 4)

    # ── Fairness signals from grading policy ─────────────────────────────
    fairness = 0.50
    if "rubric" in grading or "partial credit" in grading:
        fairness += 0.20  # explicit rubrics signal transparent grading
    if "no curve" in grading or "no curves" in grading:
        fairness -= 0.10  # no curve can hurt borderline students
    if "open" in exam_info:
        fairness += 0.10  # open-note/open-book exams lower stakes
    result["fairness"] = round(max(0.0, min(1.0, fairness)), 4)

    # ── Difficulty adjustment from exam format ────────────────────────────
    if "closed-book" in exam_info and "two" in exam_info:
        result["difficulty_adjustment"] = +0.10  # applied during fusion

    return result


# ---------------------------------------------------------------------------
# Confidence, coverage, controversy, and match-confidence scores
# ---------------------------------------------------------------------------

def compute_source_coverage_score(sources_available: list[str]) -> float:
    """
    Weighted score across the three possible data sources.
      RMP      = 0.50  (primary — student-reported teaching quality)
      LinkedIn = 0.30  (secondary — career context)
      Official = 0.20  (optional — course structure)
    """
    weights = {"rmp": 0.50, "linkedin": 0.30, "official": 0.20}
    return round(sum(weights.get(s, 0.0) for s in sources_available), 4)


def compute_confidence_score(num_reviews: int, sources_available: list[str]) -> float:
    """
    Overall confidence in the recommendation.
    Driven by:
      - Student review volume (log-scaled; 200 reviews ≈ full confidence)
      - Source coverage (how many data sources contributed)
    """
    review_conf = min(1.0, math.log1p(num_reviews) / math.log1p(200))
    source_conf = compute_source_coverage_score(sources_available)
    return round(0.60 * review_conf + 0.40 * source_conf, 4)


def compute_controversy_score(rmp: dict[str, Any]) -> float:
    """
    Estimate how polarised student opinions are. Returns [0, 1].

    Two signals:
      1. Both strongly positive and strongly negative words appear in reviews.
      2. Large divergence between overall_rating and would_take_again percentage.
    """
    score = 0.0

    # Signal 1: extreme word co-occurrence in review texts
    combined = " ".join(rmp.get("review_texts") or []).lower()
    has_positive = any(w in combined for w in _POSITIVE_EXTREMES)
    has_negative = any(w in combined for w in _NEGATIVE_EXTREMES)
    if has_positive and has_negative:
        score += 0.50

    # Signal 2: rating vs would_take_again divergence
    overall = rmp.get("overall_rating") or 3.0
    wta     = rmp.get("would_take_again")
    if wta is not None:
        norm_rating = (overall - 1.0) / 4.0
        norm_wta    = wta / 100.0
        score      += abs(norm_rating - norm_wta) * 0.50

    return round(min(1.0, score), 4)


def compute_match_confidence_score(confidence_score: float, match_score: float) -> float:
    """
    How trustworthy is this specific recommendation?
    High when data confidence is high AND the match score is decisive
    (far from the ambiguous midpoint of 0.5).
    """
    decisiveness = abs(match_score - 0.50) * 2.0  # 0 when ambiguous, 1 when decisive
    return round(confidence_score * (0.60 + 0.40 * decisiveness), 4)


# ---------------------------------------------------------------------------
# Multi-source feature fusion  →  UnifiedProfessorFeatureSchema
# ---------------------------------------------------------------------------

def fuse_features(
    rmp:      dict[str, Any],
    linkedin: dict[str, Any] | None,
    official: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Combine all available sources into a UnifiedProfessorFeatureSchema dict.

    Fusion strategy:
      1. RMP base scores provide quality, difficulty, and proxy values for
         clarity, fairness, engagement, and workload.
      2. Tag and text signals are accumulated as adjustments on top of the base.
      3. LinkedIn-derived features fill in career_relevance, industry_experience,
         and academic_strength (secondary weight; does not override RMP).
      4. Official course data applies a light correction layer to workload and
         fairness (40 % blend weight).
      5. All scores are clamped to [0, 1] before output.
    """
    sources_available: list[str] = ["rmp"]
    if linkedin is not None:
        sources_available.append("linkedin")
    if official is not None:
        sources_available.append("official")

    # ── Step 1: RMP base scores ───────────────────────────────────────────
    rmp_norm = normalize_rmp_signals(rmp)
    tag_adj  = extract_rmp_tag_features(rmp.get("tags", []))
    text_adj = extract_rmp_text_features(rmp.get("review_texts", []))

    quality_base    = rmp_norm["quality_score"]
    difficulty_base = rmp_norm["difficulty_score"]
    wta_score       = rmp_norm["wta_score"]

    # Derive initial proxy scores from quality and would_take_again
    quality    = quality_base
    clarity    = quality_base * 0.80 + wta_score * 0.20
    fairness   = quality_base * 0.70 + wta_score * 0.30
    engagement = quality_base * 0.80 + wta_score * 0.20
    workload   = difficulty_base * 0.70   # harder course usually means more work

    # ── Step 2: Accumulate tag and text adjustments ───────────────────────
    all_adj: dict[str, float] = {}
    for source_adj in (tag_adj, text_adj):
        for dim, delta in source_adj.items():
            all_adj[dim] = all_adj.get(dim, 0.0) + delta

    quality         += all_adj.get("quality",    0.0)
    difficulty_base += all_adj.get("difficulty", 0.0)
    clarity         += all_adj.get("clarity",    0.0)
    fairness        += all_adj.get("fairness",   0.0)
    engagement      += all_adj.get("engagement", 0.0)
    workload        += all_adj.get("workload",   0.0)

    # ── Step 3: LinkedIn secondary signals ───────────────────────────────
    li_features  = extract_linkedin_features(linkedin)
    career_rel   = li_features.get("career_relevance",    0.30)  # default: low
    industry_exp = li_features.get("industry_experience", 0.20)
    academic_str = li_features.get("academic_strength",   0.30)

    # A professor with strong industry background tends to use more examples
    if li_features.get("industry_experience", 0) > 0.50:
        engagement = min(1.0, engagement + 0.05)

    # Tag-based career signals (from RMP tags) also contribute
    career_rel  = min(1.0, career_rel  + all_adj.get("career_relevance",    0.0))
    industry_exp = min(1.0, industry_exp + all_adj.get("industry_experience", 0.0))

    # ── Step 4: Official course adjustments (light blend) ────────────────
    official_features = extract_official_course_features(official)
    if "workload" in official_features:
        # Blend official workload at 40 % weight with RMP-derived workload
        workload = 0.60 * workload + 0.40 * official_features["workload"]
    if "fairness" in official_features:
        fairness = 0.70 * fairness + 0.30 * official_features["fairness"]
    if "difficulty_adjustment" in official_features:
        difficulty_base = min(1.0, difficulty_base + official_features["difficulty_adjustment"])

    # ── Step 5: Metadata scores ───────────────────────────────────────────
    num_reviews     = rmp.get("num_reviews") or 0
    confidence      = compute_confidence_score(num_reviews, sources_available)
    controversy     = compute_controversy_score(rmp)
    source_coverage = compute_source_coverage_score(sources_available)

    # ── Step 6: Clamp all dimension scores to [0, 1] ─────────────────────
    def clamp(v: float) -> float:
        return round(max(0.0, min(1.0, v)), 4)

    clamped_career  = clamp(career_rel)
    clamped_industry = clamp(industry_exp)
    clamped_academic = clamp(academic_str)

    # ── Step 7: Connect opportunity score (LinkedIn networking signal) ────
    # Computed after clamping so inputs are already in [0, 1].
    # Only meaningful when LinkedIn data is present; stays low otherwise.
    connect_opportunity = compute_connect_opportunity_score(
        career_relevance    = clamped_career,
        industry_experience = clamped_industry,
        academic_strength   = clamped_academic,
        confidence          = confidence,
    )

    # Pass through the raw LinkedIn profile URL so the frontend can render a link.
    linkedin_profile_url: str | None = (
        linkedin.get("linkedin_profile_url") if linkedin else None
    )

    return {
        "professor_name":             rmp["professor_name"],
        "school":                     rmp.get("school"),
        "department":                 rmp.get("department"),
        "course_code":                rmp.get("course_code"),
        # Teaching dimension scores
        "quality_score":              clamp(quality),
        "difficulty_score":           clamp(difficulty_base),
        "workload_score":             clamp(workload),
        "clarity_score":              clamp(clarity),
        "fairness_score":             clamp(fairness),
        "engagement_score":           clamp(engagement),
        # Career / LinkedIn dimension scores
        "career_relevance_score":     clamped_career,
        "industry_experience_score":  clamped_industry,
        "academic_strength_score":    clamped_academic,
        # LinkedIn networking score
        "connect_opportunity_score":  connect_opportunity,
        # Metadata scores
        "confidence_score":           confidence,
        "controversy_score":          controversy,
        "source_coverage_score":      source_coverage,
        "match_confidence_score":     0.0,   # filled in after scoring
        "sources_available":          sources_available,
        # Pass-through for frontend link rendering
        "linkedin_profile_url":       linkedin_profile_url,
        "feature_sources": {
            "quality":              "rmp",
            "difficulty":           "rmp+official" if official else "rmp",
            "workload":             "rmp+official" if official else "rmp",
            "clarity":              "rmp",
            "fairness":             "rmp+official" if official else "rmp",
            "engagement":           "rmp+linkedin" if linkedin else "rmp",
            "career_relevance":     "linkedin+rmp" if linkedin else "rmp",
            "industry_experience":  "linkedin"      if linkedin else "none",
            "academic_strength":    "linkedin"      if linkedin else "none",
            "connect_opportunity":  "linkedin"      if linkedin else "none",
        },
    }


# ---------------------------------------------------------------------------
# Scoring per ranking mode
# ---------------------------------------------------------------------------

def score_professor(features: dict[str, Any], mode: str) -> float:
    """
    Compute a match_score in [0, 1] for one professor under the given mode.

    Uses RANKING_MODE_WEIGHTS. Negative weights (difficulty, workload for
    gpa_focused) reduce the score when those dimensions are high.

    The raw weighted sum is normalised by the theoretical min/max of the
    weight vector so the output always lands in [0, 1].
    """
    weights = RANKING_MODE_WEIGHTS.get(mode, RANKING_MODE_WEIGHTS["balanced"])

    dim_values = {
        "quality":              features["quality_score"],
        "difficulty":           features["difficulty_score"],
        "workload":             features["workload_score"],
        "clarity":              features["clarity_score"],
        "fairness":             features["fairness_score"],
        "engagement":           features["engagement_score"],
        "career_relevance":     features["career_relevance_score"],
        "industry_experience":  features["industry_experience_score"],
        "academic_strength":    features["academic_strength_score"],
        # Connect opportunity dimension — weighted 0.00 in gpa/learning, 0.10 in career
        "connect_opportunity":  features.get("connect_opportunity_score", 0.0),
    }

    raw = sum(w * dim_values[dim] for dim, w in weights.items())

    # Theoretical min: negative-weight dims at 1.0, positive-weight dims at 0.0
    # Theoretical max: positive-weight dims at 1.0, negative-weight dims at 0.0
    min_possible  = sum(w for w in weights.values() if w < 0)
    max_possible  = sum(w for w in weights.values() if w > 0)
    score_range   = max_possible - min_possible

    if score_range <= 0:
        return 0.50

    normalised = (raw - min_possible) / score_range
    return round(max(0.0, min(1.0, normalised)), 4)


# ---------------------------------------------------------------------------
# Human-readable output generators
# ---------------------------------------------------------------------------

def generate_why_recommended(
    features:    dict[str, Any],
    mode:        str,
    match_score: float,
) -> list[str]:
    """
    Return up to 4 concise, user-facing reasons why this professor is recommended,
    keyed to the top-weighted dimensions for the chosen mode.
    """
    t = _THRESHOLDS
    reasons: list[str] = []

    q  = features["quality_score"]
    cl = features["clarity_score"]
    fa = features["fairness_score"]
    en = features["engagement_score"]
    cr = features["career_relevance_score"]
    ie = features["industry_experience_score"]
    ac = features["academic_strength_score"]
    di = features["difficulty_score"]
    co = features.get("connect_opportunity_score", 0.0)

    if q >= t["high_quality"]:
        reasons.append(f"Highly rated by students (quality score {q:.0%})")

    if cl >= t["high_clarity"] and mode in ("gpa_focused", "balanced", "learning_focused"):
        reasons.append("Known for clear explanations and well-structured lectures")

    if fa >= t["high_fairness"] and mode in ("gpa_focused", "balanced"):
        reasons.append("Students report fair, transparent grading with explicit rubrics")

    if en >= t["high_engagement"] and mode in ("learning_focused", "balanced"):
        reasons.append("Engaging and passionate teaching style keeps students motivated")

    if cr >= t["high_career_relevance"] and mode in ("career_focused", "balanced"):
        reasons.append("Strong industry connections make course material career-relevant")

    if ie >= t["high_industry_exp"] and mode == "career_focused":
        reasons.append(f"Brings real-world industry experience (score {ie:.0%})")

    if ac >= t["high_academic"] and mode in ("learning_focused", "career_focused"):
        reasons.append("Deep research background enriches subject-matter expertise")

    if mode == "learning_focused" and di >= 0.55:
        reasons.append("Intellectually challenging — course promotes deep understanding")

    if mode == "gpa_focused" and di < 0.45 and fa >= 0.60:
        reasons.append("Lower difficulty with fair grading — solid GPA opportunity")

    # LinkedIn / networking reasons — only surface in career-oriented or balanced modes
    if co >= t["high_connect_opportunity"] and mode in ("career_focused", "balanced"):
        reasons.append(
            "Active professional profile suggests good networking potential"
        )

    return reasons[:4]  # cap at 4 items to keep the UI readable


def generate_potential_concerns(features: dict[str, Any], mode: str) -> list[str]:
    """
    Return concern strings for dimension scores that may be problematic
    given the user's implied priorities under the chosen mode.
    """
    t = _THRESHOLDS
    concerns: list[str] = []

    wl   = features["workload_score"]
    di   = features["difficulty_score"]
    fa   = features["fairness_score"]
    conf = features["confidence_score"]
    cr   = features["career_relevance_score"]

    if wl >= t["concern_workload"]:
        concerns.append("High workload reported — plan your schedule accordingly")

    if di >= t["concern_difficulty"] and mode in ("gpa_focused", "balanced"):
        concerns.append("Course rated quite difficult — may impact GPA")

    if fa <= t["concern_fairness"]:
        concerns.append("Some students report unclear or inconsistent grading")

    if conf < 0.40:
        concerns.append("Limited data available — recommendation may be less reliable")

    if mode == "career_focused" and cr < 0.40:
        concerns.append("Low career-relevance signal — limited industry background found")

    return concerns


def generate_mixed_feedback_warning(controversy_score: float) -> str:
    """
    Return a warning string when student reviews are polarised.
    Returns an empty string when feedback is broadly consistent.
    """
    if controversy_score >= 0.70:
        return (
            "Students have very mixed experiences with this professor. "
            "Read individual reviews carefully before enrolling."
        )
    if controversy_score >= _THRESHOLDS["controversy_warning"]:
        return (
            "Some mixed feedback detected. "
            "Check recent reviews to see if concerns are still relevant."
        )
    return ""


# ---------------------------------------------------------------------------
# Build the final FrontendRecommendationOutputSchema record
# ---------------------------------------------------------------------------

def build_recommendation_output(
    features: dict[str, Any],
    mode:     str,
) -> dict[str, Any]:
    """
    Score one professor under the given mode and package the result into
    FrontendRecommendationOutputSchema format.
    """
    match_score = score_professor(features, mode)

    # Shallow copy to avoid mutating the shared features dict
    f = dict(features)
    f["match_confidence_score"] = compute_match_confidence_score(
        f["confidence_score"], match_score
    )

    return {
        "professor_name":           f["professor_name"],
        "course_code":              f.get("course_code"),
        "match_score":              match_score,
        "match_type":               _MATCH_TYPES.get(mode, mode),
        "why_recommended":          generate_why_recommended(f, mode, match_score),
        "potential_concerns":       generate_potential_concerns(f, mode),
        "mixed_feedback_warning":   generate_mixed_feedback_warning(f["controversy_score"]),
        "dimension_scores": {
            "quality":              f["quality_score"],
            "difficulty":           f["difficulty_score"],
            "workload":             f["workload_score"],
            "clarity":              f["clarity_score"],
            "fairness":             f["fairness_score"],
            "engagement":           f["engagement_score"],
            "career_relevance":     f["career_relevance_score"],
            "industry_experience":  f["industry_experience_score"],
            "academic_strength":    f["academic_strength_score"],
            # New: connect opportunity for frontend radar/bar chart
            "connect_opportunity":  f.get("connect_opportunity_score", 0.0),
        },
        "confidence_score":         f["confidence_score"],
        "controversy_score":        f["controversy_score"],
        "source_coverage_score":    f["source_coverage_score"],
        "sources_used":             f["sources_available"],
        # New: LinkedIn fields — frontend renders profile link and connect badge
        "linkedin_profile_url":     f.get("linkedin_profile_url"),
        "connect_reasons":          generate_connect_reasons(f),
    }


# ---------------------------------------------------------------------------
# Preference nudges — personalise scores on top of the mode baseline
# ---------------------------------------------------------------------------

def _apply_preference_nudges(
    output:      dict[str, Any],
    features:    dict[str, Any],
    preferences: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply small match_score adjustments (+/- 0.05) based on explicit user
    preferences. These nudges refine but do not dominate the mode-based ranking.
    """
    score        = output["match_score"]
    workload_tol = preferences.get("workload_tolerance", "medium")

    # Penalise heavy workload for low-tolerance students
    if workload_tol == "low"  and features["workload_score"]  >= 0.65:
        score -= 0.05
    # Reward heavy workload slightly for high-tolerance students who like challenge
    if workload_tol == "high" and features["workload_score"]  >= 0.65:
        score += 0.03

    if preferences.get("prefer_clear_grading") and features["fairness_score"] >= 0.70:
        score += 0.03

    if preferences.get("prefer_engaging_lectures") and features["engagement_score"] >= 0.65:
        score += 0.03

    if preferences.get("career_oriented") and features["career_relevance_score"] >= 0.60:
        score += 0.04

    # Networking nudge: only give a small boost when the student is open to it
    # AND the connect_opportunity score clears the meaningful threshold.
    # open_to_networking=False → no change (networking should not hurt ranking).
    if (
        preferences.get("open_to_networking")
        and features.get("connect_opportunity_score", 0.0) >= _THRESHOLDS["high_connect_opportunity"]
    ):
        score += 0.03   # intentionally small — teaching fit still dominates

    output["match_score"] = round(max(0.0, min(1.0, score)), 4)
    return output


# ---------------------------------------------------------------------------
# Main recommendation pipeline
# ---------------------------------------------------------------------------

def recommend_professors(
    all_features: list[dict[str, Any]],
    preferences:  dict[str, Any],
    mode:         str,
    top_n:        int = 5,
) -> list[dict[str, Any]]:
    """
    Score all professors under the given mode, apply preference nudges,
    and return the top N results sorted by match_score descending.

    Args:
        all_features:  List of UnifiedProfessorFeatureSchema dicts.
        preferences:   UserPreferenceSchema dict for nudge adjustments.
        mode:          One of: gpa_focused | balanced | learning_focused | career_focused.
        top_n:         Maximum number of results to return.

    Returns:
        List of FrontendRecommendationOutputSchema dicts, sorted best-first.
    """
    outputs = []
    for features in all_features:
        output = build_recommendation_output(features, mode)
        output = _apply_preference_nudges(output, features, preferences)
        outputs.append(output)

    outputs.sort(key=lambda x: x["match_score"], reverse=True)
    return outputs[:top_n]


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_recommendation_results(mode: str, results: list[dict[str, Any]]) -> None:
    """Print ranked recommendation results for one mode in a readable format."""
    mode_label = _MATCH_TYPES.get(mode, mode)
    bar        = "─" * 70

    print(f"\n{bar}")
    print(f"  Mode: {mode_label}  ({mode})")
    print(bar)

    for rank, r in enumerate(results, 1):
        name    = r["professor_name"]
        score   = r["match_score"]
        conf    = r["confidence_score"]
        contr   = r["controversy_score"]
        sources = ", ".join(r["sources_used"])
        ds      = r["dimension_scores"]

        print(f"\n  #{rank}  {name}")
        print(f"       Match {score:.0%}  ·  confidence {conf:.0%}  ·  controversy {contr:.0%}")
        print(f"       Sources  : {sources}")
        print(
            f"       quality {ds['quality']:.2f}  "
            f"clarity {ds['clarity']:.2f}  "
            f"fairness {ds['fairness']:.2f}  "
            f"engagement {ds['engagement']:.2f}"
        )
        print(
            f"       difficulty {ds['difficulty']:.2f}  "
            f"workload {ds['workload']:.2f}  "
            f"career {ds['career_relevance']:.2f}  "
            f"industry {ds['industry_experience']:.2f}  "
            f"academic {ds['academic_strength']:.2f}"
        )
        print(f"       connect_opportunity {ds['connect_opportunity']:.2f}")

        if r["why_recommended"]:
            print("       Why:")
            for reason in r["why_recommended"]:
                wrapped = textwrap.fill(reason, width=60, subsequent_indent=" " * 11)
                print(f"         + {wrapped}")

        if r["potential_concerns"]:
            print("       Concerns:")
            for concern in r["potential_concerns"]:
                wrapped = textwrap.fill(concern, width=60, subsequent_indent=" " * 11)
                print(f"         ! {wrapped}")

        if r["mixed_feedback_warning"]:
            wrapped = textwrap.fill(
                r["mixed_feedback_warning"], width=58, subsequent_indent=" " * 11
            )
            print(f"         ⚠  {wrapped}")

        # LinkedIn / networking section — only shown when relevant
        if r.get("connect_reasons"):
            print("       LinkedIn connect:")
            for cr in r["connect_reasons"]:
                wrapped = textwrap.fill(cr, width=60, subsequent_indent=" " * 11)
                print(f"         ★ {wrapped}")
        if r.get("linkedin_profile_url"):
            print(f"         → {r['linkedin_profile_url']}")

    print(f"\n{bar}\n")


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── 1. Build mock raw data ────────────────────────────────────────────
    all_rmp      = make_mock_rmp_data()
    all_linkedin = make_mock_linkedin_data()
    all_official = make_mock_official_data()

    # Two preference profiles for demo:
    #   default_prefs    — learning-oriented student, not seeking networking
    #   career_prefs     — career-oriented student, open to LinkedIn connections
    default_prefs = make_mock_preferences()
    career_prefs  = make_mock_career_preferences()

    # Index by professor_name for fast O(1) lookup during fusion
    li_by_name  = {d["professor_name"]: d for d in all_linkedin}
    off_by_name = {d["professor_name"]: d for d in all_official}

    print("\n" + "=" * 70)
    print("  Professor Recommendation Engine — Demo")
    print("  5 professors  ·  4 ranking modes  ·  mock data")
    print("=" * 70)

    # ── 2. Fuse features from all available sources for each professor ────
    all_features: list[dict[str, Any]] = []
    for rmp in all_rmp:
        name     = rmp["professor_name"]
        linkedin = li_by_name.get(name)   # None if no LinkedIn data
        official = off_by_name.get(name)  # None if no official data (Bob, Emily)
        unified  = fuse_features(rmp, linkedin, official)
        all_features.append(unified)

    # ── 3. GPA / Learning / Balanced modes — default student preferences ──
    for mode in ("gpa_focused", "balanced", "learning_focused"):
        results = recommend_professors(all_features, default_prefs, mode, top_n=3)
        print_recommendation_results(mode, results)

    # ── 4. Career-focused mode — student open to networking ───────────────
    # This demo shows connect_reasons and linkedin_profile_url in ranked output.
    print("\n" + "=" * 70)
    print("  Career-Focused Demo  (open_to_networking = True)")
    print("  Connect reasons and LinkedIn URLs will appear below.")
    print("=" * 70)
    career_results = recommend_professors(
        all_features, career_prefs, "career_focused", top_n=3
    )
    print_recommendation_results("career_focused", career_results)


# ---------------------------------------------------------------------------
# Update summary  (added for LinkedIn / networking support)
# ---------------------------------------------------------------------------
#
# WHAT WAS CHANGED
# ─────────────────────────────────────────────────────────────────────────
# 1. RANKING_MODE_WEIGHTS
#      Added "connect_opportunity" weight to all four modes.
#      career_focused: +0.10 (meaningful signal)
#      balanced:       +0.03 (minor bonus)
#      learning_focused: +0.02 (minimal)
#      gpa_focused:    +0.00 (irrelevant to GPA)
#
# 2. _THRESHOLDS
#      Added "high_connect_opportunity": 0.60 — controls when connect reasons
#      and LinkedIn signals surface in explanations.
#
# 3. make_mock_preferences()   [extended]
#      Added "open_to_networking" field (default False).
#
# 4. make_mock_career_preferences()   [NEW]
#      Returns a career-oriented student profile with open_to_networking=True
#      for the demo section.
#
# 5. compute_connect_opportunity_score()   [NEW]
#      Simple weighted formula:
#        0.40 * career_relevance + 0.30 * industry_experience
#        + 0.20 * academic_strength + 0.10 * confidence
#
# 6. generate_connect_reasons()   [NEW]
#      Generates human-readable LinkedIn connect reason strings from
#      fused feature scores.  Returns [] when score is below threshold
#      so the frontend section stays hidden for low-opportunity professors.
#
# 7. fuse_features()   [extended]
#      Step 6: now clamps career dims before passing to connect formula.
#      Step 7 (new): calls compute_connect_opportunity_score() and
#        extracts linkedin_profile_url from raw LinkedIn data.
#      Return dict: added connect_opportunity_score, linkedin_profile_url,
#        and "connect_opportunity" key in feature_sources.
#
# 8. score_professor()   [extended]
#      Added "connect_opportunity" to dim_values dict so
#      RANKING_MODE_WEIGHTS["connect_opportunity"] is applied correctly.
#
# 9. generate_why_recommended()   [extended]
#      Appends an "active professional profile" reason when
#      connect_opportunity_score >= threshold and mode is career/balanced.
#
# 10. build_recommendation_output()   [extended]
#       dimension_scores now includes "connect_opportunity".
#       Output now includes "linkedin_profile_url" and "connect_reasons".
#
# 11. _apply_preference_nudges()   [extended]
#       When open_to_networking=True AND connect_opportunity >= threshold,
#       adds a +0.03 nudge (capped — teaching fit still dominates).
#
# 12. print_recommendation_results()   [extended]
#       Prints connect_opportunity score, LinkedIn connect reasons (★),
#       and linkedin_profile_url (→) for each result.
#
# 13. if __name__ == "__main__"   [extended]
#       Replaced single preferences dict with two profiles.
#       Added a dedicated career_focused demo block (step 4) that shows
#       open_to_networking=True output with connect_reasons and URLs.
#
# FRONTEND CONSUMPTION GUIDE
# ─────────────────────────────────────────────────────────────────────────
# New fields in FrontendRecommendationOutputSchema:
#
#   linkedin_profile_url  (str | None)
#     → Render as a "View Profile" button/link.  Hide element when None.
#
#   connect_reasons  (list[str])
#     → Show as a "Why connect?" card below the professor card.
#     → Hide the card entirely when the list is empty.
#
#   dimension_scores["connect_opportunity"]  (float 0–1)
#     → Add as a new spoke on the radar chart or an extra bar in the
#       dimension breakdown.  Label it "Network Potential".
#
# No other existing fields were removed or renamed.
