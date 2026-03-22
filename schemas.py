###########################################################
# FINAL HACKATHON RECOMMENDATION SYSTEM SCHEMA (v2)
# Multi-source professor recommendation engine
#
# Sources:
# - Rate My Professors (primary)
# - LinkedIn (secondary)
# - Official syllabus/course info (optional)
#
# Added:
# - LinkedIn connection support
# - networking signal scoring
###########################################################

from __future__ import annotations
from typing import Any


###########################################################
# A. RAW CRAWLER INPUT SCHEMA — RATE MY PROFESSORS
###########################################################

RMPRawSchema = {
    "professor_name": str,
    "school": str,
    "department": str,
    "course_code": str | None,

    "overall_rating": float,
    "difficulty": float,
    "would_take_again": float | None,
    "num_reviews": int,

    "tags": list[str],
    "review_texts": list[str],
    "review_dates": list[str],
}


###########################################################
# B. LINKEDIN INPUT SCHEMA
###########################################################

LinkedInRawSchema = {
    "professor_name": str,
    "school": str | None,
    "department": str | None,
    "course_code": str | None,

    "headline": str | None,
    "current_title": str | None,
    "current_company_or_university": str | None,
    "about": str | None,

    "past_experiences": list[dict],       # [{"title": str, "company": str, "years": float}]
    "education": list[dict],              # [{"degree": str, "field": str, "school": str}]
    "skills": list[str],
    "certifications": list[str],
    "research_interests": list[str],

    "industry_experience_years": float | None,
    "has_industry_experience": bool | None,
    "has_research_background": bool | None,

    "profile_last_updated": str | None,   # ISO date string
    "linkedin_profile_url": str | None,
}


###########################################################
# C. OPTIONAL OFFICIAL COURSE / SYLLABUS INPUT SCHEMA
###########################################################

OfficialCourseRawSchema = {
    "school": str,
    "department": str,
    "course_code": str,
    "course_title": str | None,
    "professor_name": str | None,

    "section": str | None,
    "modality": str | None,              # "in-person" | "hybrid" | "online"
    "schedule": str | None,

    "syllabus_text": str | None,
    "grading_policy_text": str | None,
    "assignment_info": str | None,
    "exam_info": str | None,
}


###########################################################
# D. UNIFIED PROFESSOR FEATURE SCHEMA (ALGORITHM INPUT)
###########################################################

UnifiedProfessorFeatureSchema = {

    # Identity
    "professor_name": str,
    "school": str | None,
    "department": str | None,
    "course_code": str | None,

    # ── Teaching signals (normalized 0–1) ──────────────────────────────────
    "quality_score": float,
    "difficulty_score": float,
    "workload_score": float,
    "clarity_score": float,
    "fairness_score": float,
    "engagement_score": float,

    # ── Career / background signals (LinkedIn-driven) ───────────────────────
    "career_relevance_score": float,
    "industry_experience_score": float,
    "academic_strength_score": float,

    # ── Networking signal ───────────────────────────────────────────────────
    "connect_opportunity_score": float,

    # ── Reliability / fusion metadata ───────────────────────────────────────
    "confidence_score": float,
    "controversy_score": float,
    "source_coverage_score": float,
    "match_confidence_score": float,     # filled in after ranking, not during fusion

    # ── Source transparency ─────────────────────────────────────────────────
    "sources_available": list[str],      # e.g. ["rmp", "linkedin", "official"]
    "feature_sources": dict,             # e.g. {"quality": "rmp", "career_relevance": "linkedin+rmp"}
}


###########################################################
# E. USER PREFERENCE INPUT SCHEMA
###########################################################

UserPreferenceSchema = {
    "student_goal": str,
    # "gpa_focused" | "balanced" | "learning_focused" | "career_focused"

    "workload_tolerance": str,
    # "low" | "medium" | "high"

    "prefer_clear_grading": bool,
    "prefer_engaging_lectures": bool,
    "career_oriented": bool,

    "open_to_networking": bool,          # enables connect_opportunity score nudge
}


###########################################################
# F. FRONTEND RECOMMENDATION OUTPUT SCHEMA
###########################################################

FrontendRecommendationOutputSchema = {

    # ── Identity ────────────────────────────────────────────────────────────
    "professor_name": str,
    "course_code": str | None,

    # ── Ranking output ──────────────────────────────────────────────────────
    "match_score": float,
    "match_type": str,                   # human-readable label, e.g. "Career Booster"

    # ── Explanation output ──────────────────────────────────────────────────
    "why_recommended": list[str],
    "potential_concerns": list[str],
    "mixed_feedback_warning": str,       # empty string when no warning

    # ── Dimension breakdown (for radar / bar charts) ────────────────────────
    "dimension_scores": {
        "quality": float,
        "difficulty": float,
        "workload": float,
        "clarity": float,
        "fairness": float,
        "engagement": float,

        "career_relevance": float,
        "industry_experience": float,
        "academic_strength": float,

        "connect_opportunity": float,    # NEW
    },

    # ── Reliability metadata ────────────────────────────────────────────────
    "confidence_score": float,
    "controversy_score": float,
    "source_coverage_score": float,

    # ── Source transparency ─────────────────────────────────────────────────
    "sources_used": list[str],

    # ── LinkedIn connection support ─────────────────────────────────────────
    "linkedin_profile_url": str | None,  # NEW — None when not available
    "connect_reasons": list[str],        # NEW — empty list when below threshold
}


###########################################################
# G. MVP SIMPLIFIED FEATURE SCHEMA (FAST FALLBACK VERSION)
###########################################################

UnifiedProfessorFeatureSchema_MVP = {
    "professor_name": str,
    "course_code": str | None,

    "quality_score": float,
    "difficulty_score": float,
    "workload_score": float,
    "clarity_score": float,
    "fairness_score": float,

    "career_relevance_score": float,
    "connect_opportunity_score": float,  # NEW

    "confidence_score": float,
    "controversy_score": float,
    "source_coverage_score": float,

    "sources_available": list[str],
}


###########################################################
# H. SOURCE ROLE DEFINITIONS
###########################################################

SourceRoles: dict[str, str] = {
    "rmp":
        "primary source for teaching quality, workload, clarity, fairness, difficulty",

    "linkedin":
        "secondary source for career relevance, industry experience, "
        "academic strength, networking potential",

    "official":
        "optional structured source for grading structure, assignments, exams",
}
