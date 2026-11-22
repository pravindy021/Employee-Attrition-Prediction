"""Tests for Phase 15 interview documentation completeness."""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase15_documents_exist() -> None:
    names = [
        "PROJECT_INTERVIEW_GUIDE.md",
        "TECHNICAL_INTERVIEW_QA.md",
        "BUSINESS_CASE_STUDY.md",
        "PROJECT_WALKTHROUGH.md",
    ]
    assert all((PROJECT_ROOT / name).is_file() for name in names)


def test_technical_qa_contains_at_least_sixty_questions() -> None:
    text = (
        PROJECT_ROOT / "TECHNICAL_INTERVIEW_QA.md"
    ).read_text(encoding="utf-8")
    questions = re.findall(r"^\*\*(\d+)\.\s+[^*]+\?\*\*", text, re.MULTILINE)
    assert len(set(questions)) >= 60


def test_interview_docs_use_actual_metrics_and_thresholds() -> None:
    interview = (
        PROJECT_ROOT / "PROJECT_INTERVIEW_GUIDE.md"
    ).read_text(encoding="utf-8")
    qa = (
        PROJECT_ROOT / "TECHNICAL_INTERVIEW_QA.md"
    ).read_text(encoding="utf-8")
    combined = interview + qa
    assert "0.5654" in combined
    assert "0.5745" in combined
    assert "0.5455" in combined
    assert "0.21" in combined
    assert "30/60/90" in combined or "30-, 60-, or 90-day" in combined


def test_business_scenarios_cover_required_cases() -> None:
    text = (
        PROJECT_ROOT / "BUSINESS_CASE_STUDY.md"
    ).read_text(encoding="utf-8").lower()
    for phrase in (
        "department",
        "high predicted propensity",
        "high recall",
        "demographic",
        "reasons behind",
        "termination",
    ):
        assert phrase in text


def test_walkthrough_contains_launch_and_pages() -> None:
    text = (
        PROJECT_ROOT / "PROJECT_WALKTHROUGH.md"
    ).read_text(encoding="utf-8")
    assert "python -m streamlit run app.py" in text
    assert "Employee Risk Assessment" in text
    assert "Explainability Center" in text
    assert "Fairness" in text
    assert "Model Performance" in text
