"""Tests for Phase 14 research and executive documentation consistency."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase14_documents_exist() -> None:
    assert (PROJECT_ROOT / "EMPLOYEE_ATTRITION_RESEARCH_PAPER.md").is_file()
    assert (PROJECT_ROOT / "EXECUTIVE_SUMMARY.md").is_file()


def test_research_paper_uses_verified_held_out_metrics() -> None:
    text = (
        PROJECT_ROOT / "EMPLOYEE_ATTRITION_RESEARCH_PAPER.md"
    ).read_text(encoding="utf-8")
    assert "PR-AUC 0.5654" in text
    assert "recall 0.5745" in text
    assert "F1 0.5455" in text
    assert "threshold is 0.21" in text
    assert "Isotonic" in text
    assert "SHAP does not explain the isotonic" in text


def test_executive_summary_uses_verified_metrics_and_disclaimers() -> None:
    text = (
        PROJECT_ROOT / "EXECUTIVE_SUMMARY.md"
    ).read_text(encoding="utf-8").lower()
    assert "0.5654" in text
    assert "0.5745" in text
    assert "0.5455" in text
    assert "30, 60" in text
    assert "associations" in text
    assert "causal" in text


def test_risk_boundaries_are_consistent_in_phase14_docs() -> None:
    paper = (
        PROJECT_ROOT / "EMPLOYEE_ATTRITION_RESEARCH_PAPER.md"
    ).read_text(encoding="utf-8")
    executive = (
        PROJECT_ROOT / "EXECUTIVE_SUMMARY.md"
    ).read_text(encoding="utf-8")
    assert "below 30%" in paper
    assert "30% through 60%" in paper
    assert "above 60%" in paper
    assert "below 30%" in executive
    assert "30% through 60%" in executive
    assert "above 60%" in executive
