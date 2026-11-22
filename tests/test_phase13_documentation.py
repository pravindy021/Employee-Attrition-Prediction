"""Tests for Phase 13 documentation completeness."""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_documentation_files_exist() -> None:
    required = [
        "README.md",
        "PROJECT_OVERVIEW.md",
        "METHODOLOGY.md",
        "DATA_DICTIONARY.md",
        "MODEL_CARD.md",
        "RESPONSIBLE_AI.md",
        "TESTING_STRATEGY.md",
        "DEPLOYMENT_GUIDE.md",
    ]
    assert all((PROJECT_ROOT / name).is_file() for name in required)


def test_readme_has_verified_threshold_definitions() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Classification threshold: **0.21**" in readme
    assert "Low risk: probability below 30%" in readme
    assert "Medium risk: 30% through 60%" in readme
    assert "High risk: above 60%" in readme


def test_readme_does_not_claim_public_deployment() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "no public url is claimed" in readme


def test_readme_architecture_and_eight_pages_are_documented() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "```mermaid" in readme
    for page in (
        "Executive Overview",
        "Employee Risk Assessment",
        "Department Analytics",
        "Workforce Insights",
        "Explainability Center",
        "Fairness & Responsible AI",
        "What-If Scenario Lab",
        "Model Performance",
    ):
        assert page in readme
