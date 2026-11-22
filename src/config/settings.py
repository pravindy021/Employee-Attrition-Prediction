"""Centralized paths and project-level configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectSettings:
    """Resolved project paths and dashboard/model assumptions."""

    root: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    reports_dir: Path
    figures_dir: Path
    models_dir: Path
    app_dir: Path
    random_state: int
    test_size: float
    classification_threshold: float
    low_risk_max: float
    medium_risk_max: float

    @classmethod
    def load(cls, root: Path | None = None) -> "ProjectSettings":
        """Load settings and resolve all paths beneath the project root."""
        project_root = root or Path(__file__).resolve().parents[2]
        config_path = project_root / "configs" / "project.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Central project configuration is missing: {config_path}"
            )
        values: dict[str, Any] = json.loads(
            config_path.read_text(encoding="utf-8")
        )
        return cls(
            root=project_root,
            data_dir=project_root / "data",
            raw_data_dir=project_root / "data" / "raw",
            processed_data_dir=project_root / "data" / "processed",
            reports_dir=project_root / "reports",
            figures_dir=project_root / "reports" / "figures",
            models_dir=project_root / "models",
            app_dir=project_root / "app",
            random_state=int(values["random_state"]),
            test_size=float(values["test_size"]),
            classification_threshold=float(values["classification_threshold"]),
            low_risk_max=float(values["low_risk_max"]),
            medium_risk_max=float(values["medium_risk_max"]),
        )

    def path(self, *parts: str) -> Path:
        """Resolve a project-relative path."""
        return self.root.joinpath(*parts)
