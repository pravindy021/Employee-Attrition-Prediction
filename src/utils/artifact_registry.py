"""Artifact existence and integrity validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ArtifactCheck:
    """One artifact validation result."""

    relative_path: str
    exists: bool
    size_bytes: int | None
    sha256: str | None


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_artifacts(
    root: Path,
    relative_paths: Iterable[str],
) -> list[ArtifactCheck]:
    """Check existence, size, and SHA-256 for project artifacts."""
    results = []
    for relative_path in relative_paths:
        path = root.joinpath(relative_path)
        if not path.is_file():
            results.append(
                ArtifactCheck(
                    relative_path=relative_path,
                    exists=False,
                    size_bytes=None,
                    sha256=None,
                )
            )
            continue
        results.append(
            ArtifactCheck(
                relative_path=relative_path,
                exists=True,
                size_bytes=int(path.stat().st_size),
                sha256=sha256_file(path),
            )
        )
    return results


def write_artifact_report(
    root: Path,
    relative_paths: Iterable[str],
    output_path: Path,
) -> dict:
    """Write and return an artifact integrity report."""
    checks = inspect_artifacts(root, relative_paths)
    report = {
        "all_required_artifacts_present": all(
            check.exists for check in checks
        ),
        "artifacts": [asdict(check) for check in checks],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report
