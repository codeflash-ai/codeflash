from __future__ import annotations

from pathlib import Path


def generate_candidates(source_code_path: Path) -> list[str]:
    """Generate all the possible candidates for coverage data based on the source code path."""
    candidates = [source_code_path.name]
    current_path = source_code_path.parent

    while current_path != current_path.parent:
        candidate_path = str(Path(current_path.name) / candidates[-1])
        candidates.append(candidate_path)
        current_path = current_path.parent

    return candidates
