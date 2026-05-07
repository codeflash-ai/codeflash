from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TestDiff:
    test_name: str
    original_passed: bool
    candidate_passed: bool
    message: str


def compare_test_results(
    original_results_path: Path,
    candidate_results_path: Path,
    project_root: Path | None = None,
    project_classpath: str | None = None,
) -> tuple[bool, list[TestDiff]]:
    original = _load_results(original_results_path)
    candidate = _load_results(candidate_results_path)

    diffs: list[TestDiff] = []

    all_tests = set(original.keys()) | set(candidate.keys())

    for test_name in sorted(all_tests):
        orig = original.get(test_name)
        cand = candidate.get(test_name)

        if orig is None:
            diffs.append(
                TestDiff(
                    test_name=test_name,
                    original_passed=False,
                    candidate_passed=cand or False,
                    message=f"Test {test_name} only present in candidate results",
                )
            )
            continue

        if cand is None:
            diffs.append(
                TestDiff(
                    test_name=test_name,
                    original_passed=orig,
                    candidate_passed=False,
                    message=f"Test {test_name} missing from candidate results",
                )
            )
            continue

        if orig != cand:
            diffs.append(
                TestDiff(
                    test_name=test_name,
                    original_passed=orig,
                    candidate_passed=cand,
                    message=f"Test {test_name}: original {'passed' if orig else 'failed'}, candidate {'passed' if cand else 'failed'}",
                )
            )

    are_equivalent = len(diffs) == 0
    return are_equivalent, diffs


def _load_results(path: Path) -> dict[str, bool]:
    results: dict[str, bool] = {}
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        logger.debug("Could not read results file %s", path)
        return results

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        action = event.get("Action")
        test_name = event.get("Test")
        if test_name is None:
            continue

        if action == "pass":
            results[test_name] = True
        elif action == "fail":
            results[test_name] = False

    return results
