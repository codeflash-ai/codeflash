from __future__ import annotations

import re
import subprocess
from enum import IntEnum
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import code_print, console, logger

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.models.models import OptimizedCandidate


class DiffbehaviorReturnCode(IntEnum):
    DID_NOT_RUN = -1
    NO_DIFFERENCES = 0
    COUNTER_EXAMPLES = 1
    ERROR = 2


def run_concolic_correctness(
    function_to_optimize: FunctionToOptimize,
    function_to_optimize_original_worktree_fqn: str,
    candidates: list[OptimizedCandidate],
    worktrees: list[Path],
    worktree_root: Path,
    git_root: Path,
) -> None:
    logger.info("Generating concolic coverage tests for the original code…")
    original_code_coverage_tests = subprocess.run(
        [
            "crosshair",
            "cover",
            "--example_output_format=pytest",
            "--max_uninteresting_iterations=256",
            function_to_optimize_original_worktree_fqn,
        ],
        capture_output=True,
        text=True,
        cwd=worktree_root,
        check=False,
    )
    logger.info(f"Tests generated through concolic coverage:\n{original_code_coverage_tests.stdout}")
    console.rule()
    logger.info("Running concolic behavior correctness checking and coverage generation on optimized code…")
    console.rule()
    diffbehavior_results: dict[str, DiffbehaviorReturnCode] = {}
    for candidate_index, candidate in enumerate(candidates, start=1):
        logger.info(f"Optimization candidate {candidate_index}/{len(candidates)}:")
        code_print(candidate.source_code)
        function_to_optimize_optimized_worktree_fqn = (
            str(
                worktrees[candidate_index].name / function_to_optimize.file_path.relative_to(git_root).with_suffix("")
            ).replace("/", ".")
            + "."
            + function_to_optimize.qualified_name
        )
        result = subprocess.run(
            [
                "crosshair",
                "diffbehavior",
                "--max_uninteresting_iterations=256",
                function_to_optimize_optimized_worktree_fqn,
                function_to_optimize_original_worktree_fqn,
            ],
            capture_output=True,
            text=True,
            cwd=worktree_root,
            check=False,
        )
        optimized_code_tests = subprocess.run(
            [
                "crosshair",
                "cover",
                "--example_output_format=pytest",
                "--max_uninteresting_iterations=256",
                function_to_optimize_optimized_worktree_fqn,
            ],
            capture_output=True,
            text=True,
            cwd=worktree_root,
            check=False,
        )

        if result.returncode == DiffbehaviorReturnCode.ERROR:
            diffbehavior_results[candidate.optimization_id] = DiffbehaviorReturnCode.ERROR
            logger.info("Inconclusive results from concolic behavior correctness checking.")
            logger.warning(f"Error running crosshair diffbehavior{': ' + result.stderr if result.stderr else '.'}")
        elif result.returncode == DiffbehaviorReturnCode.COUNTER_EXAMPLES:
            split_counter_examples = re.split("(Given: )", result.stdout)[1:]
            joined_counter_examples = [
                "".join(map(str, split_counter_examples[i : i + 2])) for i in range(0, len(split_counter_examples), 2)
            ]
            concrete_counter_examples = "".join(
                [elt for elt in joined_counter_examples if not re.search(r" object at 0x[0-9a-fA-F]+", elt)]
            )
            if concrete_counter_examples:
                diffbehavior_results[candidate.optimization_id] = DiffbehaviorReturnCode.COUNTER_EXAMPLES
                logger.info(
                    f"Optimization candidate failed concolic behavior correctness "
                    f"checking:\n{concrete_counter_examples}"
                )
                if result.stdout != concrete_counter_examples:
                    object_id_counter_examples = "".join(
                        [elt for elt in joined_counter_examples if re.search(r" object at 0x[0-9a-fA-F]+", elt)]
                    )
                    logger.warning(f"Counter-examples with object ID found:\n{object_id_counter_examples}")
            else:
                diffbehavior_results[candidate.optimization_id] = DiffbehaviorReturnCode.ERROR
                logger.info("Inconclusive results from concolic behavior correctness checking.")
                console.rule()
                logger.warning(f"Counter-examples with object ID found:\n{result.stdout}")
        elif result.returncode == DiffbehaviorReturnCode.NO_DIFFERENCES:
            diffbehavior_results[candidate.optimization_id] = DiffbehaviorReturnCode.NO_DIFFERENCES
            first_line = "".join([": ", chr(10), result.stdout.split(chr(10), 1)[0]])
            logger.info(
                f"Optimization candidate passed concolic behavior correctness checking"
                f"{first_line if chr(10) in result.stdout else '.'}"
            )
            paths_exhausted = "All paths exhausted, functions are likely the same!\n"
            if result.stdout.endswith(paths_exhausted):
                logger.info(paths_exhausted)
            else:
                logger.warning("Consider increasing the --max_uninteresting_iterations option.")
        else:
            logger.info("Inconclusive results from concolic behavior correctness checking.")
            logger.error("Unknown return code running crosshair diffbehavior.")
        console.rule()
        logger.info(f"Tests generated through concolic coverage:\n{optimized_code_tests.stdout}")
        console.rule()

    def report_concolic_results(
        equal_results: bool, diffbehavior_result: DiffbehaviorReturnCode, success: bool
    ) -> tuple[bool, bool]:
        if diffbehavior_result == DiffbehaviorReturnCode.NO_DIFFERENCES:
            logger.info("Concolic behavior correctness check successful!")
            console.rule()
            if equal_results:
                logger.info("True negative: Concolic behavior correctness check successful and test results matched.")
            else:
                logger.warning(
                    "False negative for concolic testing: Concolic behavior correctness check successful but test "
                    "results did not match."
                )
            console.rule()
        elif diffbehavior_result == DiffbehaviorReturnCode.COUNTER_EXAMPLES:
            logger.warning("Concolic behavior correctness check failed.")
            console.rule()
            if equal_results:
                logger.warning(
                    "False negative for regression testing: Concolic behavior correctness check failed but test "
                    "results matched."
                )
                success = False
                equal_results = False
            else:
                logger.info("True positive: Concolic behavior correctness check failed and test results did not match.")
            console.rule()
        elif diffbehavior_result == DiffbehaviorReturnCode.ERROR:
            logger.warning("Concolic behavior correctness check inconclusive.")
            console.rule()
        return equal_results, success
