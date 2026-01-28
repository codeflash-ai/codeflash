from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import exit_with_message
from codeflash.code_utils.git_utils import git_root_dir
from codeflash.models.models import Phase1Output, TestResults
from codeflash.result.create_pr import check_create_pr
from codeflash.result.explanation import Explanation

if TYPE_CHECKING:
    from argparse import Namespace

# Pattern to extract file path from markdown code block: ```python:path/to/file.py
MARKDOWN_FILE_PATH_PATTERN = re.compile(r"```python:([^\n]+)")


def extract_file_path_from_markdown(markdown_code: str) -> str | None:
    """Extract the file path from markdown code block format.

    Format: ```python:path/to/file.py
    """
    match = MARKDOWN_FILE_PATH_PATTERN.search(markdown_code)
    if match:
        return match.group(1).strip()
    return None


def extract_code_from_markdown(markdown_code: str) -> str:
    r"""Extract the code content from markdown code block.

    Removes the ```python:path\n ... ``` wrapper.
    """
    # Remove opening markdown fence with optional path
    code = re.sub(r"^```python(?::[^\n]*)?\n", "", markdown_code)
    # Remove closing fence
    return re.sub(r"\n```$", "", code)


def create_pr(args: Namespace) -> None:
    """Create a PR from previously applied optimizations."""
    results_file = Path(args.results_file)

    if not results_file.exists():
        exit_with_message(f"Results file not found: {results_file}", error_on_exit=True)

    # Load and parse results
    with results_file.open(encoding="utf-8") as f:
        data = json.load(f)

    try:
        output = Phase1Output.model_validate(data)
    except Exception as e:
        exit_with_message(f"Failed to parse results file: {e}", error_on_exit=True)

    # Find the function result
    if len(output.functions) == 0:
        exit_with_message("No functions in results file", error_on_exit=True)

    if len(output.functions) > 1 and not args.function:
        func_names = [f.function_name for f in output.functions]
        exit_with_message(
            f"Multiple functions in results. Specify one with --function: {func_names}", error_on_exit=True
        )

    func_result = output.functions[0]
    if args.function:
        func_result = next((f for f in output.functions if f.function_name == args.function), None)
        if not func_result:
            exit_with_message(f"Function {args.function} not found in results", error_on_exit=True)
        assert func_result is not None  # for type checker - exit_with_message doesn't return

    if not func_result.best_candidate_id:
        exit_with_message("No successful optimization found in results", error_on_exit=True)

    # Get file path - prefer explicit field, fall back to extracting from markdown
    file_path_str = func_result.file_path
    if not file_path_str:
        file_path_str = extract_file_path_from_markdown(func_result.original_source_code)

    if not file_path_str:
        exit_with_message(
            "Could not determine file path from results. Results file may be from an older version of codeflash.",
            error_on_exit=True,
        )
    assert file_path_str is not None  # for type checker - exit_with_message doesn't return

    file_path = Path(file_path_str)
    if not file_path.exists():
        exit_with_message(f"Source file not found: {file_path}", error_on_exit=True)

    # Read current (optimized) file content
    current_content = file_path.read_text(encoding="utf-8")

    # Extract original code (strip markdown)
    original_code = extract_code_from_markdown(func_result.original_source_code)

    # Get the best candidate's explanation
    best_explanation = func_result.best_candidate_explanation
    if not best_explanation:
        # Fall back to the candidate's explanation if the final explanation wasn't captured
        best_candidate = next(
            (c for c in func_result.candidates if c.optimization_id == func_result.best_candidate_id), None
        )
        best_explanation = best_candidate.explanation if best_candidate else "Optimization applied"

    # Build Explanation object for PR creation
    explanation = Explanation(
        raw_explanation_message=best_explanation,
        winning_behavior_test_results=TestResults(),
        winning_benchmarking_test_results=TestResults(),
        original_runtime_ns=func_result.original_runtime_ns or 0,
        best_runtime_ns=func_result.best_runtime_ns or func_result.original_runtime_ns or 0,
        function_name=func_result.function_name,
        file_path=file_path,
    )

    logger.info(f"Creating PR for optimized function: {func_result.function_name}")
    logger.info(f"File: {file_path}")
    if func_result.best_speedup_ratio:
        logger.info(f"Speedup: {func_result.best_speedup_ratio * 100:.1f}%")

    # Call existing PR creation
    check_create_pr(
        original_code={file_path: original_code},
        new_code={file_path: current_content},
        explanation=explanation,
        existing_tests_source=func_result.existing_tests_source or "",
        generated_original_test_source="",
        function_trace_id=func_result.trace_id,
        coverage_message="",
        replay_tests=func_result.replay_tests_source or "",
        concolic_tests=func_result.concolic_tests_source or "",
        optimization_review="",
        root_dir=git_root_dir(),
        git_remote=getattr(args, "git_remote", None),
        precomputed_test_report=func_result.test_report,
        precomputed_loop_count=func_result.loop_count,
    )

    # Cleanup results file after successful PR creation
    results_file.unlink()
    logger.info(f"Cleaned up results file: {results_file}")
