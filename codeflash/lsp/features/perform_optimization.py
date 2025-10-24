from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import code_print
from codeflash.code_utils.git_worktree_utils import create_diff_patch_from_worktree
from codeflash.either import is_successful

if TYPE_CHECKING:
    import threading

    from codeflash.lsp.server import CodeflashLanguageServer


def get_cancelled_reponse() -> dict[str, str]:
    return {"status": "canceled", "message": "Task was canceled"}


def abort_if_cancelled(cancel_event: threading.Event) -> None:
    if cancel_event.is_set():
        raise RuntimeError("cancelled")


def sync_perform_optimization(server: CodeflashLanguageServer, cancel_event: threading.Event, params) -> dict[str, str]:  # noqa
    server.show_message_log(f"Starting optimization for function: {params.functionName}", "Info")
    should_run_experiment, code_context, original_helper_code = server.current_optimization_init_result
    function_optimizer = server.optimizer.current_function_optimizer
    current_function = function_optimizer.function_to_optimize

    code_print(
        code_context.read_writable_code.flat,
        file_name=current_function.file_path,
        function_name=current_function.function_name,
    )
    abort_if_cancelled(cancel_event)

    optimizable_funcs = {current_function.file_path: [current_function]}

    devnull_writer = open(os.devnull, "w")  # noqa
    with contextlib.redirect_stdout(devnull_writer):
        function_to_tests, _num_discovered_tests = server.optimizer.discover_tests(optimizable_funcs)
        function_optimizer.function_to_tests = function_to_tests

    abort_if_cancelled(cancel_event)
    test_setup_result = function_optimizer.generate_and_instrument_tests(
        code_context, should_run_experiment=should_run_experiment
    )
    abort_if_cancelled(cancel_event)
    if not is_successful(test_setup_result):
        return {"functionName": params.functionName, "status": "error", "message": test_setup_result.failure()}
    (
        generated_tests,
        function_to_concolic_tests,
        concolic_test_str,
        optimizations_set,
        generated_test_paths,
        generated_perf_test_paths,
        instrumented_unittests_created_for_function,
        original_conftest_content,
    ) = test_setup_result.unwrap()

    baseline_setup_result = function_optimizer.setup_and_establish_baseline(
        code_context=code_context,
        original_helper_code=original_helper_code,
        function_to_concolic_tests=function_to_concolic_tests,
        generated_test_paths=generated_test_paths,
        generated_perf_test_paths=generated_perf_test_paths,
        instrumented_unittests_created_for_function=instrumented_unittests_created_for_function,
        original_conftest_content=original_conftest_content,
    )

    abort_if_cancelled(cancel_event)
    if not is_successful(baseline_setup_result):
        return {"functionName": params.functionName, "status": "error", "message": baseline_setup_result.failure()}

    (
        function_to_optimize_qualified_name,
        function_to_all_tests,
        original_code_baseline,
        test_functions_to_remove,
        file_path_to_helper_classes,
    ) = baseline_setup_result.unwrap()

    best_optimization = function_optimizer.find_and_process_best_optimization(
        optimizations_set=optimizations_set,
        code_context=code_context,
        original_code_baseline=original_code_baseline,
        original_helper_code=original_helper_code,
        file_path_to_helper_classes=file_path_to_helper_classes,
        function_to_optimize_qualified_name=function_to_optimize_qualified_name,
        function_to_all_tests=function_to_all_tests,
        generated_tests=generated_tests,
        test_functions_to_remove=test_functions_to_remove,
        concolic_test_str=concolic_test_str,
    )

    abort_if_cancelled(cancel_event)
    if not best_optimization:
        server.show_message_log(
            f"No best optimizations found for function {function_to_optimize_qualified_name}", "Warning"
        )
        return {
            "functionName": params.functionName,
            "status": "error",
            "message": f"No best optimizations found for function {function_to_optimize_qualified_name}",
        }
    # generate a patch for the optimization
    relative_file_paths = [code_string.file_path for code_string in code_context.read_writable_code.code_strings]
    speedup = original_code_baseline.runtime / best_optimization.runtime

    patch_path = create_diff_patch_from_worktree(
        server.optimizer.current_worktree, relative_file_paths, function_to_optimize_qualified_name
    )

    abort_if_cancelled(cancel_event)
    if not patch_path:
        return {
            "functionName": params.functionName,
            "status": "error",
            "message": "Failed to create a patch for optimization",
        }

    server.show_message_log(f"Optimization completed for {params.functionName} with {speedup:.2f}x speedup", "Info")

    return {
        "functionName": params.functionName,
        "status": "success",
        "message": "Optimization completed successfully",
        "extra": f"Speedup: {speedup:.2f}x faster",
        "patch_file": str(patch_path),
        "task_id": params.task_id,
        "explanation": best_optimization.explanation_v2,
    }
