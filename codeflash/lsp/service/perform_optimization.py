import contextlib
import os
from pathlib import Path

from codeflash.code_utils.git_worktree_utils import create_diff_patch_from_worktree
from codeflash.either import is_successful
from codeflash.lsp.server import CodeflashLanguageServer


# ruff: noqa: PLR0911, ANN001
def sync_perform_optimization(server: CodeflashLanguageServer, params) -> dict[str, str]:
    current_function = server.optimizer.current_function_being_optimized
    if not current_function:
        server.show_message_log(f"No current function being optimized for {params.functionName}", "Error")
        return {
            "functionName": params.functionName,
            "status": "error",
            "message": "No function currently being optimized",
        }

    module_prep_result = server.optimizer.prepare_module_for_optimization(current_function.file_path)
    if not module_prep_result:
        return {
            "functionName": params.functionName,
            "status": "error",
            "message": "Failed to prepare module for optimization",
        }

    validated_original_code, original_module_ast = module_prep_result

    function_optimizer = server.optimizer.create_function_optimizer(
        current_function,
        function_to_optimize_source_code=validated_original_code[current_function.file_path].source_code,
        original_module_ast=original_module_ast,
        original_module_path=current_function.file_path,
        function_to_tests={},
    )
    server.optimizer.current_function_optimizer = function_optimizer
    if not function_optimizer:
        return {"functionName": params.functionName, "status": "error", "message": "No function optimizer found"}

    initialization_result = function_optimizer.can_be_optimized()
    if not is_successful(initialization_result):
        return {"functionName": params.functionName, "status": "error", "message": initialization_result.failure()}

    should_run_experiment, code_context, original_helper_code = initialization_result.unwrap()

    # All the synchronous, potentially blocking calls
    optimizable_funcs = {current_function.file_path: [current_function]}
    devnull_writer = open(os.devnull, "w")  # noqa
    with contextlib.redirect_stdout(devnull_writer):
        function_to_tests, num_discovered_tests = server.optimizer.discover_tests(optimizable_funcs)
        function_optimizer.function_to_tests = function_to_tests

    test_setup_result = function_optimizer.generate_and_instrument_tests(
        code_context, should_run_experiment=should_run_experiment
    )
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

    if not best_optimization:
        server.show_message_log(
            f"No best optimizations found for function {function_to_optimize_qualified_name}", "Warning"
        )
        return {
            "functionName": params.functionName,
            "status": "error",
            "message": f"No best optimizations found for function {function_to_optimize_qualified_name}",
        }

    relative_file_paths = [code_string.file_path for code_string in code_context.read_writable_code.code_strings]
    speedup = original_code_baseline.runtime / best_optimization.runtime
    original_args, _ = server.optimizer.original_args_and_test_cfg
    relative_file_path = current_function.file_path.relative_to(server.optimizer.current_worktree)
    original_file_path = Path(original_args.project_root / relative_file_path).resolve()

    metadata = create_diff_patch_from_worktree(
        server.optimizer.current_worktree,
        relative_file_paths,
        metadata_input={
            "fto_name": function_to_optimize_qualified_name,
            "explanation": best_optimization.explanation_v2,
            "file_path": str(original_file_path),
            "speedup": speedup,
        },
    )
    server.show_message_log(f"Optimization completed for {params.functionName} with {speedup:.2f}x speedup", "Info")

    return {
        "functionName": params.functionName,
        "status": "success",
        "message": "Optimization completed successfully",
        "extra": f"Speedup: {speedup:.2f}x faster",
        "patch_file": metadata["patch_path"],
        "patch_id": metadata["id"],
        "explanation": best_optimization.explanation_v2,
    }
