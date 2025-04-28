from codeflash.optimization.function_optimizer import FunctionOptimizer


def test_code_replacement_pr():
    fn_opt = FunctionOptimizer()
    code_context = ''
    best_optimization = ''
    explanation = ''
    existing_tests_source_for = ''
    original_helper_code = ''
    function_to_all_tests = ''
    fn_opt.replace_function_and_helpers_with_optimized_code(
        code_context=code_context, optimized_code=best_optimization.candidate.source_code
    )

    new_code, new_helper_code = fn_opt.reformat_code_and_helpers(
        code_context.helper_functions, explanation.file_path, fn_opt.function_to_optimize_source_code
    )

    existing_tests = existing_tests_source_for(
        fn_opt.function_to_optimize.qualified_name_with_modules_from_root(fn_opt.project_root),
        function_to_all_tests,
        tests_root=fn_opt.test_cfg.tests_root,
    )

    original_code_combined = original_helper_code.copy()
    original_code_combined[explanation.file_path] = fn_opt.function_to_optimize_source_code
    new_code_combined = new_helper_code.copy()
    new_code_combined[explanation.file_path] = new_code

