new_code = """```javascript:code_to_optimize/js/code_to_optimize_js/calculator.js
const { sumArray, average, findMax, findMin } = require('./math_helpers');

/**
 * This is a modified comment
 */
function calculateStats(numbers) {
    if (numbers.length === 0) {
        return {
            sum: 0,
            average: 0,
            min: 0,
            max: 0,
            range: 0
        };
    }

    // Single-pass optimization: compute all stats in one loop
    let sum = 0;
    let min = numbers[0];
    let max = numbers[0];

    for (let i = 0, len = numbers.length; i < len; i++) {
        const num = numbers[i];
        sum += num;
        if (num < min) min = num;
        if (num > max) max = num;
    }

    const avg = sum / numbers.length;
    const range = max - min;

    return {
        sum,
        average: avg,
        min,
        max,
        range
    };
}
```
```javascript:code_to_optimize/js/code_to_optimize_js/math_helpers.js
/**
 * Normalize an array of numbers to a 0-1 range.
 * @param numbers - Array of numbers to normalize
 * @returns Normalized array
 */
function findMax(numbers) {
    if (numbers.length === 0) return -Infinity;

    // Optimized implementation - linear scan instead of sorting
    let max = -Infinity;
    for (let i = 0; i < numbers.length; i++) {
        if (numbers[i] > max) {
            max = numbers[i];
        }
    }
    return max;
}

/**
 * Find the minimum value in an array.
 * @param numbers - Array of numbers
 * @returns The minimum value
 */
function findMin(numbers) {
    if (numbers.length === 0) return Infinity;

    // Optimized implementation - linear scan instead of sorting
    let min = Infinity;
    for (let i = 0; i < numbers.length; i++) {
        if (numbers[i] < min) {
            min = numbers[i];
        }
    }
    return min;
}
```
"""

from pathlib import Path
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.registry import get_language_support
from codeflash.models.models import CodeOptimizationContext, CodeStringsMarkdown
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.verification_utils import TestConfig


class Args:
    disable_imports_sorting = True
    formatter_cmds = ["disabled"]

def test_js_replcement() -> None:
    try:
        root_dir = Path(__file__).parent.parent.parent.resolve()

        main_file = (root_dir / "code_to_optimize/js/code_to_optimize_js/calculator.js").resolve()
        helper_file = (root_dir / "code_to_optimize/js/code_to_optimize_js/math_helpers.js").resolve()

        original_main = main_file.read_text("utf-8")
        original_helper = helper_file.read_text("utf-8")

        js_support = get_language_support("javascript")
        functions = js_support.discover_functions(main_file)
        target = None
        for func in functions:
            if func.name == "calculateStats":
                target = func
                break
        assert target is not None
        func = FunctionToOptimize(
            function_name=target.name,
            file_path=target.file_path,
            parents=target.parents,
            starting_line=target.start_line,
            ending_line=target.end_line,
            starting_col=target.start_col,
            ending_col=target.end_col,
            is_async=target.is_async,
            language=target.language,
        )
        test_config = TestConfig(
            tests_root=root_dir / "code_to_optimize/js/code_to_optimize_js/tests",
            tests_project_rootdir=root_dir,
            project_root_path=root_dir,
            test_framework="jest",
            pytest_cmd="jest",
        )
        func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
        result = func_optimizer.get_code_optimization_context()
        from codeflash.either import is_successful
        if not is_successful(result):
            import pytest
            pytest.skip(f"Context extraction not fully implemented for JS: {result.failure() if hasattr(result, 'failure') else result}")
        code_context: CodeOptimizationContext = result.unwrap()


        original_helper_code: dict[Path, str] = {}
        helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
        for helper_function_path in helper_function_paths:
            with helper_function_path.open(encoding="utf8") as f:
                helper_code = f.read()
                original_helper_code[helper_function_path] = helper_code

        func_optimizer.args = Args()
        did_update = func_optimizer.replace_function_and_helpers_with_optimized_code(
            code_context=code_context, optimized_code=CodeStringsMarkdown.parse_markdown_code(new_code), original_helper_code=original_helper_code
        )

        assert did_update, "Expected code to be updated"

        helper_code = helper_file.read_text(encoding="utf-8")
        main_code = main_file.read_text(encoding="utf-8")

        expected_main = """/**
 * Calculator module - demonstrates cross-file function calls.
 * Uses helper functions from math_helpers.js.
 */

const { sumArray, average, findMax, findMin } = require('./math_helpers');


/**
 * This is a modified comment
 */
function calculateStats(numbers) {
    if (numbers.length === 0) {
        return {
            sum: 0,
            average: 0,
            min: 0,
            max: 0,
            range: 0
        };
    }

    // Single-pass optimization: compute all stats in one loop
    let sum = 0;
    let min = numbers[0];
    let max = numbers[0];

    for (let i = 0, len = numbers.length; i < len; i++) {
        const num = numbers[i];
        sum += num;
        if (num < min) min = num;
        if (num > max) max = num;
    }

    const avg = sum / numbers.length;
    const range = max - min;

    return {
        sum,
        average: avg,
        min,
        max,
        range
    };
}

/**
 * Normalize an array of numbers to a 0-1 range.
 * @param numbers - Array of numbers to normalize
 * @returns Normalized array
 */
function normalizeArray(numbers) {
    if (numbers.length === 0) return [];

    const min = findMin(numbers);
    const max = findMax(numbers);
    const range = max - min;

    if (range === 0) {
        return numbers.map(() => 0.5);
    }

    return numbers.map(n => (n - min) / range);
}

/**
 * Calculate the weighted average of values with corresponding weights.
 * @param values - Array of values
 * @param weights - Array of weights (same length as values)
 * @returns The weighted average
 */
function weightedAverage(values, weights) {
    if (values.length === 0 || values.length !== weights.length) {
        return 0;
    }

    let weightedSum = 0;
    for (let i = 0; i < values.length; i++) {
        weightedSum += values[i] * weights[i];
    }

    const totalWeight = sumArray(weights);
    if (totalWeight === 0) return 0;

    return weightedSum / totalWeight;
}

module.exports = {
    calculateStats,
    normalizeArray,
    weightedAverage
};
"""

        expected_helper = """/**
 * Math helper functions - used by other modules.
 * Some implementations are intentionally inefficient for optimization testing.
 */

/**
 * Calculate the sum of an array of numbers.
 * @param numbers - Array of numbers to sum
 * @returns The sum of all numbers
 */
function sumArray(numbers) {
    // Intentionally inefficient - using reduce with spread operator
    let result = 0;
    for (let i = 0; i < numbers.length; i++) {
        result = result + numbers[i];
    }
    return result;
}

/**
 * Calculate the average of an array of numbers.
 * @param numbers - Array of numbers
 * @returns The average value
 */
function average(numbers) {
    if (numbers.length === 0) return 0;
    return sumArray(numbers) / numbers.length;
}

/**
 * Normalize an array of numbers to a 0-1 range.
 * @param numbers - Array of numbers to normalize
 * @returns Normalized array
 */
function findMax(numbers) {
    if (numbers.length === 0) return -Infinity;

    // Optimized implementation - linear scan instead of sorting
    let max = -Infinity;
    for (let i = 0; i < numbers.length; i++) {
        if (numbers[i] > max) {
            max = numbers[i];
        }
    }
    return max;
}

/**
 * Find the minimum value in an array.
 * @param numbers - Array of numbers
 * @returns The minimum value
 */
function findMin(numbers) {
    if (numbers.length === 0) return Infinity;

    // Optimized implementation - linear scan instead of sorting
    let min = Infinity;
    for (let i = 0; i < numbers.length; i++) {
        if (numbers[i] < min) {
            min = numbers[i];
        }
    }
    return min;
}

module.exports = {
    sumArray,
    average,
    findMax,
    findMin
};
"""

        assert main_code == expected_main, f"Main file mismatch.\n\nActual:\n{main_code}\n\nExpected:\n{expected_main}"
        assert helper_code == expected_helper, f"Helper file mismatch.\n\nActual:\n{helper_code}\n\nExpected:\n{expected_helper}"

    finally:
        main_file.write_text(original_main, encoding="utf-8")
        helper_file.write_text(original_helper, encoding="utf-8")
