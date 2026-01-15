from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeflash.agent.memory import Memory
from codeflash.agent.tools.base_tool import supported_tools
from codeflash.cli_cmds.cli import parse_args, process_pyproject_config
from codeflash.code_utils.code_extractor import get_opt_review_metrics
from codeflash.either import is_successful
from codeflash.optimization.optimizer import Optimizer

if TYPE_CHECKING:
    from codeflash.optimization.function_optimizer import FunctionOptimizer


MAX_AGENT_CALLS_PER_OPTIMIZATION = 10


@dataclass
class InitialAgentContext:
    function_optimizer: FunctionOptimizer
    code_context: str
    function_references: str


def build_initial_context(optimizer: Optimizer) -> InitialAgentContext | None:
    optimizable_funcs, count, _ = optimizer.get_optimizable_functions(
        must_return_a_value=False
    )  # no need for the function to return a value for agent, agent should be smart enough to evaluate the behavior of the function without a return statement

    if count == 0:
        return None

    fto = optimizable_funcs.popitem()[1][0]

    module_prep_result = optimizer.prepare_module_for_optimization(fto.file_path)
    if not module_prep_result:
        return None

    validated_original_code, original_module_ast = module_prep_result

    function_optimizer = optimizer.create_function_optimizer(
        fto,
        function_to_optimize_source_code=validated_original_code[fto.file_path].source_code,
        original_module_ast=original_module_ast,
        original_module_path=fto.file_path,
        function_to_tests={},
    )

    optimizer.current_function_optimizer = function_optimizer
    if not function_optimizer:
        return None

    ctx_result = function_optimizer.get_code_optimization_context()
    if not is_successful(ctx_result):
        return None
    ctx = ctx_result.unwrap()

    function_references = get_opt_review_metrics(
        function_optimizer.function_to_optimize_source_code,
        function_optimizer.function_to_optimize.file_path,
        function_optimizer.function_to_optimize.qualified_name,
        function_optimizer.project_root,
        function_optimizer.test_cfg.tests_root,
    )

    return InitialAgentContext(
        function_optimizer=function_optimizer,
        code_context=ctx.read_writable_code.markdown,
        function_references=function_references,
    )


def main() -> None:
    args = parse_args()

    optimizer = Optimizer(process_pyproject_config(args))
    initial_context = build_initial_context(optimizer)
    if not initial_context:
        # TODO: handle this case
        return

    memory = Memory()

    # file_path = args.file
    # function_name = args.function
    # memory.add_to_context_vars("target_file_path", str(file_path))
    # memory.add_to_context_vars("target_function_name", str(function_name))

    memory.set_messages(
        [
            {
                "role": "user",
                "content": f"""# Rewrite the following function to be more efficient and faster, while maintaining the same behavior.

### code context:
{initial_context.code_context}
### function references:
{initial_context.function_references}

Start by writing a plan for the optimization.""",
            }
        ]
    )

    memory.set_messages(
        [
            {
                "role": "user",
                "content": '# Rewrite the following function to be more efficient and faster, while maintaining the same behavior.\n\n### code context:\n```python:code_to_optimize/bubble_sort.py\ndef sorter(arr):\n    print("codeflash stdout: Sorting list")\n    for i in range(len(arr)):\n        for j in range(len(arr) - 1):\n            if arr[j] > arr[j + 1]:\n                temp = arr[j]\n                arr[j] = arr[j + 1]\n                arr[j + 1] = temp\n    print(f"result: {arr}")\n```\n### function references:\n```python:code_to_optimize/bubble_sort_from_another_file.py\ndef sort_from_another_file(arr):\n    sorted_arr = sorter(arr)\n    return sorted_arr\n```\n```python:temp_test.py\ndef run_and_capture(arr):\n    saved_stdout = sys.stdout\n    try:\n        out = StringIO()\n        sys.stdout = out\n        data = arr.copy()\n        sorter(data)\n        return out.getvalue(), data\n    finally:\n        sys.stdout = saved_stdout\n```\n\n\nStart by writing a plan for the optimization.',
            },
            {
                "role": "assistant",
                "content": '## Optimization Plan\n\n### 1- Understand Code Behavior\n\n- **Current Functionality:**  \n  The `sorter` function implements the bubble sort algorithm for sorting a list `arr` in-place, with debug output before and after sorting.\n    - Before sorting: prints `"codeflash stdout: Sorting list"`\n    - After sorting: prints `"result: ..."`, showing the sorted list.\n    - Sorts `arr` in place and does not return a value.\n- **Usage:**  \n  - Called by `sort_from_another_file`, although its return value is not used (likely functions for side-effects and in-place mutation).\n  - Used in `run_and_capture` with output redirection for test validation.\n- **Performance Bottlenecks:**\n    - The classic bubble sort is O(n²) and always loops over the entire unsorted list for every pass.\n    - `range(len(arr) - 1)` does not shrink on each outer iteration—does not exploit the fact that the end section is already sorted.\n    - No early exit when the list is already sorted or no swaps occur—wastes unnecessary passes.\n    - Swapping via a temporary variable instead of Python\'s tuple swap (minor impact).\n    - Performs redundant `len(arr)` calculations in loop bounds.\n    - Prints entire list after sorting, which may be slow for very large lists (this is part of specified external behavior and can\'t be changed).\n- **External Behavior Constraints:**\n    - Must print exactly the same messages (`stdout` is inspected in tests).\n    - Sorts in-place.\n    - No return value expected (returning would change externally visible behavior).\n\n---\n\n### 2- Begin Optimization\n\n- **Algorithm Improvement:**  \n    - Optimize bubble sort by:\n        - Shortening the range of the inner loop as the sorted suffix grows.\n        - Adding an early exit—if no swaps occur in a pass, terminate early (list is sorted).\n- **Pythonic Improvements:**  \n    - Use tuple swap for readability and slight speedup.\n    - Hoist `len(arr)` to a local variable to avoid repeated computation.\n- **Other Considerations:**  \n    - Ensure print statements (before and after) produce byte-for-byte identical output.\n    - Preserve in-place modification and no return value.\n\n**Concrete Steps:**\n  1. Store `len(arr)` once in a variable.\n  2. Change the inner loop\'s range to shrink with each pass: `range(0, n - i - 1)`.\n  3. Add a `swapped` flag; if no swaps occur on a pass, break early.\n  4. Use tuple swap: `arr[j], arr[j+1] = arr[j+1], arr[j]`.\n  5. Keep print statements unchanged in position and format.\n\n---\n\n### 3- Execute & Verify\n\n- **Testing:**\n    - Use existing test (`run_and_capture`) and other targeted tests, including edge cases (empty list, already-sorted list, reversed list, single-element list, and lists with duplicates).\n    - Compare outputs and final sorted array against original implementation to confirm behavioral equivalence.\n    - Check that no exceptions are raised and that the outputs and side effects are identical.\n\n---\n\n### 4- Benchmarking\n\n- **Performance Measurement:**  \n    - Benchmark sorting on a range of input sizes (especially medium and large).\n    - Compare runtime of original and optimized functions using timeit or a similar timing tool.\n    - Assess and report:\n        - Average/median runtime before and after optimization.\n        - Absolute and relative (percentage) speedup achieved.\n\n---',
            },
            {
                "role": "assistant",
                "content": '<replace_in_file>\n<path>code_to_optimize/bubble_sort.py</path>\n<diff>\n<<<<<<< SEARCH\ndef sorter(arr):\n    print("codeflash stdout: Sorting list")\n    for i in range(len(arr)):\n        for j in range(len(arr) - 1):\n            if arr[j] > arr[j + 1]:\n                temp = arr[j]\n                arr[j] = arr[j + 1]\n                arr[j + 1] = temp\n    print(f"result: {arr}")\n=======\ndef sorter(arr):\n    print("codeflash stdout: Sorting list")\n    n = len(arr)\n    for i in range(n):\n        swapped = False\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n                swapped = True\n        if not swapped:\n            break\n    print(f"result: {arr}")\n>>>>>>> REPLACE\n</diff>\n</replace_in_file>',
            },
            {"role": "user", "content": 'replace_in_file: {"status": "OK"}'},
            {
                "role": "assistant",
                "content": '<execute_code>\n<language>python</language>\n<code>\nimport sys\nfrom io import StringIO\nfrom code_to_optimize.bubble_sort import sorter\n\ndef run_and_capture(arr):\n    saved_stdout = sys.stdout\n    try:\n        out = StringIO()\n        sys.stdout = out\n        data = arr.copy()\n        sorter(data)\n        return out.getvalue(), data\n    finally:\n        sys.stdout = saved_stdout\n\n# Test cases for behavioral equivalence\ntest_vectors = [\n    [],\n    [1],\n    [2, 1],\n    [1, 2, 3],\n    [3, 2, 1],\n    [5, 3, 8, 4, 2],\n    [5, 5, 5, 5],\n    [3, 1, 2, 3, 2, 1],\n    list(range(100, 0, -1)),\n]\n\nfor vec in test_vectors:\n    out, sorted_vec = run_and_capture(vec)\n    assert "codeflash stdout: Sorting list" in out\n    assert "result: " in out\n    # result: [sorted version]\n    result_line = out.splitlines()[-1]\n    # safely eval the list (since printed via str)\n    import ast\n    printed_list = ast.literal_eval(result_line.split(\': \', 1)[1])\n    assert printed_list == sorted(vec)\n    assert sorted_vec == sorted(vec)\nprint("All behavioral tests passed for optimized bubble sort.")\n</code>\n</execute_code>',
            },
            {
                "role": "user",
                "content": 'execute_code: {"status": "OK", "stdout": "All behavioral tests passed for optimized bubble sort."}',
            },
        ]
    )
    while memory.api_calls_counter < MAX_AGENT_CALLS_PER_OPTIMIZATION:
        ai_service_client = initial_context.function_optimizer.aiservice_client
        time_start = time.perf_counter()
        response = ai_service_client.call_agent(
            trace_id=memory.trace_id,
            context_vars=memory.get_context_vars(),
            available_tools=supported_tools,
            messages=memory.get_messages(),
        )
        memory.api_calls_counter += 1
        new_messages = response.get("messages", [])
        memory.set_messages(new_messages)

        time_end = time.perf_counter()
        print(f"Time taken: {time_end - time_start:.2f} seconds")


if __name__ == "__main__":
    main()
