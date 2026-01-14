from __future__ import annotations

from enum import Enum
from typing import Any, Union

MAX_TEST_RUN_ITERATIONS = 5
OPTIMIZATION_CONTEXT_TOKEN_LIMIT = 16000
TESTGEN_CONTEXT_TOKEN_LIMIT = 16000
INDIVIDUAL_TESTCASE_TIMEOUT = 15
MAX_FUNCTION_TEST_SECONDS = 60
MIN_IMPROVEMENT_THRESHOLD = 0.05
MIN_THROUGHPUT_IMPROVEMENT_THRESHOLD = 0.10  # 10% minimum improvement for async throughput
MAX_TEST_FUNCTION_RUNS = 50
MAX_CUMULATIVE_TEST_RUNTIME_NANOSECONDS = 100e6  # 100ms
TOTAL_LOOPING_TIME = 10.0  # 10 second candidate benchmarking budget
COVERAGE_THRESHOLD = 60.0
MIN_TESTCASE_PASSED_THRESHOLD = 6
REPEAT_OPTIMIZATION_PROBABILITY = 0.1
DEFAULT_IMPORTANCE_THRESHOLD = 0.001

# pytest loop stability
# For now, we use strict thresholds (large windows and low tolerances), since this is still experimental.
STABILITY_WINDOW_SIZE = 0.35  # 35% of total window
STABILITY_CENTER_TOLERANCE = 0.0025  # ±0.25% around median
STABILITY_SPREAD_TOLERANCE = 0.0025  # 0.25% window spread

# Refinement
REFINED_CANDIDATE_RANKING_WEIGHTS = (2, 1)  # (runtime, diff), runtime is more important than diff by a factor of 2

# LSP-specific
TOTAL_LOOPING_TIME_LSP = 10.0  # Kept same timing for LSP mode to avoid in increase in performance reporting

# setting this value to 1 will disable repair if there is at least one correct candidate
MIN_CORRECT_CANDIDATES = 2

try:
    from codeflash.lsp.helpers import is_LSP_enabled

    _IS_LSP_ENABLED = is_LSP_enabled()
except ImportError:
    _IS_LSP_ENABLED = False

TOTAL_LOOPING_TIME_EFFECTIVE = TOTAL_LOOPING_TIME_LSP if _IS_LSP_ENABLED else TOTAL_LOOPING_TIME

MAX_CONTEXT_LEN_REVIEW = 1000


class EffortLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EffortKeys(str, Enum):
    N_OPTIMIZER_CANDIDATES = "N_OPTIMIZER_CANDIDATES"
    N_OPTIMIZER_LP_CANDIDATES = "N_OPTIMIZER_LP_CANDIDATES"
    N_GENERATED_TESTS = "N_GENERATED_TESTS"
    MAX_CODE_REPAIRS_PER_TRACE = "MAX_CODE_REPAIRS_PER_TRACE"
    REPAIR_UNMATCHED_PERCENTAGE_LIMIT = "REPAIR_UNMATCHED_PERCENTAGE_LIMIT"
    TOP_VALID_CANDIDATES_FOR_REFINEMENT = "TOP_VALID_CANDIDATES_FOR_REFINEMENT"
    ADAPTIVE_OPTIMIZATION_THRESHOLD = "ADAPTIVE_OPTIMIZATION_THRESHOLD"
    MAX_ADAPTIVE_OPTIMIZATIONS_PER_TRACE = "MAX_ADAPTIVE_OPTIMIZATIONS_PER_TRACE"


EFFORT_VALUES: dict[str, dict[EffortLevel, Any]] = {
    EffortKeys.N_OPTIMIZER_CANDIDATES.value: {EffortLevel.LOW: 3, EffortLevel.MEDIUM: 5, EffortLevel.HIGH: 6},
    EffortKeys.N_OPTIMIZER_LP_CANDIDATES.value: {EffortLevel.LOW: 4, EffortLevel.MEDIUM: 6, EffortLevel.HIGH: 7},
    # we don't use effort with generated tests for now
    EffortKeys.N_GENERATED_TESTS.value: {EffortLevel.LOW: 2, EffortLevel.MEDIUM: 2, EffortLevel.HIGH: 2},
    # maximum number of repairs we will do for each function (in case the valid candidates is less than MIN_CORRECT_CANDIDATES)
    EffortKeys.MAX_CODE_REPAIRS_PER_TRACE.value: {EffortLevel.LOW: 2, EffortLevel.MEDIUM: 3, EffortLevel.HIGH: 5},
    # if the percentage of unmatched tests is greater than this, we won't fix it (lowering this value makes the repair more stricted)
    # on the low effort we lower the limit to 20% to be more strict (less repairs, less time)
    EffortKeys.REPAIR_UNMATCHED_PERCENTAGE_LIMIT.value: {
        EffortLevel.LOW: 0.2,
        EffortLevel.MEDIUM: 0.3,
        EffortLevel.HIGH: 0.4,
    },
    # Top valid candidates for refinements
    EffortKeys.TOP_VALID_CANDIDATES_FOR_REFINEMENT: {EffortLevel.LOW: 2, EffortLevel.MEDIUM: 3, EffortLevel.HIGH: 4},
    # max number of adaptive optimization calls to make per a single candidates tree
    EffortKeys.ADAPTIVE_OPTIMIZATION_THRESHOLD.value: {EffortLevel.LOW: 0, EffortLevel.MEDIUM: 0, EffortLevel.HIGH: 2},
    # max number of adaptive optimization calls to make per a single trace
    EffortKeys.MAX_ADAPTIVE_OPTIMIZATIONS_PER_TRACE.value: {
        EffortLevel.LOW: 0,
        EffortLevel.MEDIUM: 0,
        EffortLevel.HIGH: 4,
    },
}


def get_effort_value(key: EffortKeys, effort: Union[EffortLevel, str]) -> Any:  # noqa: ANN401
    key_str = key.value

    if isinstance(effort, str):
        try:
            effort = EffortLevel(effort)
        except ValueError:
            msg = f"Invalid effort level: {effort}"
            raise ValueError(msg) from None

    if key_str not in EFFORT_VALUES:
        msg = f"Invalid key: {key_str}"
        raise ValueError(msg)

    return EFFORT_VALUES[key_str][effort]
