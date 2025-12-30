from enum import StrEnum, auto

MAX_TEST_RUN_ITERATIONS = 5
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

REFINED_CANDIDATE_RANKING_WEIGHTS = (2, 1)  # (runtime, diff), runtime is more important than diff by a factor of 2

# LSP-specific
TOTAL_LOOPING_TIME_LSP = 10.0  # Kept same timing for LSP mode to avoid in increase in performance reporting

try:
    from codeflash.lsp.helpers import is_LSP_enabled

    _IS_LSP_ENABLED = is_LSP_enabled()
except ImportError:
    _IS_LSP_ENABLED = False

TOTAL_LOOPING_TIME_EFFECTIVE = TOTAL_LOOPING_TIME_LSP if _IS_LSP_ENABLED else TOTAL_LOOPING_TIME

MAX_CONTEXT_LEN_REVIEW = 1000


class EffortLevel(StrEnum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()


class EffortKeys(StrEnum):
    N_OPTIMIZER_CANDIDATES = auto()
    N_OPTIMIZER_LP_CANDIDATES = auto()
    N_GENERATED_TESTS = auto()
    MAX_CODE_REPAIRS_PER_TRACE = auto()
    REPAIR_UNMATCHED_PERCENTAGE_LIMIT = auto()
    TOP_VALID_CANDIDATES_FOR_REFINEMENT = auto()


EFFORT_VALUES: dict[str, dict[EffortLevel, any]] = {
    EffortKeys.N_OPTIMIZER_CANDIDATES.value: {EffortLevel.LOW: 3, EffortLevel.MEDIUM: 4, EffortLevel.HIGH: 5},
    EffortKeys.N_OPTIMIZER_LP_CANDIDATES.value: {EffortLevel.LOW: 3, EffortLevel.MEDIUM: 5, EffortLevel.HIGH: 6},
    # we don't use effort with generated tests for now
    EffortKeys.N_GENERATED_TESTS.value: {EffortLevel.LOW: 2, EffortLevel.MEDIUM: 2, EffortLevel.HIGH: 2},
    # maximum number of repairs we will do for each function
    EffortKeys.MAX_CODE_REPAIRS_PER_TRACE.value: {EffortLevel.LOW: 2, EffortLevel.MEDIUM: 4, EffortLevel.HIGH: 5},
    # if the percentage of unmatched tests is greater than this, we won't fix it (lowering this value makes the repair more stricted)
    # on the low effort we lower the limit to 20% to be more strict (less repairs, less time)
    EffortKeys.REPAIR_UNMATCHED_PERCENTAGE_LIMIT.value: {
        EffortLevel.LOW: 0.2,
        EffortLevel.MEDIUM: 0.4,
        EffortLevel.HIGH: 0.5,
    },
    # Top valid candidates for refinements
    EffortKeys.TOP_VALID_CANDIDATES_FOR_REFINEMENT: {EffortLevel.LOW: 2, EffortLevel.MEDIUM: 3, EffortLevel.HIGH: 4},
}


def get_effort_value(key: EffortKeys, effort: EffortLevel) -> any:
    key_str = key.value
    if key_str in EFFORT_VALUES:
        if effort in EFFORT_VALUES[key_str]:
            return EFFORT_VALUES[key_str][effort]
        msg = f"Invalid effort level: {effort}"
        raise ValueError(msg)
    msg = f"Invalid key: {key_str}"
    raise ValueError(msg)
