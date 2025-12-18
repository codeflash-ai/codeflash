from enum import Enum

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

# LSP-specific
TOTAL_LOOPING_TIME_LSP = 10.0  # Kept same timing for LSP mode to avoid in increase in performance reporting

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


class Effort:
    @staticmethod
    def get_number_of_optimizer_candidates(effort: str) -> int:
        if effort == EffortLevel.LOW.value:
            return 3
        if effort == EffortLevel.MEDIUM.value:
            return 4
        if effort == EffortLevel.HIGH.value:
            return 5
        msg = f"Invalid effort level: {effort}"
        raise ValueError(msg)

    @staticmethod
    def get_number_of_optimizer_lp_candidates(effort: str) -> int:
        if effort == EffortLevel.LOW.value:
            return 3
        if effort == EffortLevel.MEDIUM.value:
            return 5
        if effort == EffortLevel.HIGH.value:
            return 6
        msg = f"Invalid effort level: {effort}"
        raise ValueError(msg)

    @staticmethod
    def get_number_of_generated_tests(effort: str) -> int:  # noqa: ARG004
        # we don't use effort with generated tests for now
        return 2
