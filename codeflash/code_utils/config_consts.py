MAX_TEST_RUN_ITERATIONS = 5
INDIVIDUAL_TESTCASE_TIMEOUT = 15
MAX_FUNCTION_TEST_SECONDS = 60
N_CANDIDATES = 5
MIN_IMPROVEMENT_THRESHOLD = 0.05
MAX_TEST_FUNCTION_RUNS = 50
MAX_CUMULATIVE_TEST_RUNTIME_NANOSECONDS = 100e6  # 100ms
N_TESTS_TO_GENERATE = 2
TOTAL_LOOPING_TIME = 10.0  # 10 second candidate benchmarking budget
COVERAGE_THRESHOLD = 60.0
MIN_TESTCASE_PASSED_THRESHOLD = 6
REPEAT_OPTIMIZATION_PROBABILITY = 0.1
DEFAULT_IMPORTANCE_THRESHOLD = 0.001
N_CANDIDATES_LP = 6

# LSP-specific
N_CANDIDATES_LSP = 3
N_TESTS_TO_GENERATE_LSP = 2
TOTAL_LOOPING_TIME_LSP = 10.0  # Kept same timing for LSP mode to avoid in increase in performance reporting
N_CANDIDATES_LP_LSP = 3


def get_n_candidates() -> int:
    from codeflash.lsp.helpers import is_LSP_enabled

    return N_CANDIDATES_LSP if is_LSP_enabled() else N_CANDIDATES


def get_n_candidates_lp() -> int:
    from codeflash.lsp.helpers import is_LSP_enabled

    return N_CANDIDATES_LP_LSP if is_LSP_enabled() else N_CANDIDATES_LP


def get_n_tests_to_generate() -> int:
    from codeflash.lsp.helpers import is_LSP_enabled

    return N_TESTS_TO_GENERATE_LSP if is_LSP_enabled() else N_TESTS_TO_GENERATE


def get_total_looping_time() -> float:
    from codeflash.lsp.helpers import is_LSP_enabled

    return TOTAL_LOOPING_TIME_LSP if is_LSP_enabled() else TOTAL_LOOPING_TIME
