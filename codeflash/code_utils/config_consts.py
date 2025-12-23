MAX_TEST_RUN_ITERATIONS = 5
INDIVIDUAL_TESTCASE_TIMEOUT = 15
MAX_FUNCTION_TEST_SECONDS = 60
N_CANDIDATES = 5
MIN_IMPROVEMENT_THRESHOLD = 0.05
MIN_THROUGHPUT_IMPROVEMENT_THRESHOLD = 0.10  # 10% minimum improvement for async throughput
MAX_TEST_FUNCTION_RUNS = 50
MAX_CUMULATIVE_TEST_RUNTIME_NANOSECONDS = 100e6  # 100ms
N_TESTS_TO_GENERATE = 2
TOTAL_LOOPING_TIME = 10.0  # 10 second candidate benchmarking budget
COVERAGE_THRESHOLD = 60.0
MIN_TESTCASE_PASSED_THRESHOLD = 6
REPEAT_OPTIMIZATION_PROBABILITY = 0.1
DEFAULT_IMPORTANCE_THRESHOLD = 0.001
N_CANDIDATES_LP = 6

# Refinement
REFINE_ALL_THRESHOLD = 2  # when valid optimizations count is 2 or less, refine all optimizations
REFINED_CANDIDATE_RANKING_WEIGHTS = (2, 1)  # (runtime, diff), runtime is more important than diff by a factor of 2
TOP_N_REFINEMENTS = 0.45  # top 45% of valid optimizations (based on the weighted score) are refined

# LSP-specific
N_CANDIDATES_LSP = 3
N_TESTS_TO_GENERATE_LSP = 2
TOTAL_LOOPING_TIME_LSP = 10.0  # Kept same timing for LSP mode to avoid in increase in performance reporting
N_CANDIDATES_LP_LSP = 3

# Code repair
REPAIR_UNMATCHED_PERCENTAGE_LIMIT = 0.4  # if the percentage of unmatched tests is greater than this, we won't fix it (lowering this value makes the repair more stricted)
MAX_REPAIRS_PER_TRACE = 4  # maximum number of repairs we will do for each function

MAX_N_CANDIDATES = 5
MAX_N_CANDIDATES_LP = 6

# Multi-model diversity configuration
# Each tuple is (model_name, num_calls) where each call returns 1 candidate
# Standard mode: 3 GPT-4.1 + 2 Claude Sonnet = 5 candidates
MODEL_DISTRIBUTION: list[tuple[str, int]] = [
    ("gpt-4.1", 3),
    ("claude-sonnet-4-5", 2),
]

# LSP mode: fewer candidates for faster response
MODEL_DISTRIBUTION_LSP: list[tuple[str, int]] = [
    ("gpt-4.1", 2),
    ("claude-sonnet-4-5", 1),
]

# Line profiler mode: 6 candidates total
MODEL_DISTRIBUTION_LP: list[tuple[str, int]] = [
    ("gpt-4.1", 4),
    ("claude-sonnet-4-5", 2),
]

# Line profiler LSP mode
MODEL_DISTRIBUTION_LP_LSP: list[tuple[str, int]] = [
    ("gpt-4.1", 2),
    ("claude-sonnet-4-5", 1),
]

try:
    from codeflash.lsp.helpers import is_LSP_enabled

    _IS_LSP_ENABLED = is_LSP_enabled()
except ImportError:
    _IS_LSP_ENABLED = False

N_CANDIDATES_EFFECTIVE = min(N_CANDIDATES_LSP if _IS_LSP_ENABLED else N_CANDIDATES, MAX_N_CANDIDATES)
N_CANDIDATES_LP_EFFECTIVE = min(N_CANDIDATES_LP_LSP if _IS_LSP_ENABLED else N_CANDIDATES_LP, MAX_N_CANDIDATES_LP)
N_TESTS_TO_GENERATE_EFFECTIVE = N_TESTS_TO_GENERATE_LSP if _IS_LSP_ENABLED else N_TESTS_TO_GENERATE
TOTAL_LOOPING_TIME_EFFECTIVE = TOTAL_LOOPING_TIME_LSP if _IS_LSP_ENABLED else TOTAL_LOOPING_TIME
MODEL_DISTRIBUTION_EFFECTIVE = MODEL_DISTRIBUTION_LSP if _IS_LSP_ENABLED else MODEL_DISTRIBUTION
MODEL_DISTRIBUTION_LP_EFFECTIVE = MODEL_DISTRIBUTION_LP_LSP if _IS_LSP_ENABLED else MODEL_DISTRIBUTION_LP

MAX_CONTEXT_LEN_REVIEW = 1000
