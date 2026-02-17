# Configuration

Key configuration constants, effort levels, and thresholds.

## Constants (`code_utils/config_consts.py`)

### Test Execution

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_TEST_RUN_ITERATIONS` | 5 | Maximum test loop iterations |
| `INDIVIDUAL_TESTCASE_TIMEOUT` | 15s | Timeout per individual test case |
| `MAX_FUNCTION_TEST_SECONDS` | 60s | Max total time for function testing |
| `MAX_TEST_FUNCTION_RUNS` | 50 | Max test function executions |
| `MAX_CUMULATIVE_TEST_RUNTIME_NANOSECONDS` | 100ms | Max cumulative test runtime |
| `TOTAL_LOOPING_TIME` | 10s | Candidate benchmarking budget |
| `MIN_TESTCASE_PASSED_THRESHOLD` | 6 | Minimum test cases that must pass |

### Performance Thresholds

| Constant | Value | Description |
|----------|-------|-------------|
| `MIN_IMPROVEMENT_THRESHOLD` | 0.05 (5%) | Minimum speedup to accept a candidate |
| `MIN_THROUGHPUT_IMPROVEMENT_THRESHOLD` | 0.10 (10%) | Minimum async throughput improvement |
| `MIN_CONCURRENCY_IMPROVEMENT_THRESHOLD` | 0.20 (20%) | Minimum concurrency ratio improvement |
| `COVERAGE_THRESHOLD` | 60.0% | Minimum test coverage |

### Stability Thresholds

| Constant | Value | Description |
|----------|-------|-------------|
| `STABILITY_WINDOW_SIZE` | 0.35 | 35% of total iteration window |
| `STABILITY_CENTER_TOLERANCE` | 0.0025 | ±0.25% around median |
| `STABILITY_SPREAD_TOLERANCE` | 0.0025 | 0.25% window spread |

### Context Limits

| Constant | Value | Description |
|----------|-------|-------------|
| `OPTIMIZATION_CONTEXT_TOKEN_LIMIT` | 16000 | Max tokens for optimization context |
| `TESTGEN_CONTEXT_TOKEN_LIMIT` | 16000 | Max tokens for test generation context |
| `MAX_CONTEXT_LEN_REVIEW` | 1000 | Max context length for optimization review |

### Other

| Constant | Value | Description |
|----------|-------|-------------|
| `MIN_CORRECT_CANDIDATES` | 2 | Min correct candidates before skipping repair |
| `REPEAT_OPTIMIZATION_PROBABILITY` | 0.1 | Probability of re-optimizing a function |
| `DEFAULT_IMPORTANCE_THRESHOLD` | 0.001 | Minimum addressable time to consider a function |
| `CONCURRENCY_FACTOR` | 10 | Number of concurrent executions for concurrency benchmark |
| `REFINED_CANDIDATE_RANKING_WEIGHTS` | (2, 1) | (runtime, diff) weights — runtime 2x more important |

## Effort Levels

`EffortLevel` enum: `LOW`, `MEDIUM`, `HIGH`

Effort controls the number of candidates, repairs, and refinements:

| Key | LOW | MEDIUM | HIGH |
|-----|-----|--------|------|
| `N_OPTIMIZER_CANDIDATES` | 3 | 5 | 6 |
| `N_OPTIMIZER_LP_CANDIDATES` | 4 | 6 | 7 |
| `N_GENERATED_TESTS` | 2 | 2 | 2 |
| `MAX_CODE_REPAIRS_PER_TRACE` | 2 | 3 | 5 |
| `REPAIR_UNMATCHED_PERCENTAGE_LIMIT` | 0.2 | 0.3 | 0.4 |
| `TOP_VALID_CANDIDATES_FOR_REFINEMENT` | 2 | 3 | 4 |
| `ADAPTIVE_OPTIMIZATION_THRESHOLD` | 0 | 0 | 2 |
| `MAX_ADAPTIVE_OPTIMIZATIONS_PER_TRACE` | 0 | 0 | 4 |

Use `get_effort_value(EffortKeys.KEY, effort_level)` to retrieve values.

## Project Configuration

Configuration is read from `pyproject.toml` under `[tool.codeflash]`. Key settings are auto-detected by `setup/detector.py`:
- `module-root` — Root of the module to optimize
- `tests-root` — Root of test files
- `test-framework` — pytest, unittest, jest, etc.
- `formatter-cmds` — Code formatting commands
