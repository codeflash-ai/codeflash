# Verification

How codeflash verifies candidate correctness and measures performance.

## Test Execution Architecture

Tests are executed in a **subprocess** to isolate the test environment from the main codeflash process. The test runner (`verification/test_runner.py`) invokes pytest (or Jest for JS/TS) with specific plugin configurations.

### Plugin Blocklists

- **Behavioral tests**: Block `benchmark`, `codspeed`, `xdist`, `sugar`
- **Benchmarking tests**: Block `codspeed`, `cov`, `benchmark`, `profiling`, `xdist`, `sugar`

These are defined as `BEHAVIORAL_BLOCKLISTED_PLUGINS` and `BENCHMARKING_BLOCKLISTED_PLUGINS` in `verification/test_runner.py`.

## Custom Pytest Plugin (`verification/pytest_plugin.py`)

The plugin is loaded into the test subprocess and provides:

### Deterministic Patches

`_apply_deterministic_patches()` replaces non-deterministic functions with fixed values to ensure reproducible test output:

| Module | Function | Fixed Value |
|--------|----------|-------------|
| `time` | `time()` | `1761717605.108106` |
| `time` | `perf_counter()` | Incrementing by 1ms per call |
| `datetime` | `datetime.now()` | `2021-01-01 02:05:10 UTC` |
| `datetime` | `datetime.utcnow()` | `2021-01-01 02:05:10 UTC` |
| `uuid` | `uuid4()` / `uuid1()` | `12345678-1234-5678-9abc-123456789012` |
| `random` | `random()` | `0.123456789` (seeded with 42) |
| `os` | `urandom(n)` | `b"\x42" * n` |
| `numpy.random` | seed | `42` |

Patches call the original function first to maintain performance characteristics (same call overhead).

### Timing Markers

Test results include timing markers in stdout: `!######<id>:<duration_ns>######!`

The pattern `_TIMING_MARKER_PATTERN` extracts timing data for calculating function utilization fraction.

### Loop Stability

Performance benchmarking uses configurable stability thresholds:
- `STABILITY_WINDOW_SIZE = 0.35` (35% of total iterations)
- `STABILITY_CENTER_TOLERANCE = 0.0025` (±0.25% around median)
- `STABILITY_SPREAD_TOLERANCE = 0.0025` (0.25% window spread)

### Memory Limits (Linux)

On Linux, the plugin sets `RLIMIT_AS` to 85% of total system memory (RAM + swap) to prevent OOM kills.

## Test Result Processing

### `TestResults` (`models/models.py`)

Collects `FunctionTestInvocation` results with:
- Deduplicated insertion via `unique_invocation_loop_id`
- `total_passed_runtime()` — Sum of minimum runtimes per test case (nanoseconds)
- `number_of_loops()` — Max loop index
- `usable_runtime_data_by_test_case()` — Grouped timing data

### `FunctionTestInvocation`

Each invocation records:
- `loop_index` — Iteration number (starts at 1)
- `id: InvocationId` — Fully qualified test identifier
- `did_pass: bool` — Pass/fail status
- `runtime: Optional[int]` — Time in nanoseconds
- `return_value: Optional[object]` — Captured return value
- `test_type: TestType` — Which test category

### Behavioral vs Performance Testing

1. **Behavioral**: Runs with `TestingMode.BEHAVIOR`. Compares return values and stdout between original and candidate. Any difference = candidate rejected.
2. **Performance**: Runs with `TestingMode.PERFORMANCE`. Loops for `TOTAL_LOOPING_TIME=10s` to get stable timing. Calculates speedup ratio.
3. **Line Profile**: Runs with `TestingMode.LINE_PROFILE`. Collects per-line timing data for refinement.

## Test Types

| TestType | Value | Description |
|----------|-------|-------------|
| `EXISTING_UNIT_TEST` | 1 | Pre-existing tests from the codebase |
| `INSPIRED_REGRESSION` | 2 | Tests inspired by existing tests |
| `GENERATED_REGRESSION` | 3 | AI-generated regression tests |
| `REPLAY_TEST` | 4 | Tests from recorded benchmark data |
| `CONCOLIC_COVERAGE_TEST` | 5 | Coverage-guided tests |
| `INIT_STATE_TEST` | 6 | Class init state verification |

## Coverage

Coverage is measured via `CoverageData` with a threshold of `COVERAGE_THRESHOLD=60.0%`. Low coverage may affect confidence in the optimization's correctness.
