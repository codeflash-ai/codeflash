# Add Candidate Timeout Feature

## Context

The codeflash optimization engine currently has no per-candidate timeout. Some candidates take too long during verification, wasting the optimization budget. A new feature is needed to skip candidates that exceed a configurable time limit during behavioral testing.

The timeout should vary based on the optimization effort setting — shorter timeouts for low effort runs (to save time) and longer for high effort runs (to allow more complex optimizations).

## Task

Implement a `check_candidate_timeout` function in `codeflash/optimization/function_optimizer.py` that:
1. Takes a candidate runtime and returns whether the candidate should be skipped
2. Uses a configurable timeout threshold that scales with optimization effort
3. Handles the error case where the runtime measurement is unavailable

Also add the necessary configuration constant to `codeflash/code_utils/config_consts.py`.

## Expected Outputs

- Modified `function_optimizer.py` with the new function
- Modified `config_consts.py` with the new configuration
