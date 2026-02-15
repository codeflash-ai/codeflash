---
name: debug-optimization-failure
description: Debug why a codeflash optimization failed at any pipeline stage
---

# Debug Optimization Failure

Use this workflow when an optimization run fails or produces no results. Work through the stages sequentially — stop at the first failure found.

## Step 1: Check Function Discovery

Determine if the function was discovered by `FunctionVisitor`.

1. Look at the discovery output or logs for the function name
2. Check `discovery/functions_to_optimize.py` — the `FunctionVisitor` filters out:
   - Functions that are too small or trivial
   - Functions matching exclude patterns in config
   - Functions already optimized (`was_function_previously_optimized()`)
3. Verify the function file is under the configured `module-root`

**If not discovered**: Check config patterns, file location, and function size.

## Step 2: Check Ranking

If trace data is used, check if the function was ranked high enough.

1. Look at `benchmarking/function_ranker.py` output
2. The function's **addressable time** must exceed `DEFAULT_IMPORTANCE_THRESHOLD=0.001`
3. Addressable time = own time + callee time / call count

**If ranked too low**: The function doesn't spend enough time to be worth optimizing.

## Step 3: Check Context Token Limits

Verify the function's context fits within token limits.

1. Check `OPTIMIZATION_CONTEXT_TOKEN_LIMIT=16000` and `TESTGEN_CONTEXT_TOKEN_LIMIT=16000` in `code_utils/config_consts.py`
2. Token counting is done by `encoded_tokens_len()` in `code_utils/code_utils.py`
3. Large helper function chains or deep dependency trees can blow the limit

**If context too large**: The function has too many dependencies. Consider refactoring to reduce context size.

## Step 4: Check AI Service Response

Verify the AI service returned valid candidates.

1. Check logs for `AiServiceClient` request/response
2. Look for HTTP errors (non-200 status codes)
3. Verify `_get_valid_candidates()` parsed the response — empty `code_strings` means invalid markdown code blocks
4. Check if all candidates were filtered out during parsing

**If no candidates returned**: Check API key, network connectivity, and service status.

## Step 5: Check Test Failures

Determine if candidates failed behavioral or benchmark tests.

1. **Behavioral failures**: Compare return values, stdout, pass/fail status between original baseline and candidate
   - Check `TestDiffScope`: `RETURN_VALUE`, `STDOUT`, `DID_PASS`
   - Look at JUnit XML results for specific test failures
2. **Benchmark failures**: Check if candidate met `MIN_IMPROVEMENT_THRESHOLD=0.05` (5% speedup)
3. **Stability failures**: Check if timing was stable within `STABILITY_WINDOW_SIZE=0.35`

**If behavioral failure**: The optimization changed the function's behavior. Check test diffs for specific mismatches.
**If benchmark failure**: The optimization didn't provide enough speedup.

## Step 6: Check Deduplication

Verify candidates weren't deduplicated away.

1. `CandidateEvaluationContext.ast_code_to_id` tracks normalized code → candidate mapping
2. `normalize_code()` from `code_utils/deduplicate_code.py` normalizes AST for comparison
3. If all candidates normalize to the same code, only one is actually tested

**If all duplicates**: The LLM generated the same optimization multiple times. Try higher effort level.

## Step 7: Check Repair/Refinement

If initial candidates failed, check repair and refinement stages.

1. Repair only runs if fewer than `MIN_CORRECT_CANDIDATES=2` passed
2. Repair sends `AIServiceCodeRepairRequest` with test diffs
3. Check `REPAIR_UNMATCHED_PERCENTAGE_LIMIT` — if too many tests failed, repair is skipped
4. Refinement only runs on top valid candidates

**If repair also failed**: The optimization approach may not work for this function.

## Key Files to Check

- `optimization/function_optimizer.py` — Main optimization loop, `determine_best_candidate()`
- `verification/test_runner.py` — Test execution
- `api/aiservice.py` — AI service communication
- `code_utils/config_consts.py` — Thresholds
- `context/code_context_extractor.py` — Context extraction
- `models/models.py` — `CandidateEvaluationContext`, `TestResults`
