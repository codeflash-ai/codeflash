---
name: debug-optimization-failure
description: >
  Diagnose why a codeflash optimization produced no results or failed silently.
  Use when an optimization run errors out, returns no candidates, or all candidates
  are rejected. Walks through discovery, ranking, context limits, AI service,
  test verification, deduplication, and repair stages.
---

# Debug Optimization Failure

Use this workflow when an optimization run fails or produces no results. Work through the stages sequentially — stop at the first failure found.

## Step 1: Check Function Discovery

Determine if the function was discovered by `FunctionVisitor`.

1. Search logs for the function name in discovery output:
   ```python
   # In discovery/functions_to_optimize.py, FunctionVisitor filters out:
   # - Functions matching exclude patterns in pyproject.toml [tool.codeflash]
   # - Functions already optimized (was_function_previously_optimized())
   # - Functions outside the configured module-root
   ```
2. Verify the function file is under the configured `module-root` in `pyproject.toml`
3. Check if the function was previously optimized — look for it in the optimization history

**Checkpoint**: If the function doesn't appear in discovery output, fix config patterns or file location before proceeding.

## Step 2: Check Ranking

If trace data is used, check if the function was ranked high enough.

1. Look at `benchmarking/function_ranker.py` output for the function's addressable time
2. The function must exceed `DEFAULT_IMPORTANCE_THRESHOLD=0.001`:
   ```python
   # Addressable time = own time + callee time / call count
   # Grep for the function in ranking output:
   # grep -i "function_name" in ranking logs
   ```
3. Functions below the threshold are silently skipped

**Checkpoint**: If ranked too low, the function doesn't spend enough time to be worth optimizing. No fix needed — this is expected.

## Step 3: Check Context Token Limits

Verify the function's context fits within token limits.

1. Check thresholds in `code_utils/config_consts.py`:
   ```python
   OPTIMIZATION_CONTEXT_TOKEN_LIMIT = 16000  # tokens
   TESTGEN_CONTEXT_TOKEN_LIMIT = 16000       # tokens
   ```
2. Token counting uses `encoded_tokens_len()` from `code_utils/code_utils.py`
3. Common causes: large helper function chains, deep dependency trees, large class hierarchies

**Checkpoint**: If context exceeds limits, the function is rejected. Consider refactoring to reduce dependencies or splitting large modules.

## Step 4: Check AI Service Response

Verify the AI service returned valid candidates.

1. Look for HTTP errors in logs:
   ```
   # Error patterns to search for:
   "Error generating optimized candidates"
   "Error generating jit rewritten candidate"
   "cli-optimize-error-caught"
   "cli-optimize-error-response"
   ```
2. Check `_get_valid_candidates()` in `api/aiservice.py` — empty `code_strings` after `CodeStringsMarkdown.parse_markdown_code()` means the LLM returned malformed code blocks
3. Verify API key is valid (`get_codeflash_api_key()`)

**Checkpoint**: If no candidates returned, check API key, network, and service status before proceeding.

## Step 5: Check Test Failures

Determine if candidates failed behavioral or benchmark tests.

1. **Behavioral failures** — compare return values, stdout, pass/fail between baseline and candidate:
   ```python
   # TestDiffScope enum values to look for:
   # RETURN_VALUE - function returned different value
   # STDOUT - different stdout output
   # DID_PASS - test passed/failed differently
   ```
2. **Benchmark failures** — candidate must beat `MIN_IMPROVEMENT_THRESHOLD=0.05` (5% speedup)
3. **Stability failures** — timing must be stable within `STABILITY_WINDOW_SIZE=0.35` (35% of iterations)
4. Check JUnit XML test results in the temp directory for specific failure messages

**Checkpoint**: Behavioral failure = optimization changed behavior (check test diffs). Benchmark failure = not fast enough. Stability failure = noisy timing environment.

## Step 6: Check Deduplication

Verify candidates weren't deduplicated away.

1. `CandidateEvaluationContext.ast_code_to_id` tracks normalized AST → candidate mapping
2. `normalize_code()` from `code_utils/deduplicate_code.py` strips comments/whitespace and normalizes the AST
3. If all candidates normalize to identical code, only the first is tested — the rest copy its results

**Checkpoint**: If all duplicates, the LLM generated the same optimization repeatedly. Try a higher effort level for more diverse candidates.

## Step 7: Check Repair/Refinement

If initial candidates failed, check repair and refinement stages.

1. Repair only triggers if fewer than `MIN_CORRECT_CANDIDATES=2` passed behavioral tests
2. Repair sends `AIServiceCodeRepairRequest` with `TestDiff` objects showing what went wrong
3. Check `REPAIR_UNMATCHED_PERCENTAGE_LIMIT` (effort-dependent: 0.2/0.3/0.4) — if too many tests failed, repair is skipped entirely
4. Refinement only runs on the top valid candidates (count depends on effort level)

**Checkpoint**: If repair also fails, the optimization approach likely doesn't work for this function. The function may rely on side effects or external state that the LLM can't safely optimize.

## Key Files Reference

| File | What to check |
|------|---------------|
| `optimization/function_optimizer.py` | Main loop, `determine_best_candidate()` |
| `verification/test_runner.py` | Test subprocess execution |
| `api/aiservice.py` | AI service requests/responses |
| `code_utils/config_consts.py` | All thresholds and limits |
| `context/code_context_extractor.py` | Context extraction and token counting |
| `models/models.py` | `CandidateEvaluationContext`, `TestResults`, `TestDiff` |
| `code_utils/deduplicate_code.py` | AST normalization for deduplication |
