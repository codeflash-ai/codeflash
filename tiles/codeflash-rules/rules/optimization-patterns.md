# Optimization Pipeline Patterns

- All major operations return `Result[SuccessType, ErrorType]` — construct with `Success(value)` / `Failure(error)`, check with `is_successful()` before calling `unwrap()`
- Code context has token limits (`OPTIMIZATION_CONTEXT_TOKEN_LIMIT=16000`, `TESTGEN_CONTEXT_TOKEN_LIMIT=16000` in `code_utils/config_consts.py`) — exceeding them rejects the function
- `read_writable_code` (modifiable code) can span multiple files; `read_only_context_code` is reference-only dependency code
- Code is serialized as markdown code blocks: `` ```language:filepath\ncode\n``` `` — see `CodeStringsMarkdown` in `models/models.py`
- Candidates form a forest (DAG): refinements/repairs reference `parent_id` on previous candidates via `OptimizedCandidateSource` (OPTIMIZE, REFINE, REPAIR, ADAPTIVE, JIT_REWRITE)
- Test generation and optimization run concurrently — coordinate through `CandidateEvaluationContext`
- Generated tests are instrumented with `codeflash_capture.py` to record return values and traces
- Minimum improvement threshold is 5% (`MIN_IMPROVEMENT_THRESHOLD=0.05`) — candidates below this are rejected
- Stability thresholds: `STABILITY_WINDOW_SIZE=0.35`, `STABILITY_CENTER_TOLERANCE=0.0025`, `STABILITY_SPREAD_TOLERANCE=0.0025`
