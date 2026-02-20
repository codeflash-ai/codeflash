---
paths:
  - "codeflash/optimization/**/*.py"
  - "codeflash/verification/**/*.py"
  - "codeflash/benchmarking/**/*.py"
  - "codeflash/context/**/*.py"
---

# Optimization Pipeline Patterns

- All major operations return `Result[SuccessType, ErrorType]` — construct with `Success(value)` / `Failure(error)`, check with `is_successful()` before calling `unwrap()`
- Code context has token limits (`OPTIMIZATION_CONTEXT_TOKEN_LIMIT`, `TESTGEN_CONTEXT_TOKEN_LIMIT` in `config_consts.py`) — exceeding them rejects the function
- `read_writable_code` can span multiple files; `read_only_context_code` is reference-only
- Code is serialized as markdown code blocks: ` ```language:filepath\ncode\n``` ` (see `CodeStringsMarkdown`)
- Candidates form a forest (DAG): refinements/repairs reference `parent_id` on previous candidates
- Test generation and optimization run concurrently — coordinate through `CandidateEvaluationContext`
- Generated tests are instrumented with `codeflash_capture.py` to record return values and traces
