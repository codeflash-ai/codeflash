# AI Service

How codeflash communicates with the AI optimization backend.

## `AiServiceClient` (`api/aiservice.py`)

The client connects to the AI service at `https://app.codeflash.ai` (or `http://localhost:8000` when `CODEFLASH_AIS_SERVER=local`).

Authentication uses Bearer token from `get_codeflash_api_key()`. All requests go through `make_ai_service_request()` which handles JSON serialization via Pydantic encoder.

Timeout: 90s for production, 300s for local.

## Endpoints

### `/ai/optimize` — Generate Candidates

Method: `optimize_code()`

Sends source code + dependency context to generate optimization candidates.

Payload:
- `source_code` — The read-writable code (markdown format)
- `dependency_code` — Read-only context code
- `trace_id` — Unique trace ID for the optimization run
- `language` — `"python"`, `"javascript"`, or `"typescript"`
- `n_candidates` — Number of candidates to generate (controlled by effort level)
- `is_async` — Whether the function is async
- `is_numerical_code` — Whether the code is numerical (affects optimization strategy)

Returns: `list[OptimizedCandidate]` with `source=OptimizedCandidateSource.OPTIMIZE`

### `/ai/optimize_line_profiler` — Line-Profiler-Guided Candidates

Method: `optimize_python_code_line_profiler()`

Like `/optimize` but includes `line_profiler_results` to guide the LLM toward hot lines.

Returns: candidates with `source=OptimizedCandidateSource.OPTIMIZE_LP`

### `/ai/refine` — Refine Existing Candidate

Method: `refine_code()`

Request type: `AIServiceRefinerRequest`

Sends an existing candidate with runtime data and line profiler results to generate an improved version.

Key fields:
- `original_source_code` / `optimized_source_code` — Before and after
- `original_code_runtime` / `optimized_code_runtime` — Timing data
- `speedup` — Current speedup ratio
- `original_line_profiler_results` / `optimized_line_profiler_results`

Returns: candidates with `source=OptimizedCandidateSource.REFINE` and `parent_id` set to the refined candidate's ID

### `/ai/repair` — Fix Failed Candidate

Method: `repair_code()`

Request type: `AIServiceCodeRepairRequest`

Sends a failed candidate with test diffs showing what went wrong.

Key fields:
- `original_source_code` / `modified_source_code`
- `test_diffs: list[TestDiff]` — Each with `scope` (return_value/stdout/did_pass), original vs candidate values, and test source code

Returns: candidates with `source=OptimizedCandidateSource.REPAIR` and `parent_id` set

### `/ai/adaptive_optimize` — Multi-Candidate Adaptive

Method: `adaptive_optimize()`

Request type: `AIServiceAdaptiveOptimizeRequest`

Sends multiple previous candidates with their speedups for the LLM to learn from and generate better candidates.

Key fields:
- `candidates: list[AdaptiveOptimizedCandidate]` — Previous candidates with source code, explanation, source type, and speedup

Returns: candidates with `source=OptimizedCandidateSource.ADAPTIVE`

### `/ai/rewrite_jit` — JIT Rewrite

Method: `get_jit_rewritten_code()`

Rewrites code to use JIT compilation (e.g., Numba).

Returns: candidates with `source=OptimizedCandidateSource.JIT_REWRITE`

## Candidate Parsing

All endpoints return JSON with an `optimizations` array. Each entry has:
- `source_code` — Markdown-formatted code blocks
- `explanation` — LLM explanation
- `optimization_id` — Unique ID
- `parent_id` — Optional parent reference
- `model` — Which LLM model was used

`_get_valid_candidates()` parses the markdown code via `CodeStringsMarkdown.parse_markdown_code()` and filters out entries with empty code blocks.

## `LocalAiServiceClient`

Used when `CODEFLASH_EXPERIMENT_ID` is set. Mirrors `AiServiceClient` but sends to a separate experimental endpoint for A/B testing optimization strategies.

## LLM Call Sequencing

`AiServiceClient` tracks call sequence via `llm_call_counter` (itertools.count). Each request includes a `call_sequence` number, used by the backend to maintain conversation context across multiple calls for the same function.
