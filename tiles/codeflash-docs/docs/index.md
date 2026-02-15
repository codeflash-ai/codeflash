# Codeflash Internal Documentation

CodeFlash is an AI-powered Python code optimizer that automatically improves code performance while maintaining correctness. It uses LLMs to generate optimization candidates, verifies correctness through test execution, and benchmarks performance improvements.

## Pipeline Overview

```
Discovery → Ranking → Context Extraction → Test Gen + Optimization → Baseline → Candidate Evaluation → PR
```

1. **Discovery** (`discovery/`): Find optimizable functions across the codebase using `FunctionVisitor`
2. **Ranking** (`benchmarking/function_ranker.py`): Rank functions by addressable time using trace data
3. **Context** (`context/`): Extract code dependencies — split into read-writable (modifiable) and read-only (reference)
4. **Optimization** (`optimization/`, `api/`): Generate candidates via AI service, runs concurrently with test generation
5. **Verification** (`verification/`): Run candidates against tests via custom pytest plugin, compare outputs
6. **Benchmarking** (`benchmarking/`): Measure performance, select best candidate by speedup
7. **Result** (`result/`, `github/`): Create PR with winning optimization

## Key Entry Points

| Task | File |
|------|------|
| CLI arguments & commands | `cli_cmds/cli.py` |
| Optimization orchestration | `optimization/optimizer.py` → `Optimizer.run()` |
| Per-function optimization | `optimization/function_optimizer.py` → `FunctionOptimizer` |
| Function discovery | `discovery/functions_to_optimize.py` |
| Context extraction | `context/code_context_extractor.py` |
| Test execution | `verification/test_runner.py`, `verification/pytest_plugin.py` |
| Performance ranking | `benchmarking/function_ranker.py` |
| Domain types | `models/models.py`, `models/function_types.py` |
| AI service | `api/aiservice.py` → `AiServiceClient` |
| Configuration | `code_utils/config_consts.py` |

## Documentation Pages

- [Domain Types](domain-types.md) — Core data types and their relationships
- [Optimization Pipeline](optimization-pipeline.md) — Step-by-step data flow through the pipeline
- [Context Extraction](context-extraction.md) — How code context is extracted and token-limited
- [Verification](verification.md) — Test execution, pytest plugin, deterministic patches
- [AI Service](ai-service.md) — AI service client endpoints and request types
- [Configuration](configuration.md) — Config schema, effort levels, thresholds
