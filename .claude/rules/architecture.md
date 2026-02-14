# Architecture

```
codeflash/
├── main.py                 # CLI entry point
├── cli_cmds/               # Command handling, console output (Rich)
├── discovery/              # Find optimizable functions
├── context/                # Extract code dependencies and imports
├── optimization/           # Generate optimized code via AI
│   ├── optimizer.py        # Main optimization orchestration
│   └── function_optimizer.py  # Per-function optimization logic
├── verification/           # Run deterministic tests (pytest plugin)
├── benchmarking/           # Performance measurement
├── github/                 # PR creation
├── api/                    # AI service communication
├── code_utils/             # Code parsing, git utilities
├── models/                 # Pydantic models and types
├── languages/              # Multi-language support (Python, JavaScript/TypeScript)
├── setup/                  # Config schema, auto-detection, first-run experience
├── picklepatch/            # Serialization/deserialization utilities
├── tracing/                # Function call tracing
├── tracer.py               # Root-level tracer entry point for profiling
├── lsp/                    # IDE integration (Language Server Protocol)
├── telemetry/              # Sentry, PostHog
├── either.py               # Functional Result type for error handling
├── result/                 # Result types and handling
└── version.py              # Version information
```

## Key Entry Points

| Task | Start here |
|------|------------|
| CLI arguments & commands | `cli_cmds/cli.py` |
| Optimization orchestration | `optimization/optimizer.py` → `run()` |
| Per-function optimization | `optimization/function_optimizer.py` |
| Function discovery | `discovery/functions_to_optimize.py` |
| Context extraction | `context/code_context_extractor.py` |
| Test execution | `verification/test_runner.py`, `verification/pytest_plugin.py` |
| Performance ranking | `benchmarking/function_ranker.py` |
| Domain types | `models/models.py`, `models/function_types.py` |
| Result handling | `either.py` (`Result`, `Success`, `Failure`, `is_successful`) |
