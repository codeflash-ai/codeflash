# Architecture

When adding, moving, or deleting source files, update this doc to match.

```
codeflash/
├── main.py                 # CLI entry point
├── cli_cmds/               # Command handling, console output (Rich)
├── discovery/              # Find optimizable functions
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
│   ├── base.py                    # LanguageSupport protocol and shared data types
│   ├── registry.py                # Language registration and lookup by extension/enum
│   ├── current.py                 # Current language singleton (set_current_language / current_language_support)
│   ├── code_replacer.py           # Language-agnostic code replacement
│   ├── python/
│   │   ├── support.py             # PythonSupport (LanguageSupport implementation)
│   │   ├── function_optimizer.py  # PythonFunctionOptimizer subclass
│   │   ├── optimizer.py           # Python module preparation & AST resolution
│   │   └── normalizer.py          # Python code normalization for deduplication
│   └── javascript/
│       ├── support.py             # JavaScriptSupport (LanguageSupport implementation)
│       ├── function_optimizer.py  # JavaScriptFunctionOptimizer subclass
│       ├── optimizer.py           # JS project root finding & module preparation
│       └── normalizer.py          # JS/TS code normalization for deduplication
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
| Per-function optimization | `optimization/function_optimizer.py` (base), `languages/python/function_optimizer.py`, `languages/javascript/function_optimizer.py` |
| Function discovery | `discovery/functions_to_optimize.py` |
| Context extraction | `languages/<lang>/context/code_context_extractor.py` |
| Test execution | `languages/<lang>/support.py` (`run_behavioral_tests`, etc.), `verification/pytest_plugin.py` |
| Performance ranking | `benchmarking/function_ranker.py` |
| Domain types | `models/models.py`, `models/function_types.py` |
| Result handling | `either.py` (`Result`, `Success`, `Failure`, `is_successful`) |
