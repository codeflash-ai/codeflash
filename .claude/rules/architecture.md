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
├── languages/              # Multi-language support (Python, JavaScript/TypeScript, Java planned)
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

## LanguageSupport Protocol Methods

Core protocol in `languages/base.py`. Each language (`PythonSupport`, `JavaScriptSupport`) implements these.

| Category | Method/Property | Purpose |
|----------|----------------|---------|
| Identity | `language`, `file_extensions`, `default_file_extension` | Language identification |
| Identity | `comment_prefix`, `dir_excludes` | Language conventions |
| AI service | `default_language_version` | Language version for API payloads (`None` for Python, `"ES2022"` for JS) |
| AI service | `valid_test_frameworks` | Allowed test frameworks for validation |
| Discovery | `discover_functions`, `discover_tests` | Find optimizable functions and their tests |
| Discovery | `adjust_test_config_for_discovery` | Pre-discovery config adjustment (no-op default) |
| Context | `extract_code_context`, `find_helper_functions`, `find_references` | Code dependency extraction |
| Transform | `replace_function`, `format_code`, `normalize_code` | Code modification |
| Validation | `validate_syntax` | Syntax checking |
| Test execution | `run_behavioral_tests`, `run_benchmarking_tests`, `run_line_profile_tests` | Test runners |
| Test results | `test_result_serialization_format` | `"pickle"` (Python) or `"json"` (JS) |
| Test results | `load_coverage` | Load coverage from language-specific format |
| Test results | `compare_test_results` | Equivalence checking between original and candidate |
| Test gen | `postprocess_generated_tests` | Post-process `GeneratedTestsList` objects |
| Test gen | `process_generated_test_strings` | Instrument/transform raw generated test strings |
| Module | `detect_module_system` | Detect project module system (`None` for Python, `"esm"`/`"commonjs"` for JS) |
| Module | `prepare_module` | Parse/validate module before optimization |
| Setup | `setup_test_config` | One-time project setup after language detection |
| Optimizer | `function_optimizer_class` | Return `FunctionOptimizer` subclass for this language |
