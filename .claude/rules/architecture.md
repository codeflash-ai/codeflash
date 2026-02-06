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
├── tracing/                # Function call tracing
├── lsp/                    # IDE integration (Language Server Protocol)
├── telemetry/              # Sentry, PostHog
├── either.py               # Functional Result type for error handling
└── result/                 # Result types and handling
```
