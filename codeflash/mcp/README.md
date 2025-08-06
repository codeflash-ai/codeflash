# Codeflash MCP Server Integration

This directory contains the Model Context Protocol (MCP) server implementation that makes Codeflash available as a Claude Code subagent.

## Architecture Overview

```
codeflash/mcp/
├── __init__.py          # Package initialization
├── server.py            # Main FastMCP server implementation
├── integration.py       # Claude Code integration utilities
└── README.md           # This file
```

## Components

### server.py
The main MCP server implementation using FastMCP. Provides:
- **12 MCP Tools** for all codeflash optimization workflows
- **Structured responses** using Pydantic models
- **Error handling** and timeout management
- **Resource providers** for project context

### integration.py
Handles automatic setup and configuration:
- **Auto-detection** of Claude Code installation
- **Configuration management** for MCP servers
- **Cross-platform support** (macOS, Linux, Windows)
- **Status checking** and troubleshooting

## Available MCP Tools

| Tool | Description | Usage |
|------|-------------|--------|
| `optimize_function` | Optimize a specific function | `optimize_function("bubble_sort", "algorithms.py")` |
| `optimize_file` | Optimize all functions in a file | `optimize_file("algorithms.py")` |
| `trace_and_optimize` | Trace script and optimize | `trace_and_optimize("python main.py")` |
| `optimize_from_replay_tests` | Use tests for optimization | `optimize_from_replay_tests(["test_*.py"])` |
| `optimize_all_functions` | Optimize entire project | `optimize_all_functions()` ⚠️ |
| `initialize_project` | Set up codeflash | `initialize_project()` |
| `setup_github_actions` | Configure CI/CD | `setup_github_actions()` |
| `verify_installation` | Test setup | `verify_installation()` |
| `run_benchmarks` | Performance testing | `run_benchmarks()` |
| `get_codeflash_status` | Project status | `get_codeflash_status()` |
| `get_optimization_help` | Usage guide | `get_optimization_help()` |

## Data Models

### OptimizationResult
```python
{
    "success": bool,
    "message": str,
    "optimized_functions": List[str],
    "performance_improvement": Optional[str],
    "file_path": Optional[str],
    "errors": List[str]
}
```

### ProjectStatus
```python
{
    "is_initialized": bool,
    "config_file": Optional[str],
    "module_root": Optional[str],
    "tests_root": Optional[str],
    "version": str
}
```

### TraceResult
```python
{
    "success": bool,
    "trace_file": Optional[str],
    "functions_traced": int,
    "message": str
}
```

## Installation & Setup

### 1. Install with MCP Support
```bash
pip install codeflash[mcp]
```

### 2. Auto-Configure Claude Code
```bash
codeflash setup claude-code
```

### 3. Manual Configuration (if needed)
Add to your Claude Code `config.json`:
```json
{
  "mcpServers": {
    "codeflash": {
      "command": "codeflash-mcp",
      "args": [],
      "env": {},
      "disabled": false
    }
  }
}
```

## Usage Examples

### In Claude Code
```
Human: Optimize the bubble_sort function in my algorithms.py file

Claude: I'll use codeflash to optimize your bubble_sort function.
[Uses optimize_function tool]