# Daemon Mode

High-performance daemon mode for faster incremental type checking in development environments. The dmypy daemon provides significant performance improvements for repeated type checking of large codebases.

## Capabilities

### Daemon Operations

```python { .api }
# Available through dmypy command-line tool
# Start daemon: dmypy daemon
# Check files: dmypy check [files...]
# Stop daemon: dmypy stop
# Restart daemon: dmypy restart
# Get status: dmypy status
```

#### Performance Benefits

- **First run**: Same speed as regular mypy
- **Subsequent runs**: 10-50x faster for large codebases
- **Incremental analysis**: Only re-analyzes changed files and dependencies
- **Memory persistence**: Keeps type information in memory between runs

### Integration Usage

```python
from mypy import api

# Use daemon through programmatic API (not thread-safe)
result = api.run_dmypy(['check', 'myfile.py'])
stdout, stderr, exit_code = result

# Daemon management
api.run_dmypy(['daemon'])  # Start daemon
api.run_dmypy(['stop'])    # Stop daemon
```

For detailed daemon usage patterns, see [Command Line Tools](./command-line-tools.md) and [Programmatic API](./programmatic-api.md) documentation.