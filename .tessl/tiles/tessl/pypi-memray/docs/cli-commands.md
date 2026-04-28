# CLI Commands

Command-line interface for profiling applications and generating reports. Provides comprehensive tools for capturing profiles, generating visualizations, and analyzing memory usage patterns through terminal commands.

## Capabilities

### Profiling Commands

#### run

Execute Python scripts with memory tracking enabled.

```bash
memray run [options] script.py [script_args...]
```

Options:
- `--output FILE, -o FILE`: Output file for captured data (default: memray-{pid}.bin)
- `--live-remote`: Enable live remote monitoring
- `--live-port PORT`: Port for live monitoring (default: 12345)
- `--native`: Enable native stack traces
- `--follow-fork`: Continue tracking in forked processes
- `--trace-python-allocators`: Trace Python allocators separately
- `--quiet, -q`: Suppress output
- `--aggregate`: Use aggregated allocations format

Usage examples:

```bash
# Basic profiling
memray run --output profile.bin my_script.py

# With native traces and live monitoring
memray run --native --live-remote --live-port 8080 my_script.py

# Profile with arguments passed to script
memray run --output profile.bin my_script.py --input data.csv --verbose
```

#### attach

Attach to a running Python process for live profiling.

```bash
memray attach [options] pid
```

Options:
- `--output FILE, -o FILE`: Output file for captured data
- `--native`: Enable native stack traces
- `--duration SECONDS`: Duration to profile (default: indefinite)
- `--method METHOD`: Attachment method (default: auto)

Usage examples:

```bash
# Attach to process by PID
memray attach --output live_profile.bin 12345

# Attach for specific duration with native traces
memray attach --native --duration 60 --output short_profile.bin 12345
```

#### detach

Detach from a running Python process that was previously attached.

```bash
memray detach [options] pid
```

Usage example:

```bash
# Detach from process
memray detach 12345
```

### Report Generation Commands

#### flamegraph

Generate interactive HTML flame graphs from capture files.

```bash
memray flamegraph [options] capture_file.bin
```

Options:
- `--output FILE, -o FILE`: Output HTML file (default: memray-flamegraph-{pid}.html)
- `--leaks`: Show only leaked allocations
- `--temporary-allocations`: Show temporary allocations
- `--merge-threads`: Merge allocations across threads
- `--inverted`: Generate inverted flame graph
- `--temporal`: Show temporal allocation patterns

Usage examples:

```bash
# Basic flame graph
memray flamegraph profile.bin

# Focus on memory leaks
memray flamegraph --leaks --output leaks.html profile.bin

# Temporal flame graph with merged threads
memray flamegraph --temporal --merge-threads profile.bin
```

#### table

Generate interactive HTML table reports with sortable allocation data.

```bash
memray table [options] capture_file.bin
```

Options:
- `--output FILE, -o FILE`: Output HTML file (default: memray-table-{pid}.html)
- `--leaks`: Show only leaked allocations  
- `--temporary-allocations`: Show temporary allocations
- `--merge-threads`: Merge allocations across threads

Usage examples:

```bash
# Basic table report
memray table profile.bin

# Leaked allocations table
memray table --leaks --output leak_table.html profile.bin
```

#### tree

Generate text-based tree reports showing allocation hierarchies.

```bash
memray tree [options] capture_file.bin
```

Options:
- `--leaks`: Show only leaked allocations
- `--temporary-allocations`: Show temporary allocations  
- `--merge-threads`: Merge allocations across threads
- `--biggest-allocs N`: Show N biggest allocations (default: 10)

Usage examples:

```bash
# Basic tree report  
memray tree profile.bin

# Show top 20 leaked allocations
memray tree --leaks --biggest-allocs 20 profile.bin
```

### Analysis Commands

#### summary

Generate summary statistics from capture files.

```bash
memray summary [options] capture_file.bin
```

Options:
- `--json`: Output in JSON format
- `--merge-threads`: Merge statistics across threads

Usage examples:

```bash
# Text summary
memray summary profile.bin

# JSON output for scripting
memray summary --json profile.bin > summary.json
```

#### stats

Generate detailed statistics and allocation breakdowns.

```bash
memray stats [options] capture_file.bin
```

Options:
- `--json`: Output in JSON format
- `--merge-threads`: Merge statistics across threads

Usage example:

```bash
# Detailed statistics
memray stats profile.bin
```

#### parse

Parse and extract raw data from capture files.

```bash
memray parse [options] capture_file.bin
```

Options:
- `--output FILE, -o FILE`: Output file for parsed data
- `--format FORMAT`: Output format (json, csv)

Usage example:

```bash
# Extract to JSON
memray parse --format json --output data.json profile.bin
```

### Live Monitoring Commands

#### live

Connect to and monitor live profiling sessions.

```bash
memray live [options] port
```

Options:
- `--refresh-rate SECONDS`: Update interval (default: 1)
- `--merge-threads`: Merge allocations across threads

Usage examples:

```bash
# Monitor live session
memray live 12345

# Monitor with custom refresh rate
memray live --refresh-rate 0.5 12345
```

### File Transformation Commands

#### transform

Transform capture files between different formats.

```bash
memray transform [options] input_file.bin output_file.bin
```

Options:
- `--format FORMAT`: Target format (all_allocations, aggregated_allocations)

Usage example:

```bash
# Convert to aggregated format
memray transform --format aggregated_allocations profile.bin profile_agg.bin
```

## Global Options

Most commands support these global options:

- `--help, -h`: Show command help
- `--version`: Show memray version  
- `--verbose, -v`: Verbose output
- `--quiet, -q`: Suppress non-error output

## Integration Examples

### Continuous Integration

```bash
#!/bin/bash
# CI script for memory profiling

# Run tests with profiling
memray run --output ci_profile.bin -m pytest tests/

# Generate reports
memray flamegraph --output reports/flamegraph.html ci_profile.bin
memray summary --json ci_profile.bin > reports/summary.json

# Check for memory leaks
if memray tree --leaks ci_profile.bin | grep -q "Leaked"; then
    echo "Memory leaks detected!"
    exit 1
fi
```

### Development Workflow

```bash
# Profile during development
memray run --live-remote --live-port 8080 my_app.py &
APP_PID=$!

# Monitor in another terminal
memray live 8080

# Generate reports after stopping
memray flamegraph memray-${APP_PID}.bin
memray table memray-${APP_PID}.bin
```