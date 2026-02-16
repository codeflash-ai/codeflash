# Diagnose Silent Optimization Skip

## Context

A user reports that when running codeflash on their project, a specific function `calculate_metrics` in `analytics/processor.py` never appears in the optimization results. The function exists in the module root, is not in the exclude list, and has not been previously optimized. Trace data shows the function is called frequently but with very short execution times (averaging 0.0005 seconds total addressable time). The function has moderate dependencies.

## Task

Write a diagnostic report explaining why this function is being skipped and at which stage in the pipeline the function is filtered out. Include the specific threshold or condition that causes the skip.

## Expected Outputs

A markdown file `diagnostic-report.md` explaining the root cause.
