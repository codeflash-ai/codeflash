# Implement a Function Optimization Status Tracker

## Context

The codeflash team needs a status tracker that logs what happens to each function during an optimization run. For each function, it should record the function identity, which pipeline stages it passed through, and how long each stage took.

## Task

Write a design document explaining:
1. What data structure represents a function being optimized, including its identity fields and how nested functions (methods inside classes) are represented
2. The full name resolution strategy for identifying functions uniquely
3. Which stages of the pipeline operate on a single function at a time vs. operating on multiple functions
4. Where in the codebase the per-function optimization is orchestrated and what the top-level entry point is

## Expected Outputs

- A markdown file `status-tracker-design.md`
