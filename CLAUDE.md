# CLAUDE.md

## Project Overview

CodeFlash is an AI-powered code optimizer that automatically improves performance while maintaining correctness. It supports Python, JavaScript, TypeScript, Java with more languages planned. It uses LLMs to generate optimization candidates, verifies correctness through test execution, and benchmarks performance improvements.

## Optimization Pipeline

```
Discovery -> Ranking -> Context Extraction -> Test Gen + Optimization -> Baseline -> Candidate Evaluation -> PR
```

See `.claude/rules/architecture.md` for directory mapping and entry points.

## Setup

```bash
uv sync                  # Install all dependencies
uv run prek install      # Install git pre-commit hooks (ruff + mypy)
```

## Bug Fix Workflow

Follow these steps in order, do not skip ahead:

1. Read the relevant code to understand the bug
2. Write a test that reproduces the bug (run it to confirm it fails)
3. Spawn subagents (using the Agent tool) to attempt the fix — each subagent should apply a fix and run the test to prove it passes
4. Review the subagent results, pick the best fix, and apply it
5. Never jump straight to writing a fix yourself — always go through steps 1-4

Everything that can be tested should have tests.
