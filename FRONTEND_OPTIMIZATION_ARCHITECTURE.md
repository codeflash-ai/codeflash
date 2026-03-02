# Frontend Optimization Architecture

## Overview

React frontend optimization extends Codeflash's existing JS/TS support with
framework-specific component discovery, React Profiler instrumentation,
memoization-focused optimization, and render-count-aware acceptance criteria.

## Design Principle

React optimization is a **framework plugin** under JS/TS, not a separate language.
It reuses the existing tree-sitter parser (JSX/TSX), Jest/Vitest runner, comparator,
and instrumentation infrastructure. Framework-specific logic is isolated in
`languages/javascript/frameworks/react/`.

This pattern is designed to be reused for other frontend frameworks (Vue, Angular, Svelte)
by adding new directories under `languages/javascript/frameworks/<framework>/`.

## Architecture Diagram

    ┌──────────────────────────────────────────────────────────┐
    │                     CLI Entry Point                       │
    │  codeflash optimize --file src/components/TaskList.tsx    │
    └────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
    ┌──────────────────────────────────────────────────────────┐
    │              Framework Detection (detector.py)            │
    │  package.json → FrameworkInfo(name="react", version=18)   │
    └────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
    ┌──────────────────────────────────────────────────────────┐
    │           Component Discovery (discovery.py)              │
    │  tree-sitter → ReactComponentInfo[]                       │
    │  PascalCase + JSX return + hooks → is_react_component     │
    └────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
    ┌──────────────────────────────────────────────────────────┐
    │         Context Extraction (context.py + analyzer.py)     │
    │  Component source + props interface + hooks + parents     │
    │  + optimization opportunities (missing memo/useMemo)      │
    └────────────────────────┬─────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
    ┌───────────────────┐      ┌──────────────────────────────┐
    │   Test Generation  │      │    Optimization Generation    │
    │   (testgen.py)     │      │    (aiservice + react prompts)│
    │                    │      │                                │
    │  React Testing     │      │  React.memo wrapping          │
    │  Library tests     │      │  useMemo for computations     │
    │  + re-render       │      │  useCallback for handlers     │
    │  counting tests    │      │  Inline object elimination    │
    └────────┬──────────┘      └──────────────┬───────────────┘
              │                                │
              └──────────────┬─────────────────┘
                             ▼
    ┌──────────────────────────────────────────────────────────┐
    │           Verification (existing Jest/Vitest runner)       │
    │  Run existing + generated tests on original AND candidate  │
    │  compare_test_results() → behavioral equivalence           │
    └────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
    ┌──────────────────────────────────────────────────────────┐
    │         React Benchmarking (profiler.py + benchmarking.py)│
    │  React Profiler instrumentation → render count + duration  │
    │  Parse !######REACT_RENDER:...######! markers              │
    │  Compare original vs optimized render profiles             │
    └────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
    ┌──────────────────────────────────────────────────────────┐
    │              Critic (render_efficiency_critic)             │
    │  Accept if: render_count reduced >= 20%                   │
    │         OR: render_duration reduced >= threshold           │
    │        AND: all tests pass (behavioral equivalence)        │
    └────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
    ┌──────────────────────────────────────────────────────────┐
    │                    PR Creation                             │
    │  "Reduced re-renders from 47 → 3 (93.6% fewer)"          │
    │  "Render time: 340ms → 12ms (28x faster)"                 │
    │  Includes generated tests + optimized component code       │
    └──────────────────────────────────────────────────────────┘

## File Structure

    languages/javascript/
    ├── support.py                    # Delegates to react/ when framework detected
    ├── treesitter.py                 # Reused — already handles JSX/TSX
    ├── test_runner.py                # Reused — runs Jest/Vitest
    ├── comparator.py                 # Reused — compares test results
    ├── instrument.py                 # Reused — capturePerf wrapping
    ├── parse.py                      # Extended — parse REACT_RENDER markers
    ├── frameworks/
    │   ├── __init__.py
    │   ├── detector.py               # Detect React from package.json
    │   └── react/
    │       ├── __init__.py
    │       ├── discovery.py          # Find React components via tree-sitter
    │       ├── analyzer.py           # Detect optimization opportunities
    │       ├── profiler.py           # React Profiler instrumentation
    │       ├── context.py            # Component context extraction
    │       ├── benchmarking.py       # Render count/duration comparison
    │       └── testgen.py            # React Testing Library test helpers

## Verification Model

React optimization uses the SAME verification guarantees as code optimization:

    ┌─────────────────────────────────────────────────┐
    │           Behavioral Equivalence                 │
    │                                                   │
    │  For each test invocation:                        │
    │    original.did_pass == candidate.did_pass        │
    │    original.return_value ≡ candidate.return_value │
    │    original.stdout ≡ candidate.stdout             │
    │                                                   │
    │  PLUS for React:                                  │
    │    candidate.render_count <= original.render_count │
    │    candidate.render_duration < original.render_duration │
    └─────────────────────────────────────────────────┘

## Optimization Patterns (Tier 1)

| Pattern | Detection | Optimization | Verification |
|---------|-----------|-------------|--------------|
| Missing React.memo | Component not wrapped + receives stable props | Wrap with React.memo() | Same render output, fewer renders |
| Missing useMemo | Array .filter/.sort/.map in render body | Wrap computation in useMemo() | Same computed result, fewer executions |
| Missing useCallback | Function defined in render, passed as prop | Wrap with useCallback() | Same behavior, stable reference |
| Inline object props | Object literal in JSX prop position | Extract to useMemo or module constant | Same prop values, stable reference |

## Extending to Other Frameworks

To add support for a new framework (e.g., Vue):

1. Create `languages/javascript/frameworks/vue/` directory
2. Add discovery module: detect Vue SFCs (`.vue` files), `<script setup>`, composition API
3. Add analyzer module: detect reactivity patterns (missing `computed()`, inline watchers)
4. Add context module: extract component props, emits, slots, composables
5. Add profiler module: instrument with Vue Devtools performance API
6. Add benchmarking module: compare reactive dependency tracking counts
7. Update `detector.py` to detect Vue from `package.json`
8. Add backend prompts in `django/aiservice/core/languages/js_ts/prompts/vue_optimizer/`

The framework detection → discovery → context → optimize → verify → benchmark pipeline
is shared infrastructure. Only the framework-specific detection and analysis differ.

## Test Project

A comprehensive React test project lives at `code_to_optimize/js/code_to_optimize_react/`
with components that exercise each optimization pattern:

| Component | Optimization Patterns |
|-----------|----------------------|
| TaskBoard | useMemo, useCallback, inline styles, React.memo |
| DataGrid | useMemo (aggregations, pagination), useCallback, inline styles |
| SearchableList | Unstable references defeating memo'd children, useMemo |
| FormWithValidation | useMemo (validation), useCallback (handlers), regex caching |
| VirtualizedTree | useMemo (tree flattening, search), useCallback |
| OptimizedCounter | Already optimized — should be skipped |
| ServerDashboard | Server Component — should be skipped |
| useFilteredData | Hook returning unstable object reference |
| useDebounce | Standard hook — should be detected as hook, not component |
