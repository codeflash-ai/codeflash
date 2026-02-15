# Context Extraction

How codeflash extracts and limits code context for optimization and test generation.

## Overview

Context extraction (`context/code_context_extractor.py`) builds a `CodeOptimizationContext` containing all code needed for the LLM to understand and optimize a function, split into:

- **Read-writable code** (`CodeContextType.READ_WRITABLE`): The function being optimized plus its helper functions — code the LLM is allowed to modify
- **Read-only context** (`CodeContextType.READ_ONLY`): Dependency code for reference — imports, type definitions, base classes
- **Testgen context** (`CodeContextType.TESTGEN`): Context for test generation, may include imported class definitions and external base class inits
- **Hashing context** (`CodeContextType.HASHING`): Used for deduplication of optimization runs

## Token Limits

Both optimization and test generation contexts are token-limited:
- `OPTIMIZATION_CONTEXT_TOKEN_LIMIT = 16000` tokens
- `TESTGEN_CONTEXT_TOKEN_LIMIT = 16000` tokens

Token counting uses `encoded_tokens_len()` from `code_utils/code_utils.py`. Functions whose context exceeds these limits are skipped.

## Context Building Process

### 1. Helper Discovery

For the target function (`FunctionToOptimize`), the extractor finds:
- **Helpers of the function**: Functions/classes in the same file that the target function calls
- **Helpers of helpers**: Transitive dependencies of the helper functions

These are organized as `dict[Path, set[FunctionSource]]` — mapping file paths to the set of helper functions found in each file.

### 2. Code Extraction

`extract_code_markdown_context_from_files()` builds `CodeStringsMarkdown` from the helper dictionaries. Each file's relevant code is extracted as a `CodeString` with its file path.

### 3. Testgen Context Enrichment

`build_testgen_context()` extends the basic context with:
- Imported class definitions (resolved from imports)
- External base class `__init__` methods
- External class `__init__` methods referenced in the context

### 4. Unused Definition Removal

`detect_unused_helper_functions()` and `remove_unused_definitions_by_function_names()` from `context/unused_definition_remover.py` prune definitions that are not transitively reachable from the target function, reducing token usage.

### 5. Deduplication

The hashing context (`hashing_code_context`) generates a hash (`hashing_code_context_hash`) used to detect when the same function context has already been optimized in a previous run, avoiding redundant work.

## Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `build_testgen_context()` | `context/code_context_extractor.py` | Build enriched testgen context |
| `extract_code_markdown_context_from_files()` | `context/code_context_extractor.py` | Convert helper dicts to `CodeStringsMarkdown` |
| `detect_unused_helper_functions()` | `context/unused_definition_remover.py` | Find unused definitions |
| `remove_unused_definitions_by_function_names()` | `context/unused_definition_remover.py` | Remove unused definitions |
| `collect_top_level_defs_with_usages()` | `context/unused_definition_remover.py` | Analyze definition usage |
| `encoded_tokens_len()` | `code_utils/code_utils.py` | Count tokens in code |
