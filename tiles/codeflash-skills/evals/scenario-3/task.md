# Write Tests for Context Hash Comparison

## Context

The codeflash context extraction module has a function `compare_context_hashes(context_a, context_b)` that takes two `CodeOptimizationContext` objects and returns whether their hashing contexts are identical. This is used to detect when the same function has already been optimized.

```python
# In codeflash/context/code_context_extractor.py
def compare_context_hashes(context_a: CodeOptimizationContext, context_b: CodeOptimizationContext) -> bool:
    return context_a.hashing_code_context_hash == context_b.hashing_code_context_hash
```

## Task

Write a test file `tests/test_context/test_hash_comparison.py` with tests for this function. Include tests for:
1. Two contexts with identical code producing the same hash
2. Two contexts with different code producing different hashes
3. A context compared with itself

The tests should create temporary Python source files to build realistic context objects.

## Expected Outputs

- `tests/test_context/test_hash_comparison.py`
