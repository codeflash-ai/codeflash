# Code Replacement Experiment Results

Generated: 2026-01-14 18:26:02

## Summary

| Approach | Available | Passed | Failed | Errors | Pass Rate | Total Time |
|----------|-----------|--------|--------|--------|-----------|------------|
| Approach B: Text-Based | Yes | 19 | 0 | 0 | 100.0% | 0.04ms |
| Approach C: Hybrid | Yes | 19 | 0 | 0 | 100.0% | 0.08ms |
| Approach A: jscodeshift | Yes | 0 | 0 | 0 | 0.0% | 0.00ms |

## Approach B: Text-Based

**Description**: Pure Python text manipulation using line numbers

**Pass Rate**: 100.0% (19/19)

**Total Time**: 0.04ms

## Approach C: Hybrid

**Description**: Tree-sitter analysis + text replacement

**Pass Rate**: 100.0% (19/19)

**Total Time**: 0.08ms

## Approach A: jscodeshift

**Description**: AST-based replacement via Node.js subprocess

**Pass Rate**: 0.0% (0/0)

**Total Time**: 0.00ms

## Recommendations

**Recommended Approach**: Approach B: Text-Based

- Pass Rate: 100.0%
- Average Time: 0.00ms per test