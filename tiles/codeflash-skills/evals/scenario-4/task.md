# Add Optimization Confidence Score

## Context

The codeflash team wants to add a confidence score to each optimization result. The score should capture how confident the system is that an optimization is both correct and beneficial. It combines test coverage percentage, number of passing test cases, and speedup stability into a single metric.

The score needs to be:
- Attached to each candidate during evaluation (immutable once computed)
- Included in the final PR report (needs JSON serialization)
- Computed during the candidate evaluation phase

## Task

1. Define the data types needed for the confidence score
2. Write a `compute_confidence_score` function that takes coverage percentage (float), passing test count (int), and stability ratio (float) and returns the confidence result
3. Place all code in the appropriate codeflash modules

## Expected Outputs

- New/modified type definitions in the appropriate models file
- New function in the appropriate module
