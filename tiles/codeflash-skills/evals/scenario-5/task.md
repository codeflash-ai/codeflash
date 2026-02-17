# Investigate Low Candidate Diversity

## Context

A codeflash user is optimizing a data processing function at medium effort level. The AI service returns 5 candidates, but the optimization log shows only 1 candidate was actually benchmarked. Of the 5 candidates, 1 passed behavioral tests but didn't meet the performance threshold. The user wants to understand what happened to the other 4 candidates and why no repair attempts were made.

## Task

Write an analysis document explaining:
1. Why only 1 out of 5 candidates was benchmarked
2. How the system determines which candidates to actually test
3. Under what conditions the system would have attempted to repair the failing candidates
4. What the user could change to get more diverse results

## Expected Outputs

A markdown file `analysis.md` with the explanation.
