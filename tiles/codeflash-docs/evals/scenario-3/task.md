# Explain Test Reproducibility Guarantees

## Context

A codeflash user notices that their optimization candidate passes behavioral tests on one run but fails on the next. They suspect non-determinism in the test execution. They want to understand what guarantees codeflash provides for test reproducibility and how the system ensures consistent results.

## Task

Write a technical explanation of how codeflash ensures deterministic test execution. Cover the execution environment setup, what sources of non-determinism are controlled, and any specific values or configurations used. Also explain the test execution architecture.

## Expected Outputs

- A markdown file `test-reproducibility.md`
