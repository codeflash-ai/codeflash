---
sidebar_position: 4
---
# How Codeflash Works

Codeflash follows a "generate and verify" approach to optimize code. It uses LLMs to generate optimizations, then it rigorously verifies if those optimizations are indeed
faster and if they have the same behavior. The basic unit of optimization is a function—Codeflash tries to speed up the function, and tries to ensure that it still behaves the same way. This way if you merge the optimized code, it simply runs faster without breaking any functionality.

## Analysis of your code

Codeflash scans your codebase to identify all available functions. It locates existing unit tests in your projects and maps which functions they test. When optimizing a function, Codeflash runs these discovered tests to verify nothing has broken.

#### What kind of functions can Codeflash optimize?

Codeflash works best with self-contained functions that have minimal side effects (like communicating with external systems or sending network requests). Codeflash optimizes a group of functions - consisting of an entry point function and any other functions it directly calls.
Currently, Codeflash cannot optimize async functions.

#### Test Discovery

Codeflash currently only runs tests that directly call the target function in their test body. To discover tests that indirectly call the function, you can use the Codeflash Tracer. The Tracer analyzes your test suite and identifies all tests that eventually call a function.

## Optimization Generation

To optimize code, Codeflash first gathers all necessary context from the codebase. It then calls our backend to generate several candidate optimizations. These are called "candidates" because their speed and correctness haven't been verified yet. Both properties will be verified in later steps.

## Verification of correctness

![Verification](/img/codeflash_arch_diagram.gif)

The goal of correctness verification is to ensure that when the original code is replaced by the new code, there are no behavioral changes in the code and the rest of the system. This means the replacement should be completely safe.

To verify correctness, Codeflash calls the function with numerous inputs, confirming that the new function behaves identically to the original.

Codeflash verifies these specific behaviors to be correct -

- function return values match exactly
- inputs to function have been mutated exactly the same way as before
- exception types remain consistent

Additionally, Codeflash checks for sufficient line coverage of the optimized code, increasing confidence in the testing process.

Codeflash also evaluates that there is sufficient line coverage of the code under optimization. This provides more confidence with testing.

We recommend manually reviewing the optimized code, since there might be important input cases that we haven’t verified where the behavior could differ.

#### Test Generation

Codeflash generates two types of tests:

- LLM Generated tests - Codeflash uses LLMs to create several regression test cases that cover typical function usage, edge cases, and large-scale inputs to verify both correctness and performance.
- Concolic coverage tests - Codeflash uses state-of-the-art concolic testing with an SMT Solver (a theorem prover) to explore execution paths and generate function arguments. This aims to maximize code coverage for the function being optimized. Codeflash runs the resulting test file to verify correctness. Currently, this feature only supports pytest.

## Code Execution

Codeflash runs tests for the target function using either pytest or unittest frameworks. The tests execute on your machine, ensuring access to the Python environment and any other dependencies associated to let Codeflash run your code properly. Running on your machine also ensures accurate performance measurements since runtime varies by system.

#### Performance benchmarking

Codeflash implements several techniques to measure code performance accurately. In particular, it runs multiple iterations of the code in a loop to determine the best performance with the minimum runtime. Codeflash compares performance of the original code against the optimization, requiring at least a 10% speed improvement before considering it faster. This approach eliminates most runtime measurement variability, even on noisy CI systems and virtual machines. The final runtime Codeflash reports is the minimum total time it took to run all the test cases.

## Creating Pull Requests

Once an optimization passes all checks, Codeflash creates a pull request through the Codeflash GitHub app directly in your repository. The pull request includes the new code, the speedup percentage, an explanation of the optimization, test statistics including coverage, and the test content itself. You can review and merge the new code if it meets your standards. Feel free to modify the code as needed—we welcome your improvements!
