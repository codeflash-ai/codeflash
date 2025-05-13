---
sidebar_position: 1
slug: /
---
## What is Codeflash?

Welcome! Codeflash is an AI performance optimizer for Python code.
Codeflash speeds up Python code by figuring out the best way to rewrite your code while verifying that the behavior of the code is unchanged.

The optimizations Codeflash finds are generally better algorithms, opportunities to remove wasteful compute, better logic, and utilization of more efficient library methods. Codeflash
does not modify the architecture of your code, but it tries to find the most efficient implementation of that architecture.

### How does Codeflash verify correctness?

Codeflash verifies the correctness of the optimizations it finds by generating and running new regression tests, as well as any existing tests you may already have. Codeflash tries to ensure that your
code behaves the same way before and after the optimization.
This offers high confidence that the behavior of your code remains unchanged.

### Continuous Optimization

Because Codeflash is an automated process, you can install it as a GitHub action and have it optimize the new code on every pull request.
When Codeflash finds an optimization, it will ask you to review it. It will write a detailed explanation of the changes it made, and include all relevant info like % speed increase and proofs of correctness.

This is a great way to ensure that your code, your team's code and your AI Agent's code are optimized for performance before it causes a performance regression. We call this *Continuous Optimization*.

### Features

<!--- TODO: Add links to the relevant sections of the documentation and style the table --->

| Feature                                                                                 | Description                                                                                                                                                                                         |
|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Optimize a single function](optimizing-with-codeflash/one-function)                    | Basic unit of optimization by asking Codeflash to optimize a particular function                                                                                                                    |
| [Optimize all code in a repo](optimizing-with-codeflash/codeflash-all)                  | Codeflash discovers all functions in a repo and optimizes all of them!                                                                                                                              |
| [Optimize every new pull request](optimizing-with-codeflash/optimize-prs)               | Codeflash runs as a GitHub action and GitHub app and reviews all new code for Optimizations                                                                                                         |
| [Optimize a whole workflow by Tracing it](optimizing-with-codeflash/trace-and-optimize) | End to end optimization for all the functions called in a workflow, by tracing to collect real inputs seen during execution and ensuring correctness and performance optimization with those inputs |

## How to use these docs

On the left side of the screen you'll find the docs navigation bar.
Start by installing Codeflash, then explore the different ways of using it to optimize your code.

## Questions or Feedback?

We love feedback! If you have any questions or feedback, use the Intercom button in the lower right, join our [Discord](https://www.codeflash.ai/discord), or drop us a note at [contact@codeflash.ai](mailto:founders@codeflash.ai) - we read every message!
