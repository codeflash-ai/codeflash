---
sidebar_position: 1
slug: /
---
# Introduction
Welcome to the Codeflash documentation!

## What is Codeflash?

Welcome! Codeflash is an AI performance optimizer for Python code.
Codeflash speeds up Python code by figuring out the best way to rewrite a particular function, while verifying the behavior of the code is unchanged.

The optimizations Codeflash finds are generally better algorithms, opportunities to remove wasteful compute, better logic, and utilization of more efficient library methods.

### How does Codeflash verify correctness?

Codeflash verifies the correctness of the optimizations it finds by generating and running new regression tests, as well as any existing tests you may already have.
This offers additional confidence that the behavior of your code remains unchanged.

### Continuous Optimization

Because Codeflash is an automated process, you can install it as a GitHub action and have it run on every pull request made to your codebase.
When Codeflash finds an optimization, it will ask you to review it. It will write a detailed explanation of the changes it made, and include all relevant info like % speed increase and proofs of correctness.

Having Codeflash installed on your Github repository gives you the peace of mind that your code is always written optimally. We call it *Continuous Optimization*.

### Features

<!--- TODO: Add links to the relevant sections of the documentation and style the table --->

| Feature                                                                                 | Description                                                                                                                                                                                         |
|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Optimize a single function](optimizing-with-codeflash/one-function)                    | Basic unit of optimization by asking Codeflash to optimize a particular function                                                                                                                    |
| [Optimize all code in a repo](optimizing-with-codeflash/codeflash-all)                  | Codeflash discovers all functions in a repo and optimizes all of them!                                                                                                                              |
| [Optimize every new pull request](optimizing-with-codeflash/optimize-prs)               | Codeflash runs as a GitHub action and GitHub app and reviews all new code for Optimizations                                                                                                         |
| [Optimize a whole workflow by Tracing it](optimizing-with-codeflash/trace-and-optimize) | End to end optimization for all the functions called in a workflow, by tracing to collect real inputs seen during execution and ensuring correctness and performance optimization with those inputs |
| Correctness Verification                                                                | The way Codeflash gains high confidence that the newly generated optimization has the same behavior as the originally written function.                                                             |
| Performance Measurement                                                                 | Measuring execution times on a set of inputs to estimate runtime performance.                                                                                                                       |


## How to use these docs

On the left side of the screen you'll find the docs navigation bar.
Start by installing Codeflash, then explore the different ways of using it to optimize your code.

## Questions or Feedback?

We love feedback! If you have any questions or feedback, use the Intercom button in the lower right, or drop us a note at [founders@codeflash.ai](mailto:founders@codeflash.ai) - we read every message!
