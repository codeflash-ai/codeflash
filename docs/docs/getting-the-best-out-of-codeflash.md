---
sidebar_position: 5
---

# Getting the best out of Codeflash

Codeflash is a powerful tool; here are our recommendations, tips and tricks on getting the best out of it. We do these ourselves, so we hope you will too!

### Install the Github App and actions workflow

After you install Codeflash on an actively developed project, [installing the GitHub App](getting-started/codeflash-github-actions) and setting up the
GitHub Actions workflow will automatically optimize your code whenever new pull requests are opened. This ensures you get the best version of any changes you make to your code without any extra effort. We find that PRs are also the best time to review these changes, because the code is fresh in your mind.

### Find optimizations on your whole codebase with `codeflash --all`

If you have a lot of existing code, run [`codeflash --all`](optimizing-with-codeflash/codeflash-all) to discover and fix any
slow code in your project. Codeflash will open new pull requests for any optimizations it finds, and you can review and merge them at your own pace.

### Find and optimize bottlenecks with the Codeflash Tracer

Find the best results by running [Codeflash Tracer](optimizing-with-codeflash/trace-and-optimize) on the entry point of your script before optimizing it. The Codeflash Tracer will generate a trace file and a Replay Test file that will help Codeflash understand the behavior & inputs of your functions and generate the highest quality optimizations.

### Review the PRs Codeflash opens

We're constantly improving Codeflash and the underlying AI models it uses. The state of the art changes weekly, and you can be confident the optimizer will always use the best performing LLMs to find optimizations for your code. That said, because Codeflash uses generative AI, it's still possible that the optimized code may actually have different behavior than the original code under certain conditions. Please review all the PRs that Codeflash opens to ensure that the optimized code is correct, just as you would review any other PR opened by a team member. And don't forget to send us feedback on how we can improve Codeflash - we're always listening!