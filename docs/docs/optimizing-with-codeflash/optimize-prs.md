---
sidebar_position: 3
---

# Optimize Pull Requests

Codeflash can optimize your pull requests by analyzing the changes in the pull request 
and generating optimized versions of the functions that have changed.

## How to optimize a pull request
After following the setup steps in the [Automate Code Optimization with GitHub Actions](/getting-started/codeflash-github-actions) guide,
Codeflash will automatically optimize your pull requests when they are opened.

If Codeflash finds any successful optimizations, it will comment on the pull request asking you to review the changes.

![Codeflash PR Comment](/img/review-comment.png)

Codeflash can ask you to review the changes in two ways:
### Opening a dependent pull request
Codeflash will open a new pull request with the optimized code. 
You can review the changes in this pull request, make changes if you want, and merge it if you are satisfied with the optimizations.
The changes will be merged back into the original pull request as a new commit.

![Codeflash PR Review](/img/dependent-pr.png)
### Reviewing the changes in the original pull request
If the suggested changes are small and only affect the modified lines, Codeflash will suggest the changes in the original pull request itself.
You can choose to accept or reject the changes directly in the original pull request.
The changes can be added to a batch of changes in the original pull request as a new commit.

![Codeflash PR Review](/img/code-suggestion.png)
