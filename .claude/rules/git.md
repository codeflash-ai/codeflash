# Git

## Commits

- Never commit, amend, or push without explicit permission
- Don't commit intermediate states — wait until the full implementation is complete and approved
- Always create a new branch from `main` — never commit directly to `main`
- Conventional format: `fix:`, `feat:`, `refactor:`, `docs:`, `test:`, `chore:`
- First line: imperative verb + what changed, under 72 characters
- Body for *why*, not *what* — the diff shows what changed
- One purpose per commit: a bug fix, a new function, a refactor — not all three
- A commit that adds a function also adds its tests and exports — that's one logical change

## Sizing

- Too small: renaming a variable in one commit, updating its references in another
- Right size: adding a function with its tests, `__init__` export, and usage update
- Too large: implementing an entire subsystem in one commit

## Pre-commit / Pre-push

- Pre-commit: Run `uv run prek` before committing
- Pre-push: Run `uv run prek run --from-ref origin/<base>` to check all changed files against the PR base

## Pull Requests

- PR titles use conventional format
- Keep the PR body short and to the point
- If related to a Linear issue, include `CF-#` in the body
- Branch naming: `cf-#-title` (lowercase, hyphenated)

## Branch Hygiene

- Delete feature branches locally after merging (`git branch -d <branch>`)
- Use `/clean_gone` to prune local branches whose remote tracking branch has been deleted
