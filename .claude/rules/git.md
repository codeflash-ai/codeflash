# Git Commits & Pull Requests

## Commits
- Never commit, amend, or push without explicit permission
- Don't commit intermediate states — wait until the full implementation is complete, reviewed, and explicitly approved before committing. If the user corrects direction mid-implementation, incorporate the correction before any commit
- Always create a new branch from `main` before starting any new work — never commit directly to `main` or reuse an existing feature branch for unrelated changes
- Use conventional commit format: `fix:`, `feat:`, `refactor:`, `docs:`, `test:`, `chore:`
- Keep commits atomic - one logical change per commit
- Commit message body should be concise (1-2 sentences max)
- Merge for simple syncs, rebase when branches have diverged significantly
- When committing to an external/third-party repo, follow that repo's own conventions for versioning, changelog, and CI
- Pre-commit: Run `uv run prek` before committing — fix any issues before creating the commit
- Pre-push: Run `uv run prek run --from-ref origin/<base>` to check all changed files against the PR base — this matches CI behavior and catches issues that per-commit prek misses. To detect the base branch: `gh pr view --json baseRefName -q .baseRefName 2>/dev/null || echo main`

## Pull Requests
- PR titles should use conventional format
- Keep the PR body short and straight to the point
- If related to a Linear issue, include `CF-#` in the body
- Branch naming: `cf-#-title` (lowercase, hyphenated), no other prefixes/suffixes
