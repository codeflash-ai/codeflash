# Git Conventions

- **Always create a new branch from `main`** — never commit directly to `main` or reuse an existing feature branch for unrelated changes
- Use conventional commit format: `fix:`, `feat:`, `refactor:`, `docs:`, `test:`, `chore:`
- Keep commits atomic — one logical change per commit
- Commit message body should be concise (1-2 sentences max)
- PR titles should also use conventional format
- Branch naming: `cf-#-title` (lowercase, hyphenated) where `#` is the Linear issue number
- If related to a Linear issue, include `CF-#` in the PR body
