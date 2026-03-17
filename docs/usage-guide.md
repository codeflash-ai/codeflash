# Usage Guide

## The `/optimize` skill

`/optimize` is the primary command. It spawns a background optimizer agent that runs the codeflash CLI on your code.

### Syntax

```
/optimize [file] [function] [flags]
```

### Examples

| Command | Effect |
|---------|--------|
| `/optimize` | Let codeflash detect changed files automatically |
| `/optimize src/utils.py` | Optimize all functions in `src/utils.py` |
| `/optimize src/utils.py my_function` | Optimize only `my_function` in that file |
| `/optimize --all` | Optimize the entire project |
| `/optimize src/utils.py --no-pr` | Optimize without creating a PR |
| `/optimize src/utils.py --effort high` | Set optimization effort level to high |

Flags can be combined: `/optimize src/utils.py my_function --no-pr --effort high`

### What happens behind the scenes

1. The skill (defined in `skills/optimize/SKILL.md`) forks context and spawns the **optimizer agent**
2. The agent locates your project config (`pyproject.toml` or `package.json`)
3. It verifies the codeflash CLI is installed and the project is configured
4. It runs `codeflash --subagent` as a **background task** with a 10-minute timeout
5. You're notified when optimization completes with results

The agent has up to **15 turns** to complete its work (install codeflash, configure the project, run optimization).

## The `/setup` command

`/setup` configures auto-permissions so codeflash runs without prompting.

### What it does

1. Finds `.claude/settings.json` in your project root
2. Checks if `Bash(*codeflash*)` is already in `permissions.allow`
3. If not, adds it (creating the file and directory if needed)
4. Preserves any existing settings

Running `/setup` multiple times is safe -- it's idempotent. If permissions are already configured, it reports "No changes needed."

## Automatic post-commit suggestions

After every Claude response (the **Stop** hook), the plugin checks whether you committed Python, JS, or TS files during the current session. If so, it suggests running `/optimize`.

### How commit detection works

1. The hook determines the session start time from the transcript file's creation timestamp
2. It queries `git log --after=@<session_start>` for commits touching `*.py`, `*.js`, `*.ts`, `*.jsx`, `*.tsx` files
3. It deduplicates so the same commits don't trigger suggestions twice
4. If new commits are found, it blocks Claude's stop and injects a suggestion

The suggestion varies depending on the project state:

| State | Suggestion |
|-------|------------|
| Configured + installed | Run `codeflash --subagent` in the background |
| Configured, not installed | Install codeflash first, then run |
| Not configured | Auto-discover config, write it, then run |
| No venv (Python) | Create venv, install codeflash, configure, then run |

If `Bash(*codeflash*)` is not yet in `.claude/settings.json`, the suggestion also includes adding it for auto-permissions.

## Python-specific workflow

For Python projects, the optimizer agent:

1. Checks for an active virtual environment (`$VIRTUAL_ENV`)
2. If none, searches for `.venv` or `venv` directories in the project dir and repo root
3. Verifies `codeflash` is installed in the venv (`$VIRTUAL_ENV/bin/codeflash --version`)
4. Reads `[tool.codeflash]` from `pyproject.toml` for configuration
5. Runs: `source $VIRTUAL_ENV/bin/activate && codeflash --subagent [flags]`

The agent also checks `formatter-cmds` in the config and verifies formatters are installed.

## JS/TS-specific workflow

For JavaScript/TypeScript projects, the optimizer agent:

1. Checks codeflash is available via `npx codeflash --version`
2. Reads the `"codeflash"` key from `package.json` for configuration
3. Always runs from the project root (the directory containing `package.json`)
4. Runs: `npx codeflash --subagent [flags]`

No virtual environment is needed -- JS/TS projects use `npx`/`npm` directly.
