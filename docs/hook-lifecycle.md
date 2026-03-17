# Hook Lifecycle

Deep dive into `scripts/suggest-optimize.sh` -- the Stop hook that detects commits and suggests optimizations.

## When the hook fires

The hook fires on **every Claude stop** (every time Claude finishes a response). This is configured in `hooks/hooks.json` with a `*` matcher, meaning it runs unconditionally regardless of what Claude just did.

## Decision tree

The hook evaluates a series of conditions and takes the first matching exit path:

```
1. stop_hook_active == true?
   YES -> exit 0 (allow stop, prevent infinite loop)

2. Not inside a git repo?
   YES -> exit 0

3. No transcript_path or file doesn't exist?
   YES -> exit 0

4. Session start time unavailable?
   YES -> exit 0

5. No commits with .py/.js/.ts/.jsx/.tsx files since session start?
   YES -> exit 0

6. Commit hash set already seen (in codeflash-seen)?
   YES -> exit 0

7. JS/TS project with JS changes?
   7a. Not configured -> block: set up config (+ install if needed)
   7b. Configured, not installed -> block: install codeflash
   7c. Configured + installed -> block: run codeflash

8. No Python changes?
   YES -> exit 0

9. Python project, no venv found?
   YES -> block: create venv, install, configure, run

10. Python project with venv:
    10a. Not configured -> block: set up config (+ install if needed)
    10b. Configured, not installed -> block: install codeflash
    10c. Configured + installed -> block: run codeflash
```

Every `block` decision also appends auto-allow instructions if `Bash(*codeflash*)` is not yet in `.claude/settings.json`.

## Infinite loop prevention

When the hook blocks Claude's stop with a suggestion, Claude acts on it (e.g., runs codeflash). When Claude finishes that response, the hook fires again. To prevent an infinite loop:

1. Claude sets `stop_hook_active: true` in the hook input when it's responding to a previous hook block
2. The hook checks this flag first and immediately exits if true

This means the hook only triggers once per "natural" Claude stop, not on stops caused by responding to hook suggestions.

## Session boundary detection

The hook needs to know when the current session started to find only commits made during this session.

1. It reads `transcript_path` from the hook input JSON
2. It gets the transcript file's **birth time** (creation timestamp) using `stat`:
   - **macOS**: `stat -f %B <file>` (birth time)
   - **Linux**: `stat -c %W <file>` (birth time), falls back to `stat -c %Y` (modification time) if birth time is unavailable
3. This timestamp becomes `SESSION_START`, used in `git log --after=@$SESSION_START`

## Commit detection

```bash
git log --after="@$SESSION_START" --name-only --diff-filter=ACMR \
  --pretty=format: -- '*.py' '*.js' '*.ts' '*.jsx' '*.tsx'
```

- `--after=@$SESSION_START` -- only commits after session start (Unix timestamp)
- `--diff-filter=ACMR` -- Added, Copied, Modified, Renamed files only
- `--pretty=format:` -- suppress commit metadata, show only file names
- File patterns filter to Python and JS/TS extensions

The results are sorted and deduplicated. The hook also determines which language families have changes (`HAS_PYTHON_CHANGES`, `HAS_JS_CHANGES`) by grepping the file list for extension patterns.

## Deduplication

The hook prevents suggesting optimization for the same set of commits twice:

1. It computes the commit hashes of all matching commits:
   ```bash
   git log --after="@$SESSION_START" --pretty=format:%H -- '*.py' '*.js' '*.ts' '*.jsx' '*.tsx'
   ```
2. It hashes the full list with SHA-256:
   ```bash
   ... | shasum -a 256 | cut -d' ' -f1
   ```
3. It checks this hash against `$TRANSCRIPT_DIR/codeflash-seen`
4. If found, the hook exits (already processed)
5. If not found, appends the hash and continues

The seen-marker file lives in the transcript directory, so it's scoped to the current session/project.

## Project detection (`detect_project`)

The `detect_project` function walks from `$PWD` upward to `$REPO_ROOT`:

1. At each directory level, check for `pyproject.toml` first, then `package.json`
2. The **first** config file found determines the project type
3. It records:
   - `PROJECT_TYPE`: `"python"` or `"js"`
   - `PROJECT_DIR`: directory containing the config file
   - `PROJECT_CONFIG_PATH`: full path to the config file
   - `PROJECT_CONFIGURED`: `"true"` if codeflash config section exists

For Python, it checks for `[tool.codeflash]` in `pyproject.toml`. For JS/TS, it checks for a `"codeflash"` key in `package.json` using `jq`.

The walk stops at `$REPO_ROOT` -- it never searches above the git repository root.

## Auto-allow suggestion

Every `block` decision checks whether codeflash is already auto-allowed:

```bash
SETTINGS_JSON="$REPO_ROOT/.claude/settings.json"
jq -e '.permissions.allow // [] | any(test("codeflash"))' "$SETTINGS_JSON"
```

If no matching entry exists, the block message appends instructions to add `Bash(*codeflash*)` to `permissions.allow`. This means after the first optimization, future runs won't need permission prompts.

## Debug logging

The hook writes all debug output to `/tmp/codeflash-hook-debug.log`:

```bash
LOGFILE="/tmp/codeflash-hook-debug.log"
exec 2>>"$LOGFILE"
set -x
```

All stderr (including bash trace output from `set -x`) is appended to this file. To debug hook issues:

```bash
tail -f /tmp/codeflash-hook-debug.log
```

The log persists across sessions and is not automatically cleaned up.
