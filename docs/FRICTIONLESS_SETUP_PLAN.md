# Frictionless Setup Implementation Plan

## Executive Summary

Transform Codeflash from a "config-required" tool to a "smart-defaults with optional config" tool, reducing setup friction while maintaining enterprise-grade configurability.

**Key Principle: Use native config files only (pyproject.toml, package.json) - no new config formats.**

---

## Current State vs Target State

| Aspect | Current State | Target State |
|--------|--------------|--------------|
| First run | Error: "Run codeflash init" | Auto-detect → Quick confirm → Run |
| Config requirement | Mandatory for Python | Optional (smart defaults) |
| Setup questions | 5-10 questions | 1 confirmation (or 0 with `--yes`) |
| Config storage | pyproject.toml / package.json | Same - native files only |
| `codeflash init` | Required for all users | Optional - for GitHub Actions setup |

---

## User Flows

### Flow 1: First-Time User (Zero Friction)
```
$ codeflash --file src/utils.py --function calculate

⚡ Welcome to Codeflash!

I auto-detected your project:
┌─────────────────────────────────────────┐
│ Language:     Python                    │
│ Module root:  src/                      │
│ Test runner:  pytest                    │
│ Formatter:    ruff (from pyproject.toml)│
│ Ignoring:     __pycache__, .venv, dist  │
└─────────────────────────────────────────┘

? Proceed with these settings? [Y/n/customize]
> Y

✅ Settings saved to codeflash.yaml
🔑 Enter API key: cf-xxxxx
✅ Running optimization...
```

### Flow 2: Subsequent Runs (No Prompts)
```
$ codeflash --file src/utils.py --function calculate

⚡ Running optimization...
```

### Flow 3: Enterprise Setup (`codeflash init`)
```
$ codeflash init

⚡ Codeflash Enterprise Setup

[Full wizard with all options]
[GitHub Actions integration]
[Team-wide config]
```

### Flow 4: CI/CD (Config Required)
```
# In CI, config must exist - no interactive prompts
$ codeflash --all

# If no config: clear error with instructions
```

---

## Architecture

### New Components

```
codeflash/
├── setup/                          # NEW: Setup & detection module
│   ├── __init__.py
│   ├── detector.py                 # Universal auto-detection engine
│   ├── first_run.py                # First-run experience handler
│   ├── config_schema.py            # Universal config schema (Pydantic)
│   └── config_writer.py            # Write config to codeflash.yaml
│
├── code_utils/
│   ├── config_parser.py            # MODIFY: Add codeflash.yaml support
│   └── config_js.py                # KEEP: JS-specific detection
│
├── cli_cmds/
│   ├── cmd_init.py                 # MODIFY: Enterprise-focused
│   └── init_javascript.py          # DEPRECATE: Merge into setup/
│
└── main.py                         # MODIFY: Add first-run check
```

### Config File Strategy (Native Files)

Use **native config files** for each language ecosystem:

| Language | Config File | Section |
|----------|-------------|---------|
| Python | `pyproject.toml` | `[tool.codeflash]` |
| JavaScript | `package.json` | `{ "codeflash": {} }` |
| TypeScript | `package.json` | `{ "codeflash": {} }` |
| Rust (future) | `Cargo.toml` | `[package.metadata.codeflash]` |
| Go (future) | `codeflash.yaml` | Root level (Go has no standard) |

#### Python Config (pyproject.toml)
```toml
[tool.codeflash]
module-root = "src"
tests-root = "tests"
ignore-paths = ["vendor/", "migrations/"]
formatter-cmds = ["ruff check --fix $file", "ruff format $file"]
# Optional
git-remote = "origin"
disable-telemetry = false
benchmarks-root = "benchmarks"
```

#### JavaScript/TypeScript Config (package.json)
```json
{
  "codeflash": {
    "moduleRoot": "src",
    "ignorePaths": ["dist/", "coverage/"],
    "formatterCmds": ["npx prettier --write $file"]
  }
}
```

### Why Native Files?

1. **No extra file clutter** - Uses existing config files
2. **Ecosystem conventions** - Developers expect config in standard locations
3. **No tracking needed** - Project root always has pyproject.toml or package.json
4. **Tool discovery** - Other tools already know where to look

### Config Discovery Flow

```
find_project_and_config():
    1. Walk up from current directory
    2. Find project root (.git, package.json, pyproject.toml)
    3. Detect language from project root files
    4. Read codeflash config from appropriate native file
    5. If no codeflash section → first run experience
```

---

## Implementation Phases

## Phase 1: Universal Auto-Detection Engine
**Goal**: Create a single detection engine that works for all languages

### Task 1.1: Create Detection Module Structure
- [ ] Create `codeflash/setup/` directory
- [ ] Create `codeflash/setup/__init__.py`
- [ ] Create `codeflash/setup/detector.py`

### Task 1.2: Implement Universal Project Detector
**File**: `codeflash/setup/detector.py`
```python
@dataclass
class DetectedProject:
    language: str                    # python | javascript | typescript
    project_root: Path
    module_root: Path
    tests_root: Path | None
    test_runner: str                 # pytest | jest | vitest | mocha
    formatter: list[str] | None
    ignore_paths: list[str]
    confidence: float                # 0.0 - 1.0

def detect_project(path: Path | None = None) -> DetectedProject:
    """Auto-detect all project settings."""
    ...
```

### Task 1.3: Implement Language Detection
- [ ] Detect from file extensions count
- [ ] Detect from config files (pyproject.toml, package.json, tsconfig.json)
- [ ] Detect from lockfiles (poetry.lock, yarn.lock, etc.)
- [ ] Return confidence score

### Task 1.4: Implement Module Root Detection
- [ ] Python: Look for `__init__.py`, src/ convention, pyproject.toml name
- [ ] JS/TS: Look for package.json exports/main, src/ convention
- [ ] Generic: Find directory with most source files

### Task 1.5: Implement Tests Root Detection
- [ ] Look for tests/, test/, __tests__/ directories
- [ ] Look for *_test.py, *.test.js patterns
- [ ] Check pytest.ini, jest.config.js locations

### Task 1.6: Implement Test Runner Detection
- [ ] Python: Check for pytest, unittest in dependencies/config
- [ ] JS/TS: Check for jest, vitest, mocha in devDependencies
- [ ] Parse test scripts in package.json

### Task 1.7: Implement Formatter Detection
- [ ] Python: Check ruff.toml, pyproject.toml [tool.ruff], black config
- [ ] JS/TS: Check .prettierrc, .eslintrc, devDependencies
- [ ] Return formatter commands list

### Task 1.8: Implement Ignore Paths Detection
- [ ] Parse .gitignore
- [ ] Add language-specific defaults (node_modules, __pycache__, etc.)
- [ ] Merge with any existing config

---

## Phase 2: Universal Config Schema
**Goal**: Single config format that works for all languages

### Task 2.1: Create Config Schema
**File**: `codeflash/setup/config_schema.py`
```python
from pydantic import BaseModel

class CodeflashConfig(BaseModel):
    version: int = 1
    language: str
    module_root: str
    tests_root: str | None = None
    ignore_paths: list[str] = []
    formatter_commands: list[str] = []
    test_runner: str | None = None
    git_remote: str = "origin"
    disable_telemetry: bool = False
    benchmarks_root: str | None = None
```

### Task 2.2: Create Config Writer
**File**: `codeflash/setup/config_writer.py`
- [ ] Write to codeflash.yaml (primary)
- [ ] Optionally write to pyproject.toml (Python projects)
- [ ] Optionally write to package.json (JS/TS projects)

### Task 2.3: Update Config Parser
**File**: `codeflash/code_utils/config_parser.py`
- [ ] Add codeflash.yaml as primary config source
- [ ] Maintain backward compatibility with pyproject.toml
- [ ] Maintain backward compatibility with package.json
- [ ] Priority: codeflash.yaml > pyproject.toml > package.json

### Task 2.4: Create Config Finder
- [ ] `find_config_file()` - Search for any valid config
- [ ] Walk up directory tree to find project root
- [ ] Return config file path and type

---

## Phase 3: First-Run Experience
**Goal**: Seamless first-run with auto-detection and quick confirm

### Task 3.1: Create First-Run Handler
**File**: `codeflash/setup/first_run.py`
```python
def is_first_run() -> bool:
    """Check if this is first run (no config exists)."""
    ...

def handle_first_run(args: Namespace) -> Namespace:
    """Handle first-run experience with auto-detection."""
    ...
```

### Task 3.2: Implement First-Run Detection
- [ ] Check for codeflash.yaml in project root
- [ ] Check for [tool.codeflash] in pyproject.toml
- [ ] Check for codeflash section in package.json
- [ ] Return True if none found

### Task 3.3: Implement First-Run UI
- [ ] Show welcome message
- [ ] Display auto-detected settings in table
- [ ] Quick confirm prompt: Y/n/customize
- [ ] Handle `--yes` flag to skip prompt

### Task 3.4: Implement API Key Prompt (First Run Only)
- [ ] Check for CODEFLASH_API_KEY env var
- [ ] If missing, prompt for API key
- [ ] Offer OAuth login option
- [ ] Save to shell rc file

### Task 3.5: Implement Config Auto-Save
- [ ] After confirmation, save config to codeflash.yaml
- [ ] Show saved location
- [ ] Continue with optimization

### Task 3.6: Integrate with Main Entry Point
**File**: `codeflash/main.py`
- [ ] Check `is_first_run()` before loading config
- [ ] If first run, call `handle_first_run()`
- [ ] If not first run, proceed normally

---

## Phase 4: Refactor `codeflash init` for Enterprise
**Goal**: Make init command focused on enterprise/CI setup

### Task 4.1: Simplify Init Command
**File**: `codeflash/cli_cmds/cmd_init.py`
- [ ] Remove redundant questions (auto-detect instead)
- [ ] Focus on: API key, GitHub Actions, advanced overrides
- [ ] Add `--enterprise` flag for full wizard

### Task 4.2: Create Enterprise Wizard
- [ ] All current questions available
- [ ] Team-wide config options
- [ ] GitHub Actions setup
- [ ] CI/CD validation mode

### Task 4.3: Merge JS/TS Init into Main Init
- [ ] Remove `init_javascript.py` as separate module
- [ ] Use universal detector for all languages
- [ ] Same flow regardless of language

### Task 4.4: Update Help Text
- [ ] `codeflash init` - Quick setup with auto-detection
- [ ] `codeflash init --enterprise` - Full enterprise wizard
- [ ] Document that init is optional for most users

---

## Phase 5: CLI Updates
**Goal**: Support new flow with appropriate flags

### Task 5.1: Add New CLI Flags
**File**: `codeflash/cli_cmds/cli.py`
```python
# New flags
--yes, -y           # Skip confirmation prompts
--no-save           # Don't save config (one-off run)
--show-config       # Show detected/current config and exit
--reset-config      # Delete saved config, re-detect
```

### Task 5.2: Update Main Entry Point
**File**: `codeflash/main.py`
- [ ] Add first-run check
- [ ] Handle --yes flag
- [ ] Handle --no-save flag
- [ ] Handle --show-config flag

### Task 5.3: CI/CD Mode Detection
- [ ] Detect CI environment (CI=true, GITHUB_ACTIONS, etc.)
- [ ] In CI: Require config, no interactive prompts
- [ ] Show clear error if config missing in CI

### Task 5.4: Update Error Messages
- [ ] Replace "Run codeflash init" with helpful auto-detection message
- [ ] Show what was detected vs what's missing
- [ ] Suggest specific fixes

---

## Phase 6: Backend Updates (If Needed)
**Goal**: Ensure backend supports new flow

### Task 6.1: Review API Requirements
- [ ] Check if API needs config sent inline
- [ ] Verify API key validation flow
- [ ] Check telemetry expectations

### Task 6.2: Update Telemetry Events
- [ ] Add "first_run_auto_detect" event
- [ ] Add "config_saved" event
- [ ] Track detection confidence scores

### Task 6.3: API Key Validation
- [ ] Validate API key before first optimization
- [ ] Clear error message for invalid keys
- [ ] Support OAuth flow

---

## Phase 7: Testing & Documentation
**Goal**: Ensure reliability and clear documentation

### Task 7.1: Unit Tests
- [ ] Test detector.py for all languages
- [ ] Test config_schema.py validation
- [ ] Test config_writer.py output
- [ ] Test first_run.py flow

### Task 7.2: Integration Tests
- [ ] Test first-run on Python project
- [ ] Test first-run on JS project
- [ ] Test first-run on TS project
- [ ] Test CI mode (no interactive)
- [ ] Test --yes flag

### Task 7.3: E2E Tests
- [ ] Full flow: fresh project → first run → optimization
- [ ] Full flow: existing config → optimization
- [ ] Full flow: codeflash init → GitHub Actions

### Task 7.4: Update Documentation
- [ ] Update README with new quick start
- [ ] Update init command docs
- [ ] Add config file reference
- [ ] Add migration guide from old config

---

## Migration Strategy

### For Existing Users
1. Existing pyproject.toml/package.json configs continue to work
2. On first run after update, offer to migrate to codeflash.yaml
3. No breaking changes - backward compatible

### For New Users
1. First run auto-detects everything
2. Quick confirm saves to codeflash.yaml
3. Optional: run `codeflash init` for enterprise setup

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Time to first optimization | 5-10 min (init required) | < 1 min |
| Setup questions asked | 5-10 | 1 (confirm) or 0 (--yes) |
| Config file required | Always | Optional |
| Languages supported uniformly | 2 (Python, JS) | All (universal flow) |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Auto-detection wrong | Show detected values, allow override |
| Breaking existing configs | Full backward compatibility |
| CI/CD breaks | Detect CI mode, require explicit config |
| API key exposure | Never log API keys, use env vars |

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `codeflash/setup/__init__.py` | CREATE | New module |
| `codeflash/setup/detector.py` | CREATE | Universal detection engine |
| `codeflash/setup/first_run.py` | CREATE | First-run experience |
| `codeflash/setup/config_schema.py` | CREATE | Pydantic config model |
| `codeflash/setup/config_writer.py` | CREATE | Config file writer |
| `codeflash/main.py` | MODIFY | Add first-run check |
| `codeflash/cli_cmds/cli.py` | MODIFY | Add new flags |
| `codeflash/cli_cmds/cmd_init.py` | MODIFY | Enterprise focus |
| `codeflash/code_utils/config_parser.py` | MODIFY | Add codeflash.yaml |
| `codeflash/cli_cmds/init_javascript.py` | DEPRECATE | Merge into setup/ |

---

## Timeline Estimate

| Phase | Tasks | Complexity |
|-------|-------|------------|
| Phase 1: Detection Engine | 8 tasks | Medium |
| Phase 2: Config Schema | 4 tasks | Low |
| Phase 3: First-Run Experience | 6 tasks | Medium |
| Phase 4: Refactor Init | 4 tasks | Medium |
| Phase 5: CLI Updates | 4 tasks | Low |
| Phase 6: Backend Updates | 3 tasks | Low-Medium |
| Phase 7: Testing & Docs | 4 tasks | Medium |

---

## Next Steps

1. Review and approve this plan
2. Create GitHub issues for each phase
3. Start with Phase 1 (Detection Engine) as foundation
4. Iterate based on feedback

---

*Plan created: 2024*
*Author: Claude (Product Manager & Engineering Lead)*