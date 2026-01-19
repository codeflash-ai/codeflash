# Codeflash Installation Guide

I need to install and configure Codeflash for my project to enable AI-powered performance optimization.

## Objective

Install Codeflash CLI and configure it for either Python or JavaScript/TypeScript project with proper test data serialization and GitHub integration.

## Success Condition

Codeflash is successfully installed, configured with API key, connected to GitHub, and can run optimization commands without errors.

## TODO

1. Install Codeflash CLI for your language
2. Run initialization and configuration
3. Generate and configure API key
4. Install GitHub App
5. Verify installation

## Installation Steps

### For Python Projects

**Prerequisites:**
- Python 3.9 or above
- Virtual environment activated
- Project dependencies installed

**Commands:**

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install Codeflash
pip install codeflash

# Or as dev dependency with modern tools
uv add --dev codeflash
# or
poetry add codeflash@latest --group dev

# Initialize configuration
cd /path/to/project/root
codeflash init
```

**Configuration prompts:**
1. Enter Codeflash API key
2. Specify Python module to optimize (e.g., `my_module`)
3. Specify test location (e.g., `tests/`)
4. Select code formatter (black/ruff/other/disabled)
5. Select git remote for PRs
6. Opt-in to anonymous usage data
7. Install GitHub app
8. Install GitHub actions

### For JavaScript/TypeScript Projects

**Prerequisites:**
- Node.js 16 or above
- Package manager (npm/yarn/pnpm/bun)
- Project dependencies installed

**Commands:**

```bash
# Verify Node.js version
node --version  # Should be v16.0.0+

# Install Codeflash as dev dependency
npm install --save-dev codeflash
# or
yarn add --dev codeflash
# or
pnpm add --save-dev codeflash
# or
bun add --dev codeflash

# Or install globally
npm install -g codeflash

# Initialize configuration
cd /path/to/project/root
codeflash init
```

**Configuration prompts:**
1. Enter Codeflash API key
2. Specify JavaScript/TypeScript module (e.g., `src/`)
3. Specify test location (e.g., `tests/`, `__tests__/`, `*.test.js`)
4. Select test framework (jest/vitest/mocha/ava/other)
5. Select code formatter (prettier/eslint/biome/disabled)
6. Select git remote for PRs
7. Opt-in to anonymous usage data
8. Install GitHub app
9. Install GitHub actions

**Serialization Strategy:**
JavaScript/TypeScript projects use V8 serialization automatically for optimal test data capture:
- Binary storage format
- Perfect type preservation (Date, Map, Set, TypedArrays, BigInt, etc.)
- 2-3x faster than alternatives
- Framework agnostic (React, Vue, Angular, Svelte, etc.)

### API Key Generation

**Steps:**
1. Visit https://app.codeflash.ai/
2. Sign up with GitHub account (free tier available)
3. Navigate to https://app.codeflash.ai/app/apikeys
4. Generate new API key
5. Copy and paste when prompted during `codeflash init`

### GitHub App Installation

**Steps:**
1. Visit https://github.com/apps/codeflash-ai/installations/select_target
2. Select the repository to install Codeflash
3. Grant necessary permissions
4. Confirm installation

## Verification

**Test the installation:**

```bash
# Python - Optimize single function
codeflash --file path/to/file.py --function function_name

# JavaScript/TypeScript - Optimize single function
codeflash --file path/to/file.js --function functionName

# Optimize entire codebase (both languages)
codeflash --all

# Verbose output for debugging
codeflash optimize --verbose
```

## Configuration Files

**Python:**
- Configuration stored in `pyproject.toml`
- Serialization: SQLite + pickle

**JavaScript/TypeScript:**
- Configuration stored in `codeflash.config.js`
- Serialization: V8 binary format

## Framework Support

**Python:**
- All Python projects with pytest, unittest, or similar

**JavaScript/TypeScript:**
- Frontend: React, Vue, Angular, Svelte, Solid.js
- Backend: Express, NestJS, Fastify, Koa, Hono
- Testing: Jest, Vitest, Mocha, AVA, Playwright, Cypress
- Runtimes: Node.js (✅ Recommended), Bun/Deno (Coming soon)

## Execute Installation

Run the commands above for your specific language and answer the configuration prompts to complete Codeflash installation.