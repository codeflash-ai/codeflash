# @codeflash/cli

AI-powered code optimization for JavaScript and TypeScript.

Codeflash automatically optimizes your code for better performance while maintaining correctness.

## Installation

```bash
npm install -D @codeflash/cli
# or
yarn add -D @codeflash/cli
# or
pnpm add -D @codeflash/cli
```

The installation automatically sets up:
1. **uv** - Python package manager (if not already installed)
2. **codeflash** - Python CLI for code optimization
3. **Jest runtime helpers** - Bundled test instrumentation (capture, serializer, comparator)

## Quick Start

```bash
# Optimize a specific function
npx codeflash optimize --file src/utils.ts --function fibonacci

# Optimize all functions in a file
npx codeflash optimize --file src/utils.ts

# Get help
npx codeflash --help
```

## Requirements

- **Node.js** >= 18.0.0
- **Jest** (for running tests)
- Internet connection (for AI optimization)

## How It Works

1. **Analyze**: Codeflash analyzes your code and identifies optimization opportunities
2. **Test**: Runs your existing tests to capture current behavior
3. **Optimize**: Uses AI to generate optimized versions
4. **Verify**: Runs tests again to ensure the optimization is correct
5. **Benchmark**: Measures performance improvement

## Configuration

Create a `codeflash.yaml` in your project root:

```yaml
module_root: src
tests_root: tests
```

## CI/CD

In CI environments, the postinstall script is skipped by default. Run setup manually:

```bash
npx codeflash-setup
```

Or set `CODEFLASH_SKIP_POSTINSTALL=false` to enable automatic setup.

## Troubleshooting

### uv not found

If you see "uv not found", run the setup script:

```bash
npx codeflash-setup
```

Or install uv manually:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### codeflash not in PATH

After installation, you may need to restart your terminal or run:

```bash
source ~/.bashrc  # or ~/.zshrc
```

## Links

- [Documentation](https://docs.codeflash.ai)
- [GitHub](https://github.com/codeflash-ai/codeflash)
- [Discord](https://discord.gg/codeflash)

## License

MIT
