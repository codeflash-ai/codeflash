# codeflash

AI-powered code performance optimization for JavaScript and TypeScript.

## Installation

```bash
npm install -g codeflash
# or
npx codeflash
```

## Quick Start

1. Get your API key from [codeflash.ai](https://codeflash.ai)

2. Set your API key:
```bash
export CODEFLASH_API_KEY=your-api-key
```

3. Optimize a function:
```bash
codeflash --file src/utils.ts --function slowFunction
```

## Usage

```bash
# Optimize a specific function
codeflash --file <path> --function <name>

# Optimize all functions in a directory
codeflash --all src/

# Initialize GitHub Actions workflow
codeflash init-actions

# Verify setup
codeflash --verify-setup
```

## Requirements

- Node.js >= 16.0.0
- A codeflash API key

## Supported Platforms

- Linux (x64, arm64)
- macOS (x64, arm64)
- Windows (x64)

## Documentation

See [codeflash.ai/docs](https://codeflash.ai/docs) for full documentation.

## License

BSL-1.1
