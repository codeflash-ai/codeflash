#!/usr/bin/env node

/**
 * Codeflash CLI Entry Point
 *
 * This script is the main entry point for the codeflash CLI when installed via npm.
 * It delegates to the Python codeflash CLI installed in a dedicated venv.
 *
 * Usage:
 *   npx codeflash --help
 *   npx codeflash optimize --file src/utils.ts
 */

const { spawn } = require('child_process');
const fs = require('fs');
const { getCodeflashBin } = require('../scripts/paths');

/**
 * Find the codeflash binary in the dedicated venv
 */
function findCodeflash() {
  const codeflashBin = getCodeflashBin();
  if (fs.existsSync(codeflashBin)) {
    return codeflashBin;
  }
  return null;
}

/**
 * Run the codeflash CLI
 */
function runCodeflash(args) {
  const codeflashBin = findCodeflash();

  if (!codeflashBin) {
    console.error('\x1b[31mError:\x1b[0m codeflash Python CLI not found.');
    console.error('');
    console.error('Please run the setup script:');
    console.error('  npx codeflash-setup');
    console.error('');
    process.exit(1);
  }

  // Strip VIRTUAL_ENV so the venv's Python is used, not an activated one
  const env = { ...process.env };
  delete env.VIRTUAL_ENV;
  delete env.CONDA_PREFIX;
  delete env.CONDA_DEFAULT_ENV;

  const child = spawn(codeflashBin, args, {
    stdio: 'inherit',
    env,
  });

  child.on('error', (error) => {
    console.error(`\x1b[31mError:\x1b[0m Failed to run codeflash: ${error.message}`);
    process.exit(1);
  });

  child.on('exit', (code, signal) => {
    if (signal) {
      process.exit(1);
    }
    process.exit(code || 0);
  });
}

/**
 * Show setup instructions
 */
function showSetupHelp() {
  console.log(`
\x1b[36m\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
\u2551     Codeflash CLI Setup Required           \u2551
\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D\x1b[0m

The codeflash Python CLI is not installed.

\x1b[33mTo complete setup, run:\x1b[0m
  npx codeflash-setup

\x1b[36mDocumentation:\x1b[0m https://docs.codeflash.ai
`);
}

// Main
const args = process.argv.slice(2);

// Special case: setup command
if (args[0] === 'setup' || args[0] === '--setup') {
  require('../scripts/postinstall.js');
} else {
  const codeflashBin = findCodeflash();
  if (!codeflashBin && args.length === 0) {
    showSetupHelp();
    process.exit(1);
  }

  runCodeflash(args);
}
