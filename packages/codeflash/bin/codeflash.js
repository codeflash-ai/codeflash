#!/usr/bin/env node

/**
 * Codeflash CLI Entry Point
 *
 * This script is the main entry point for the codeflash CLI when installed via npm.
 * It delegates to the Python codeflash CLI installed via uv.
 *
 * Usage:
 *   npx codeflash --help
 *   npx codeflash optimize --file src/utils.ts
 */

const { spawn, spawnSync } = require('child_process');
const os = require('os');
const path = require('path');
const fs = require('fs');

/**
 * Find the uv binary
 */
function findUv() {
  const homeDir = os.homedir();
  const platform = os.platform();

  // Check the default uv installation location first
  const uvPath = platform === 'win32'
    ? path.join(homeDir, '.local', 'bin', 'uv.exe')
    : path.join(homeDir, '.local', 'bin', 'uv');

  if (fs.existsSync(uvPath)) {
    return uvPath;
  }

  // Try to find uv in PATH by checking if it exists
  try {
    const uvInPath = spawnSync('uv', ['--version'], {
      stdio: 'ignore',
    });
    if (uvInPath.status === 0) {
      return 'uv';
    }
  } catch {
    // uv not in PATH
  }

  return null;
}

/**
 * Run the codeflash CLI via uv
 */
function runCodeflash(args) {
  const uvBin = findUv();

  if (!uvBin) {
    console.error('\x1b[31mError:\x1b[0m uv not found.');
    console.error('');
    console.error('Please run the setup script:');
    console.error('  npx codeflash-setup');
    console.error('');
    console.error('Or install uv manually:');
    console.error('  curl -LsSf https://astral.sh/uv/install.sh | sh');
    process.exit(1);
  }

  // Use uv tool run to execute codeflash
  const child = spawn(uvBin, ['tool', 'run', 'codeflash', ...args], {
    stdio: 'inherit',
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
\x1b[36m╔════════════════════════════════════════════╗
║     Codeflash CLI Setup Required           ║
╚════════════════════════════════════════════╝\x1b[0m

The codeflash Python CLI is not installed.

\x1b[33mTo complete setup, run:\x1b[0m
  npx codeflash-setup

\x1b[33mOr install manually:\x1b[0m
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv tool install codeflash

\x1b[36mDocumentation:\x1b[0m https://docs.codeflash.ai
`);
}

// Main
const args = process.argv.slice(2);

// Special case: setup command
if (args[0] === 'setup' || args[0] === '--setup') {
  require('../scripts/postinstall.js');
} else {
  // Check if codeflash is installed
  const uvBin = findUv();
  if (uvBin) {
    const check = spawnSync(uvBin, ['tool', 'run', 'codeflash', '--version'], {
      stdio: 'ignore',
    });

    if (check.status !== 0 && args.length === 0) {
      showSetupHelp();
      process.exit(1);
    }
  } else if (args.length === 0) {
    showSetupHelp();
    process.exit(1);
  }

  runCodeflash(args);
}
