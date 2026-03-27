#!/usr/bin/env node

/**
 * Codeflash CLI Postinstall Script
 *
 * This script runs after `npm install codeflash` and:
 * 1. Checks if uv (Python package manager) is installed
 * 2. If not, installs uv automatically
 * 3. Creates a dedicated venv and installs the Python codeflash CLI into it
 *
 * The codeflash Python CLI is installed into an isolated venv at:
 * - Linux:   ~/.cache/codeflash/venv/
 * - macOS:   ~/Library/Caches/codeflash/venv/
 * - Windows: %LOCALAPPDATA%\codeflash\venv\
 */

const { execSync, spawnSync } = require('child_process');
const os = require('os');
const fs = require('fs');
const { getCacheDir, getVenvDir, getCodeflashBin, getUvPath } = require('./paths');

// Clean environment without VIRTUAL_ENV so uv doesn't target an activated venv
const cleanEnv = (() => {
  const env = { ...process.env };
  delete env.VIRTUAL_ENV;
  delete env.CONDA_PREFIX;
  delete env.CONDA_DEFAULT_ENV;
  return env;
})();

// ANSI color codes for pretty output
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
  dim: '\x1b[2m',
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function logStep(step, message) {
  console.log(`${colors.cyan}[${step}]${colors.reset} ${message}`);
}

function logSuccess(message) {
  console.log(`${colors.green}\u2713${colors.reset} ${message}`);
}

function logWarning(message) {
  console.log(`${colors.yellow}\u26A0${colors.reset} ${message}`);
}

function logError(message) {
  console.error(`${colors.red}\u2717${colors.reset} ${message}`);
}

/**
 * Check if a command exists in PATH
 */
function commandExists(command) {
  try {
    const result = spawnSync(command, ['--version'], {
      stdio: 'ignore',
      shell: true,
    });
    return result.status === 0;
  } catch {
    return false;
  }
}

/**
 * Install uv using the official installer
 */
function installUv() {
  const platform = os.platform();

  logStep('1/3', 'Installing uv (Python package manager)...');

  try {
    if (platform === 'win32') {
      execSync(
        'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"',
        { stdio: 'inherit', shell: true }
      );
    } else {
      execSync(
        'curl -LsSf https://astral.sh/uv/install.sh | sh',
        { stdio: 'inherit', shell: true }
      );
    }
    logSuccess('uv installed successfully');
    return true;
  } catch (error) {
    logError(`Failed to install uv: ${error.message}`);
    return false;
  }
}

/**
 * Check if git is available
 */
function hasGit() {
  try {
    const result = spawnSync('git', ['--version'], {
      stdio: 'ignore',
      shell: true,
    });
    return result.status === 0;
  } catch {
    return false;
  }
}

/**
 * Create the codeflash venv and install the Python CLI into it.
 *
 * Installation priority:
 * 1. GitHub main branch (if git available) - gets latest features
 * 2. PyPI (fallback) - stable release
 */
function installCodeflash(uvBin) {
  logStep('2/3', 'Installing codeflash Python CLI...');

  const venvDir = getVenvDir();
  const cacheDir = getCacheDir();

  // Ensure cache directory exists
  fs.mkdirSync(cacheDir, { recursive: true });

  // Create the venv (or reuse existing)
  try {
    execSync(`"${uvBin}" venv --python python3.12 --clear "${venvDir}"`, {
      stdio: 'inherit',
      shell: true,
      env: cleanEnv,
    });
    logSuccess(`venv created at ${venvDir}`);
  } catch (error) {
    logError(`Failed to create venv: ${error.message}`);
    return false;
  }

  const GITHUB_REPO = 'git+https://github.com/codeflash-ai/codeflash.git';

  // Priority 1: Install from GitHub (latest features, requires git)
  if (hasGit()) {
    try {
      execSync(`"${uvBin}" pip install --python "${venvDir}" "${GITHUB_REPO}"`, {
        stdio: 'inherit',
        shell: true,
        env: cleanEnv,
      });
      logSuccess('codeflash CLI installed from GitHub (latest)');
      return true;
    } catch (error) {
      logWarning(`GitHub installation failed: ${error.message}`);
      logWarning('Falling back to PyPI...');
    }
  } else {
    logWarning('Git not found, installing from PyPI...');
  }

  // Priority 2: Install from PyPI (stable release fallback)
  try {
    execSync(`"${uvBin}" pip install --python "${venvDir}" codeflash`, {
      stdio: 'inherit',
      shell: true,
      env: cleanEnv,
    });
    logSuccess('codeflash CLI installed from PyPI');
    return true;
  } catch (error) {
    logError(`Failed to install codeflash: ${error.message}`);
    return false;
  }
}

/**
 * Verify the installation works
 */
function verifyInstallation() {
  const codeflashBin = getCodeflashBin();
  try {
    if (!fs.existsSync(codeflashBin)) {
      return false;
    }
    const result = spawnSync(codeflashBin, ['--version'], {
      encoding: 'utf8',
      shell: true,
      env: cleanEnv,
    });

    if (result.status === 0) {
      const version = result.stdout.trim() || result.stderr.trim();
      logSuccess(`Verified: codeflash ${version}`);
      return true;
    }
  } catch {
    // Ignore verification errors
  }
  return false;
}

/**
 * Main installation flow
 */
async function main() {
  console.log('');
  log('\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557', 'cyan');
  log('\u2551     Codeflash CLI Installation             \u2551', 'cyan');
  log('\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D', 'cyan');
  console.log('');

  // Check if running in CI or with --ignore-scripts
  if (process.env.CI || process.env.CODEFLASH_SKIP_POSTINSTALL) {
    logWarning('Skipping postinstall in CI environment');
    logWarning('Run `npx codeflash-setup` to complete installation');
    return;
  }

  let uvBin = getUvPath();

  // Step 1: Check/install uv
  if (commandExists('uv')) {
    logSuccess('uv is already installed');
    uvBin = 'uv';
  } else if (fs.existsSync(uvBin)) {
    logSuccess('uv found at ' + uvBin);
  } else {
    if (!installUv()) {
      logError('Failed to install uv. Please install it manually:');
      logError('  curl -LsSf https://astral.sh/uv/install.sh | sh');
      process.exit(1);
    }

    if (!fs.existsSync(uvBin) && !commandExists('uv')) {
      logError('uv installation completed but binary not found');
      logError('Please restart your terminal and run: npx codeflash-setup');
      process.exit(1);
    }
  }

  // Use 'uv' if it's in PATH, otherwise use full path
  if (commandExists('uv')) {
    uvBin = 'uv';
  }

  // Step 2: Install codeflash Python CLI into dedicated venv
  if (!installCodeflash(uvBin)) {
    logError('Failed to install codeflash CLI');
    logError('You can try manually: uv pip install codeflash');
    process.exit(1);
  }

  // Verify installation
  console.log('');
  logStep('3/3', 'Verifying installation...');
  verifyInstallation();

  // Print success message
  console.log('');
  log('\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550', 'green');
  logSuccess('Codeflash installation complete!');
  log('\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550', 'green');
  console.log('');
  log('Get started:', 'cyan');
  console.log('  npx codeflash --help');
  console.log('  npx codeflash optimize --file src/utils.ts');
  console.log('');
  log('Documentation: https://docs.codeflash.ai', 'dim');
  console.log('');
}

// Run the installer
main().catch((error) => {
  logError(`Installation failed: ${error.message}`);
  process.exit(1);
});
