#!/usr/bin/env node

/**
 * Codeflash CLI Postinstall Script
 *
 * This script runs after `npm install @codeflash/cli` and:
 * 1. Checks if uv (Python package manager) is installed
 * 2. If not, installs uv automatically
 * 3. Uses uv to install the Python codeflash CLI
 *
 * This approach follows the same pattern as aider and mistral-code,
 * which use uv for Python distribution.
 */

const { execSync, spawnSync } = require('child_process');
const os = require('os');
const path = require('path');
const fs = require('fs');

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
  console.log(`${colors.green}✓${colors.reset} ${message}`);
}

function logWarning(message) {
  console.log(`${colors.yellow}⚠${colors.reset} ${message}`);
}

function logError(message) {
  console.error(`${colors.red}✗${colors.reset} ${message}`);
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
 * Get the uv binary path
 * uv installs to ~/.local/bin on Unix or %USERPROFILE%\.local\bin on Windows
 */
function getUvPath() {
  const platform = os.platform();
  const homeDir = os.homedir();

  if (platform === 'win32') {
    return path.join(homeDir, '.local', 'bin', 'uv.exe');
  }
  return path.join(homeDir, '.local', 'bin', 'uv');
}

/**
 * Install uv using the official installer
 */
function installUv() {
  const platform = os.platform();

  logStep('1/3', 'Installing uv (Python package manager)...');

  try {
    if (platform === 'win32') {
      // Windows: Use PowerShell
      execSync(
        'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"',
        { stdio: 'inherit', shell: true }
      );
    } else {
      // macOS/Linux: Use curl
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
 * Install codeflash Python CLI using uv tool
 */
function installCodeflash(uvBin) {
  logStep('2/3', 'Installing codeflash Python CLI...');

  try {
    // Use uv tool install to install codeflash in an isolated environment
    // This avoids conflicts with any existing Python environments
    execSync(`"${uvBin}" tool install codeflash --force`, {
      stdio: 'inherit',
      shell: true,
    });
    logSuccess('codeflash CLI installed successfully');
    return true;
  } catch (error) {
    // If codeflash is not on PyPI yet, try installing from the local package
    logWarning('codeflash not found on PyPI, trying local installation...');
    try {
      // Try installing from the current codeflash repo if we're in development
      const cliRoot = path.resolve(__dirname, '..', '..', '..');
      const pyprojectPath = path.join(cliRoot, 'pyproject.toml');

      if (fs.existsSync(pyprojectPath)) {
        execSync(`"${uvBin}" tool install --force "${cliRoot}"`, {
          stdio: 'inherit',
          shell: true,
        });
        logSuccess('codeflash CLI installed from local source');
        return true;
      }
    } catch (localError) {
      logError(`Failed to install codeflash: ${localError.message}`);
    }
    return false;
  }
}

/**
 * Update shell configuration to include uv tools in PATH
 */
function updateShellPath(uvBin) {
  logStep('3/3', 'Updating shell configuration...');

  try {
    execSync(`"${uvBin}" tool update-shell`, {
      stdio: 'inherit',
      shell: true,
    });
    logSuccess('Shell configuration updated');
    return true;
  } catch (error) {
    logWarning(`Could not update shell: ${error.message}`);
    logWarning('You may need to add ~/.local/bin to your PATH manually');
    return true; // Non-fatal
  }
}

/**
 * Verify the installation works
 */
function verifyInstallation(uvBin) {
  try {
    const result = spawnSync(uvBin, ['tool', 'run', 'codeflash', '--version'], {
      encoding: 'utf8',
      shell: true,
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
  log('╔════════════════════════════════════════════╗', 'cyan');
  log('║     Codeflash CLI Installation             ║', 'cyan');
  log('╚════════════════════════════════════════════╝', 'cyan');
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
    uvBin = 'uv'; // Use the one in PATH
  } else if (fs.existsSync(uvBin)) {
    logSuccess('uv found at ' + uvBin);
  } else {
    if (!installUv()) {
      logError('Failed to install uv. Please install it manually:');
      logError('  curl -LsSf https://astral.sh/uv/install.sh | sh');
      process.exit(1);
    }

    // Check if uv is now available
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

  // Step 2: Install codeflash Python CLI
  if (!installCodeflash(uvBin)) {
    logError('Failed to install codeflash CLI');
    logError('You can try manually: uv tool install codeflash');
    process.exit(1);
  }

  // Step 3: Update shell PATH
  updateShellPath(uvBin);

  // Verify installation
  console.log('');
  verifyInstallation(uvBin);

  // Print success message
  console.log('');
  log('════════════════════════════════════════════', 'green');
  logSuccess('Codeflash installation complete!');
  log('════════════════════════════════════════════', 'green');
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
