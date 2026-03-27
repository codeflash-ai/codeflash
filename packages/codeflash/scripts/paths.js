/**
 * Shared path utilities for codeflash installation.
 *
 * Provides OS-specific cache directory and binary paths used by
 * both the postinstall script and the CLI entry point.
 */

const os = require('os');
const path = require('path');

/**
 * Get the OS-appropriate cache directory for codeflash.
 *
 * - Linux:   ~/.cache/codeflash
 * - macOS:   ~/Library/Caches/codeflash
 * - Windows: %LOCALAPPDATA%\codeflash
 */
function getCacheDir() {
  const platform = os.platform();
  const homeDir = os.homedir();

  if (platform === 'win32') {
    return path.join(process.env.LOCALAPPDATA || path.join(homeDir, 'AppData', 'Local'), 'codeflash');
  }
  if (platform === 'darwin') {
    return path.join(homeDir, 'Library', 'Caches', 'codeflash');
  }
  // Linux / other Unix
  return path.join(process.env.XDG_CACHE_HOME || path.join(homeDir, '.cache'), 'codeflash');
}

/**
 * Get the path to the venv directory inside the cache.
 */
function getVenvDir() {
  return path.join(getCacheDir(), 'venv');
}

/**
 * Get the path to the codeflash binary inside the venv.
 */
function getCodeflashBin() {
  const venvDir = getVenvDir();
  if (os.platform() === 'win32') {
    return path.join(venvDir, 'Scripts', 'codeflash.exe');
  }
  return path.join(venvDir, 'bin', 'codeflash');
}

/**
 * Get the path to the uv binary.
 */
function getUvPath() {
  const platform = os.platform();
  const homeDir = os.homedir();

  if (platform === 'win32') {
    return path.join(homeDir, '.local', 'bin', 'uv.exe');
  }
  return path.join(homeDir, '.local', 'bin', 'uv');
}

module.exports = { getCacheDir, getVenvDir, getCodeflashBin, getUvPath };
