#!/usr/bin/env node

/**
 * Codeflash Setup Script
 *
 * Run this manually if the postinstall script was skipped (e.g., in CI)
 * or if you need to reinstall the Python CLI.
 *
 * Usage:
 *   npx codeflash-setup
 */

require('../scripts/postinstall.js');
