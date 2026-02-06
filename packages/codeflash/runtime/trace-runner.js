#!/usr/bin/env node
/**
 * Codeflash Trace Runner
 *
 * Entry point script that runs JavaScript/TypeScript code with function tracing enabled.
 * This script:
 * 1. Registers Babel with the tracer plugin for AST transformation
 * 2. Sets up environment variables for tracing configuration
 * 3. Runs the user's script, tests, or module
 *
 * Usage:
 *   # Run a script with tracing
 *   node trace-runner.js script.js
 *
 *   # Run tests with tracing (Jest)
 *   node trace-runner.js --jest -- --testPathPattern=mytest
 *
 *   # Run tests with tracing (Vitest)
 *   node trace-runner.js --vitest -- --run
 *
 *   # Run with specific functions to trace
 *   node trace-runner.js --functions='["myFunc","otherFunc"]' script.js
 *
 * Environment Variables (also settable via command line):
 *   CODEFLASH_TRACE_DB - Path to SQLite database for storing traces
 *   CODEFLASH_PROJECT_ROOT - Project root for relative path calculation
 *   CODEFLASH_FUNCTIONS - JSON array of functions to trace
 *   CODEFLASH_MAX_FUNCTION_COUNT - Maximum traces per function (default: 256)
 *   CODEFLASH_TRACER_TIMEOUT - Timeout in seconds for tracing
 *
 * For ESM (ECMAScript modules), use the loader flag:
 *   node --loader ./esm-loader.mjs trace-runner.js script.mjs
 */

'use strict';

const path = require('path');
const fs = require('fs');

// ============================================================================
// ARGUMENT PARSING
// ============================================================================

function parseArgs(args) {
    const config = {
        traceDb: process.env.CODEFLASH_TRACE_DB || path.join(process.cwd(), 'codeflash.trace.sqlite'),
        projectRoot: process.env.CODEFLASH_PROJECT_ROOT || process.cwd(),
        functions: process.env.CODEFLASH_FUNCTIONS || null,
        maxFunctionCount: process.env.CODEFLASH_MAX_FUNCTION_COUNT || '256',
        tracerTimeout: process.env.CODEFLASH_TRACER_TIMEOUT || null,
        traceFiles: process.env.CODEFLASH_TRACE_FILES || null,
        traceExclude: process.env.CODEFLASH_TRACE_EXCLUDE || null,
        jest: false,
        vitest: false,
        module: false,
        script: null,
        scriptArgs: [],
    };

    let i = 0;
    while (i < args.length) {
        const arg = args[i];

        if (arg === '--trace-db') {
            config.traceDb = args[++i];
        } else if (arg.startsWith('--trace-db=')) {
            config.traceDb = arg.split('=')[1];
        } else if (arg === '--project-root') {
            config.projectRoot = args[++i];
        } else if (arg.startsWith('--project-root=')) {
            config.projectRoot = arg.split('=')[1];
        } else if (arg === '--functions') {
            config.functions = args[++i];
        } else if (arg.startsWith('--functions=')) {
            config.functions = arg.split('=')[1];
        } else if (arg === '--max-function-count') {
            config.maxFunctionCount = args[++i];
        } else if (arg.startsWith('--max-function-count=')) {
            config.maxFunctionCount = arg.split('=')[1];
        } else if (arg === '--timeout') {
            config.tracerTimeout = args[++i];
        } else if (arg.startsWith('--timeout=')) {
            config.tracerTimeout = arg.split('=')[1];
        } else if (arg === '--trace-files') {
            config.traceFiles = args[++i];
        } else if (arg.startsWith('--trace-files=')) {
            config.traceFiles = arg.split('=')[1];
        } else if (arg === '--trace-exclude') {
            config.traceExclude = args[++i];
        } else if (arg.startsWith('--trace-exclude=')) {
            config.traceExclude = arg.split('=')[1];
        } else if (arg === '--jest') {
            config.jest = true;
        } else if (arg === '--vitest') {
            config.vitest = true;
        } else if (arg === '-m' || arg === '--module') {
            config.module = true;
        } else if (arg === '--') {
            // Everything after -- is passed to the script/test runner
            config.scriptArgs = args.slice(i + 1);
            break;
        } else if (arg === '--help' || arg === '-h') {
            printHelp();
            process.exit(0);
        } else if (!arg.startsWith('-')) {
            // First non-flag argument is the script
            config.script = arg;
            config.scriptArgs = args.slice(i + 1);
            break;
        }

        i++;
    }

    return config;
}

function printHelp() {
    console.log(`
Codeflash Trace Runner - JavaScript Function Tracing

Usage:
  trace-runner [options] <script> [script-args...]
  trace-runner [options] --jest -- [jest-args...]
  trace-runner [options] --vitest -- [vitest-args...]

Options:
  --trace-db <path>           Path to SQLite database for traces (default: ./codeflash.trace.sqlite)
  --project-root <path>       Project root directory (default: cwd)
  --functions <json>          JSON array of functions to trace (traces all if not set)
  --max-function-count <n>    Maximum traces per function (default: 256)
  --timeout <seconds>         Timeout for tracing
  --trace-files <json>        JSON array of file patterns to trace
  --trace-exclude <json>      JSON array of patterns to exclude from tracing
  --jest                      Run with Jest test framework
  --vitest                    Run with Vitest test framework
  -m, --module                Run a module (like python -m)
  -h, --help                  Show this help message

Examples:
  # Trace a script
  trace-runner --functions='["processData"]' ./src/main.js

  # Trace Jest tests
  trace-runner --jest --functions='["myFunc"]' -- --testPathPattern=mytest

  # Trace Vitest tests
  trace-runner --vitest -- --run

Environment Variables:
  CODEFLASH_TRACE_DB          Path to SQLite database
  CODEFLASH_PROJECT_ROOT      Project root directory
  CODEFLASH_FUNCTIONS         JSON array of functions to trace
  CODEFLASH_MAX_FUNCTION_COUNT Maximum traces per function
  CODEFLASH_TRACER_TIMEOUT    Timeout in seconds
`);
}

// ============================================================================
// BABEL REGISTRATION
// ============================================================================

function registerBabel(config) {
    // Set environment variables before loading Babel
    process.env.CODEFLASH_TRACE_DB = config.traceDb;
    process.env.CODEFLASH_PROJECT_ROOT = config.projectRoot;
    process.env.CODEFLASH_MAX_FUNCTION_COUNT = config.maxFunctionCount;

    if (config.functions) {
        process.env.CODEFLASH_FUNCTIONS = config.functions;
    }
    if (config.tracerTimeout) {
        process.env.CODEFLASH_TRACER_TIMEOUT = config.tracerTimeout;
    }
    if (config.traceFiles) {
        process.env.CODEFLASH_TRACE_FILES = config.traceFiles;
    }
    if (config.traceExclude) {
        process.env.CODEFLASH_TRACE_EXCLUDE = config.traceExclude;
    }

    // Try to find @babel/register
    let babelRegister;
    try {
        babelRegister = require('@babel/register');
    } catch (e) {
        console.error('[codeflash] Error: @babel/register is required for tracing.');
        console.error('Install it with: npm install --save-dev @babel/register @babel/core');
        process.exit(1);
    }

    // Get the path to our Babel plugin
    const pluginPath = path.join(__dirname, 'babel-tracer-plugin.js');

    // Configure Babel
    const babelConfig = {
        // Use our tracer plugin
        plugins: [pluginPath],

        // Compile only project files, not node_modules
        ignore: [/node_modules/],

        // Only compile files in project root
        only: [config.projectRoot],

        // Don't look for .babelrc files - we provide all config
        babelrc: false,
        configFile: false,

        // Support TypeScript and modern JS
        presets: [],

        // Enable source maps for better error messages
        sourceMaps: 'inline',

        // Cache for faster repeated runs
        cache: true,

        // File extensions to process
        extensions: ['.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs'],
    };

    // Try to add TypeScript support if available
    try {
        require.resolve('@babel/preset-typescript');
        babelConfig.presets.push('@babel/preset-typescript');
    } catch (e) {
        // TypeScript preset not available, skip
    }

    // Try to add modern JS support
    try {
        require.resolve('@babel/preset-env');
        babelConfig.presets.push(['@babel/preset-env', { targets: { node: 'current' } }]);
    } catch (e) {
        // preset-env not available, skip
    }

    // Register Babel
    babelRegister(babelConfig);

    console.log(`[codeflash] Tracing enabled. Output: ${config.traceDb}`);
    if (config.functions) {
        console.log(`[codeflash] Tracing functions: ${config.functions}`);
    }
}

// ============================================================================
// SCRIPT EXECUTION
// ============================================================================

function runScript(config) {
    if (!config.script) {
        console.error('[codeflash] Error: No script specified');
        printHelp();
        process.exit(1);
    }

    // Resolve script path
    const scriptPath = path.resolve(config.projectRoot, config.script);

    if (!fs.existsSync(scriptPath)) {
        console.error(`[codeflash] Error: Script not found: ${scriptPath}`);
        process.exit(1);
    }

    // Update process.argv for the script
    process.argv = [process.argv[0], scriptPath, ...config.scriptArgs];

    // Run the script
    require(scriptPath);
}

function runJest(config) {
    // Find Jest
    let jestPath;
    try {
        jestPath = require.resolve('jest');
    } catch (e) {
        console.error('[codeflash] Error: Jest not found. Install it with: npm install --save-dev jest');
        process.exit(1);
    }

    // Get Jest CLI path
    const jestCli = path.join(path.dirname(jestPath), 'cli');

    // Update process.argv for Jest
    process.argv = [process.argv[0], 'jest', ...config.scriptArgs];

    // Run Jest
    const jest = require(jestCli);
    jest.run();
}

function runVitest(config) {
    // Vitest needs special handling as it's ESM-first
    // We'll spawn it as a subprocess with our loader

    const { spawn } = require('child_process');

    const args = [
        '--experimental-vm-modules',
        require.resolve('vitest/vitest.mjs'),
        'run',
        ...config.scriptArgs,
    ];

    const env = {
        ...process.env,
        CODEFLASH_TRACE_DB: config.traceDb,
        CODEFLASH_PROJECT_ROOT: config.projectRoot,
        CODEFLASH_MAX_FUNCTION_COUNT: config.maxFunctionCount,
    };

    if (config.functions) {
        env.CODEFLASH_FUNCTIONS = config.functions;
    }
    if (config.tracerTimeout) {
        env.CODEFLASH_TRACER_TIMEOUT = config.tracerTimeout;
    }

    console.log('[codeflash] Running Vitest with tracing...');
    console.log('[codeflash] Note: ESM tracing requires additional setup. See documentation.');

    const child = spawn(process.execPath, args, {
        env,
        stdio: 'inherit',
        cwd: config.projectRoot,
    });

    child.on('exit', (code) => {
        process.exit(code || 0);
    });
}

function runModule(config) {
    if (!config.script) {
        console.error('[codeflash] Error: No module specified');
        printHelp();
        process.exit(1);
    }

    // For module mode, we resolve the module from the project root
    const modulePath = require.resolve(config.script, { paths: [config.projectRoot] });

    // Update process.argv
    process.argv = [process.argv[0], modulePath, ...config.scriptArgs];

    // Run the module
    require(modulePath);
}

// ============================================================================
// MAIN
// ============================================================================

function main() {
    // Parse command line arguments (skip node and script name)
    const args = process.argv.slice(2);
    const config = parseArgs(args);

    // Register Babel with tracer plugin
    registerBabel(config);

    // Initialize the tracer
    const tracer = require('./tracer');
    tracer.init(config.traceDb, config.projectRoot);

    // Run based on mode
    if (config.jest) {
        runJest(config);
    } else if (config.vitest) {
        runVitest(config);
    } else if (config.module) {
        runModule(config);
    } else {
        runScript(config);
    }
}

main();
