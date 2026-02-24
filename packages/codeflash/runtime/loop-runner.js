/**
 * Codeflash Loop Runner - Custom Jest Test Runner for Performance Benchmarking
 *
 * Implements BATCHED LOOPING for fair distribution across all test invocations:
 *
 *   Batch 1: Test1(5 loops) → Test2(5 loops) → Test3(5 loops) → ...
 *   Batch 2: Test1(5 loops) → Test2(5 loops) → Test3(5 loops) → ...
 *   ...until time budget exhausted
 *
 * This ensures:
 *   - Fair distribution: All test invocations get equal loop counts
 *   - Batched overhead: Console.log overhead amortized over batches
 *   - Good utilization: Time budget shared across all tests
 *
 * Configuration via environment variables:
 *   CODEFLASH_PERF_LOOP_COUNT - Max loops per invocation (default: 10000)
 *   CODEFLASH_PERF_BATCH_SIZE - Loops per batch (default: 5)
 *   CODEFLASH_PERF_MIN_LOOPS - Min loops before stopping (default: 5)
 *   CODEFLASH_PERF_TARGET_DURATION_MS - Target total duration (default: 10000)
 *
 * Usage:
 *   npx jest --runner=codeflash/loop-runner
 *
 * NOTE: This runner requires jest-runner to be installed in your project.
 *       It is a Jest-specific feature and does not work with Vitest.
 *       For Vitest projects, capturePerf() does all loops internally in a single call.
 *
 * Compatibility: Works with Jest 29.x and Jest 30.x
 */

'use strict';

const { createRequire } = require('module');
const path = require('path');
const fs = require('fs');

/**
 * Recursively find jest-runner package in node_modules.
 * Works with any package manager (npm, yarn, pnpm) by searching for
 * jest-runner/package.json anywhere in the tree.
 *
 * @param {string} nodeModulesPath - Path to node_modules directory
 * @param {number} maxDepth - Maximum recursion depth (default: 5)
 * @returns {string|null} Path to jest-runner or null if not found
 */
function findJestRunnerRecursive(nodeModulesPath, maxDepth = 5) {
    function search(dir, depth) {
        if (depth > maxDepth || !fs.existsSync(dir)) return null;

        try {
            let entries = fs.readdirSync(dir, { withFileTypes: true });

            // Sort entries: prefer higher versions for jest-runner@X.Y.Z directories
            entries = entries.slice().sort((a, b) => {
                const aMatch = a.name.match(/^jest-runner@(\d+)/);
                const bMatch = b.name.match(/^jest-runner@(\d+)/);
                if (aMatch && bMatch) {
                    return parseInt(bMatch[1], 10) - parseInt(aMatch[1], 10);
                }
                return a.name.localeCompare(b.name);
            });

            for (const entry of entries) {
                if (!entry.isDirectory()) continue;

                const entryPath = path.join(dir, entry.name);

                // Found jest-runner directory - check if it's a valid package
                if (entry.name === 'jest-runner') {
                    const pkgJsonPath = path.join(entryPath, 'package.json');
                    if (fs.existsSync(pkgJsonPath)) {
                        try {
                            const pkgJson = JSON.parse(fs.readFileSync(pkgJsonPath, 'utf8'));
                            if (pkgJson.name === 'jest-runner') {
                                return entryPath;
                            }
                        } catch (e) {
                            // Ignore JSON parse errors
                        }
                    }
                }

                // Recurse into:
                // - node_modules subdirectories
                // - scoped packages (@org/pkg)
                // - hidden directories (.pnpm, .yarn, etc.)
                // - pnpm versioned directories (jest-runner@30.0.5)
                const shouldRecurse = entry.name === 'node_modules' ||
                    entry.name.startsWith('@') ||
                    entry.name === '.pnpm' || entry.name === '.yarn' ||
                    entry.name.startsWith('jest-runner@');

                if (shouldRecurse) {
                    const result = search(entryPath, depth + 1);
                    if (result) return result;
                }
            }
        } catch (e) {
            // Ignore permission errors
        }

        return null;
    }

    return search(nodeModulesPath, 0);
}

/**
 * Resolve jest-runner from the PROJECT's node_modules (not codeflash's).
 *
 * Uses recursive search to find jest-runner anywhere in node_modules,
 * working with any package manager (npm, yarn, pnpm).
 *
 * @returns {string} Path to jest-runner package
 * @throws {Error} If jest-runner cannot be found
 */
function resolveJestRunner() {
    const monorepoMarkers = ['yarn.lock', 'pnpm-workspace.yaml', 'lerna.json', 'package-lock.json'];

    // Walk up from cwd to find jest-runner, checking the project's own
    // node_modules first. In monorepos, the workspace package (cwd) may have
    // a different jest-runner version than the monorepo root. The project's
    // version takes priority since it matches the Jest config being used.
    let currentDir = process.cwd();
    const visitedDirs = new Set();

    while (currentDir !== path.dirname(currentDir)) {
        if (visitedDirs.has(currentDir)) break;
        visitedDirs.add(currentDir);

        const result = findJestRunnerRecursive(path.join(currentDir, 'node_modules'));
        if (result) return result;

        // Check if this is a workspace root - stop after this
        const isWorkspaceRoot = monorepoMarkers.some(marker =>
            fs.existsSync(path.join(currentDir, marker))
        );

        if (isWorkspaceRoot) break;
        currentDir = path.dirname(currentDir);
    }

    // Fallback: check monorepo root if Python detected one and we haven't visited it yet
    const monorepoRoot = process.env.CODEFLASH_MONOREPO_ROOT;
    if (monorepoRoot && !visitedDirs.has(monorepoRoot)) {
        const result = findJestRunnerRecursive(path.join(monorepoRoot, 'node_modules'));
        if (result) return result;
    }

    throw new Error(
        'jest-runner not found. Please install jest-runner in your project: npm install --save-dev jest-runner'
    );
}

/**
 * Jest runner components - loaded dynamically from project's node_modules.
 * This ensures we use the same version that the project uses.
 *
 * Jest 30+ uses TestRunner class with event-based architecture.
 * Jest 29 uses runTest function for direct test execution.
 */
let TestRunner;
let runTest;
let jestRunnerAvailable = false;
let jestVersion = 0;

try {
    const jestRunnerPath = resolveJestRunner();

    // Read the package.json to find the actual entry point and version
    const pkgJsonPath = path.join(jestRunnerPath, 'package.json');
    const pkgJson = JSON.parse(fs.readFileSync(pkgJsonPath, 'utf8'));

    // Require using the full path to the entry point
    const entryPoint = path.join(jestRunnerPath, pkgJson.main || 'build/index.js');
    const jestRunner = require(entryPoint);

    TestRunner = jestRunner.default || jestRunner.TestRunner;

    if (TestRunner && TestRunner.prototype && typeof TestRunner.prototype.runTests === 'function') {
        // Jest 30+ - use TestRunner class with event emitter pattern
        jestVersion = 30;
        jestRunnerAvailable = true;
    } else {
        // Try Jest 29 style import - runTest is in build/runTest.js
        try {
            const runTestPath = path.join(jestRunnerPath, 'build', 'runTest.js');
            const runTestModule = require(runTestPath);
            runTest = runTestModule.default;
            if (typeof runTest === 'function') {
                // Jest 29 - use direct runTest function
                jestVersion = 29;
                jestRunnerAvailable = true;
            }
        } catch (e29) {
            // Neither Jest 29 nor 30 style import worked
            jestRunnerAvailable = false;
        }
    }
} catch (e) {
    // try to directly import jest-runner
    try {
        const jestRunner = require('jest-runner');
        TestRunner = jestRunner.default || jestRunner.TestRunner;
        if (TestRunner && TestRunner.prototype && typeof TestRunner.prototype.runTests === 'function') {
            jestVersion = 30;
            jestRunnerAvailable = true;
        } else {
            jestRunnerAvailable = false;
        }
    } catch (e2) {
        jestRunnerAvailable = false;
    }
}

// Configuration
const PERF_LOOP_COUNT = parseInt(process.env.CODEFLASH_PERF_LOOP_COUNT || '10000', 10);
const PERF_BATCH_SIZE = parseInt(process.env.CODEFLASH_PERF_BATCH_SIZE || '10', 10);
// MAX_BATCHES = how many batches needed to reach PERF_LOOP_COUNT iterations
// Add 1 to handle any rounding, but cap at PERF_LOOP_COUNT to avoid excessive batches
const MAX_BATCHES = Math.min(Math.ceil(PERF_LOOP_COUNT / PERF_BATCH_SIZE) + 1, PERF_LOOP_COUNT);
const TARGET_DURATION_MS = parseInt(process.env.CODEFLASH_PERF_TARGET_DURATION_MS || '10000', 10);
const MIN_BATCHES = parseInt(process.env.CODEFLASH_PERF_MIN_LOOPS || '5', 10);

/**
 * Simple event emitter for Jest compatibility.
 */
class SimpleEventEmitter {
    constructor() {
        this.listeners = new Map();
    }

    on(eventName, listener) {
        if (!this.listeners.has(eventName)) {
            this.listeners.set(eventName, new Set());
        }
        this.listeners.get(eventName).add(listener);
        return () => {
            const set = this.listeners.get(eventName);
            if (set) set.delete(listener);
        };
    }

    async emit(eventName, data) {
        const set = this.listeners.get(eventName);
        if (set) {
            for (const listener of set) {
                await listener(data);
            }
        }
    }
}

/**
 * Deep copy utility.
 */
function deepCopy(obj, seen = new WeakMap()) {
    if (obj === null || typeof obj !== 'object') return obj;
    if (seen.has(obj)) return seen.get(obj);
    if (Array.isArray(obj)) {
        const copy = [];
        seen.set(obj, copy);
        for (let i = 0; i < obj.length; i++) copy[i] = deepCopy(obj[i], seen);
        return copy;
    }
    if (obj instanceof Date) return new Date(obj.getTime());
    if (obj instanceof RegExp) return new RegExp(obj.source, obj.flags);
    const copy = {};
    seen.set(obj, copy);
    for (const key of Object.keys(obj)) copy[key] = deepCopy(obj[key], seen);
    return copy;
}

/**
 * Codeflash Loop Runner with Batched Looping
 *
 * For Jest 30+, extends the TestRunner class directly.
 * For Jest 29, uses the runTest function import.
 */
class CodeflashLoopRunner {
    constructor(globalConfig, context) {
        if (!jestRunnerAvailable) {
            throw new Error(
                'codeflash/loop-runner requires jest-runner to be installed.\n' +
                'Please install it: npm install --save-dev jest-runner\n\n' +
                'If you are using Vitest, the loop-runner is not needed - ' +
                'Vitest projects use internal looping handled by capturePerf().'
            );
        }

        this._globalConfig = globalConfig;
        this._context = context || {};
        this._eventEmitter = new SimpleEventEmitter();

        // For Jest 30+, verify TestRunner is available (we create fresh instances per batch)
        if (jestVersion >= 30 && !TestRunner) {
            throw new Error(
                `Jest ${jestVersion} detected but TestRunner class not available. ` +
                `This indicates an internal error in loop-runner initialization.`
            );
        }
    }

    get supportsEventEmitters() {
        return true;
    }

    get isSerial() {
        return true;
    }

    on(eventName, listener) {
        return this._eventEmitter.on(eventName, listener);
    }

    /**
     * Run tests with batched looping for fair distribution across all test invocations.
     *
     * This implements the batched looping strategy:
     *   Batch 1: Test1(N loops) → Test2(N loops) → Test3(N loops)
     *   Batch 2: Test1(N loops) → Test2(N loops) → Test3(N loops)
     *   ...until time budget exhausted or max batches reached
     *
     * @param {Array} tests - Jest test objects to run
     * @param {Object} watcher - Jest watcher for interrupt handling
     * @param {Object} options - Jest runner options
     * @returns {Promise<void>}
     */
    async runTests(tests, watcher, ...rest) {
        const startTime = Date.now();
        let batchCount = 0;
        let hasFailure = false;
        let allConsoleOutput = '';

        // Time limit check - must use local time tracking because Jest runs tests
        // in isolated worker processes where shared state from capture.js isn't accessible
        const checkTimeLimit = () => {
            const elapsed = Date.now() - startTime;
            return elapsed >= TARGET_DURATION_MS && batchCount >= MIN_BATCHES;
        };

        // Batched looping: run all test files multiple times
        while (batchCount < MAX_BATCHES) {
            batchCount++;

            // Check time limit BEFORE each batch
            if (batchCount > MIN_BATCHES && checkTimeLimit()) {
                break;
            }

            // Check if interrupted
            if (watcher.isInterrupted()) {
                break;
            }

            // Set env var for batch number - persists across Jest module resets
            process.env.CODEFLASH_PERF_CURRENT_BATCH = String(batchCount);

            // Run all test files in this batch
            const batchResult = await this._runAllTestsOnce(tests, watcher, ...rest);
            allConsoleOutput += batchResult.consoleOutput;

            // Check time limit AFTER each batch
            if (checkTimeLimit()) {
                break;
            }
        }

        const totalTimeMs = Date.now() - startTime;

        // Output all collected console logs - this is critical for timing marker extraction
        // The console output contains the !######...######! timing markers from capturePerf
        if (allConsoleOutput) {
            process.stdout.write(allConsoleOutput);
        }
    }

    /**
     * Run all test files once (one batch).
     * Uses different approaches for Jest 29 vs Jest 30.
     */
    async _runAllTestsOnce(tests, watcher, ...args) {
        if (jestVersion >= 30) {
            return this._runAllTestsOnceJest30(tests, watcher, ...args);
        } else {
            return this._runAllTestsOnceJest29(tests, watcher);
        }
    }

    /**
     * Jest 30+ implementation - creates a fresh TestRunner for each batch to avoid
     * state corruption issues that occur when reusing runners across batches.
     */
    async _runAllTestsOnceJest30(tests, watcher, ...args) {
        let hasFailure = false;
        let allConsoleOutput = '';

        // For Jest 30, we need to collect results through event listeners
        const resultsCollector = [];

        // Create a FRESH TestRunner instance for each batch
        // Jest 30's TestRunner corrupts its internal state after running tests,
        // so we cannot reuse the same instance across multiple batches
        const batchRunner = new TestRunner(this._globalConfig, this._context);

        // Subscribe to events from the batch runner
        const unsubscribeSuccess = batchRunner.on('test-file-success', (testData) => {
            const [test, result] = testData;
            resultsCollector.push({ test, result, success: true });

            if (result && result.console && Array.isArray(result.console)) {
                allConsoleOutput += result.console.map(e => e.message || '').join('\n') + '\n';
            }

            if (result && result.numFailingTests > 0) {
                hasFailure = true;
            }

            // Forward to our event emitter
            this._eventEmitter.emit('test-file-success', testData);
        });

        const unsubscribeFailure = batchRunner.on('test-file-failure', (testData) => {
            const [test, error] = testData;
            resultsCollector.push({ test, error, success: false });
            hasFailure = true;

            // Forward to our event emitter
            this._eventEmitter.emit('test-file-failure', testData);
        });

        const unsubscribeStart = batchRunner.on('test-file-start', (testData) => {
            // Forward to our event emitter
            this._eventEmitter.emit('test-file-start', testData);
        });

        try {
            // Run tests using the fresh batch runner (always serial for benchmarking)
            await batchRunner.runTests(tests, watcher, ...args);
        } finally {
            // Cleanup subscriptions
            if (typeof unsubscribeSuccess === 'function') unsubscribeSuccess();
            if (typeof unsubscribeFailure === 'function') unsubscribeFailure();
            if (typeof unsubscribeStart === 'function') unsubscribeStart();
        }

        return { consoleOutput: allConsoleOutput, hasFailure };
    }

    /**
     * Jest 29 implementation - uses direct runTest import.
     */
    async _runAllTestsOnceJest29(tests, watcher) {
        let hasFailure = false;
        let allConsoleOutput = '';

        for (const test of tests) {
            if (watcher.isInterrupted()) break;

            const sendMessageToJest = (eventName, args) => {
                this._eventEmitter.emit(eventName, deepCopy(args));
            };

            await this._eventEmitter.emit('test-file-start', [test]);

            try {
                const result = await runTest(
                    test.path,
                    this._globalConfig,
                    test.context.config,
                    test.context.resolver,
                    this._context,
                    sendMessageToJest
                );

                if (result.console && Array.isArray(result.console)) {
                    allConsoleOutput += result.console.map(e => e.message || '').join('\n') + '\n';
                }

                if (result.numFailingTests > 0) {
                    hasFailure = true;
                }

                await this._eventEmitter.emit('test-file-success', [test, result]);
            } catch (error) {
                hasFailure = true;
                await this._eventEmitter.emit('test-file-failure', [test, error]);
            }
        }

        return { consoleOutput: allConsoleOutput, hasFailure };
    }
}

module.exports = CodeflashLoopRunner;
