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
 */

'use strict';

const { createRequire } = require('module');
const path = require('path');
const fs = require('fs');

/**
 * Resolve jest-runner with monorepo support.
 * Uses CODEFLASH_MONOREPO_ROOT environment variable if available,
 * otherwise walks up the directory tree looking for node_modules/jest-runner.
 */
function resolveJestRunner() {
    // Try standard resolution first (works in simple projects)
    try {
        return require.resolve('jest-runner');
    } catch (e) {
        // Standard resolution failed - try monorepo-aware resolution
    }

    // If Python detected a monorepo root, check there first
    const monorepoRoot = process.env.CODEFLASH_MONOREPO_ROOT;
    if (monorepoRoot) {
        const jestRunnerPath = path.join(monorepoRoot, 'node_modules', 'jest-runner');
        if (fs.existsSync(jestRunnerPath)) {
            const packageJsonPath = path.join(jestRunnerPath, 'package.json');
            if (fs.existsSync(packageJsonPath)) {
                return jestRunnerPath;
            }
        }
    }

    // Fallback: Walk up from cwd looking for node_modules/jest-runner
    const monorepoMarkers = ['yarn.lock', 'pnpm-workspace.yaml', 'lerna.json', 'package-lock.json'];
    let currentDir = process.cwd();
    const visitedDirs = new Set();

    while (currentDir !== path.dirname(currentDir)) {
        // Avoid infinite loops
        if (visitedDirs.has(currentDir)) break;
        visitedDirs.add(currentDir);

        // Try node_modules/jest-runner at this level
        const jestRunnerPath = path.join(currentDir, 'node_modules', 'jest-runner');
        if (fs.existsSync(jestRunnerPath)) {
            const packageJsonPath = path.join(jestRunnerPath, 'package.json');
            if (fs.existsSync(packageJsonPath)) {
                return jestRunnerPath;
            }
        }

        // Check if this is a workspace root (has monorepo markers)
        const isWorkspaceRoot = monorepoMarkers.some(marker =>
            fs.existsSync(path.join(currentDir, marker))
        );

        if (isWorkspaceRoot) {
            // Found workspace root but no jest-runner - stop searching
            break;
        }

        currentDir = path.dirname(currentDir);
    }

    throw new Error('jest-runner not found');
}

// Try to load jest-runner - it's a peer dependency that must be installed by the user
let runTest;
let jestRunnerAvailable = false;

try {
    const jestRunnerPath = resolveJestRunner();
    const internalRequire = createRequire(jestRunnerPath);
    runTest = internalRequire('./runTest').default;
    jestRunnerAvailable = true;
} catch (e) {
    // jest-runner not installed - this is expected for Vitest projects
    // The runner will throw a helpful error if someone tries to use it without jest-runner
    jestRunnerAvailable = false;
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
 */
class CodeflashLoopRunner {
    constructor(globalConfig, context) {
        if (!jestRunnerAvailable) {
            throw new Error(
                'codeflash/loop-runner requires jest-runner to be installed.\n' +
                'Please install it: npm install --save-dev jest-runner\n\n' +
                'If you are using Vitest, the loop-runner is not needed - ' +
                'Vitest projects use external looping handled by the Python runner.'
            );
        }
        this._globalConfig = globalConfig;
        this._context = context || {};
        this._eventEmitter = new SimpleEventEmitter();
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
     * Run tests with batched looping for fair distribution.
     */
    async runTests(tests, watcher, options) {
        const startTime = Date.now();
        let batchCount = 0;
        let hasFailure = false;
        let allConsoleOutput = '';

        // Import shared state functions from capture module
        // We need to do this dynamically since the module may be reloaded
        let checkSharedTimeLimit;
        let incrementBatch;
        try {
            const capture = require('codeflash');
            checkSharedTimeLimit = capture.checkSharedTimeLimit;
            incrementBatch = capture.incrementBatch;
        } catch (e) {
            // Fallback if codeflash module not available
            checkSharedTimeLimit = () => {
                const elapsed = Date.now() - startTime;
                return elapsed >= TARGET_DURATION_MS && batchCount >= MIN_BATCHES;
            };
            incrementBatch = () => {};
        }

        // Batched looping: run all test files multiple times
        while (batchCount < MAX_BATCHES) {
            batchCount++;

            // Check time limit BEFORE each batch
            if (batchCount > MIN_BATCHES && checkSharedTimeLimit()) {
                break;
            }

            // Check if interrupted
            if (watcher.isInterrupted()) {
                break;
            }

            // Increment batch counter in shared state and set env var
            // The env var persists across Jest module resets, ensuring continuous loop indices
            incrementBatch();
            process.env.CODEFLASH_PERF_CURRENT_BATCH = String(batchCount);

            // Run all test files in this batch
            const batchResult = await this._runAllTestsOnce(tests, watcher);
            allConsoleOutput += batchResult.consoleOutput;

            if (batchResult.hasFailure) {
                hasFailure = true;
                break;
            }

            // Check time limit AFTER each batch
            if (checkSharedTimeLimit()) {
                break;
            }
        }

        const totalTimeMs = Date.now() - startTime;

        // Output all collected console logs - this is critical for timing marker extraction
        // The console output contains the !######...######! timing markers from capturePerf
        if (allConsoleOutput) {
            process.stdout.write(allConsoleOutput);
        }

        console.log(`[codeflash] Batched runner completed: ${batchCount} batches, ${tests.length} test files, ${totalTimeMs}ms total`);
    }

    /**
     * Run all test files once (one batch).
     */
    async _runAllTestsOnce(tests, watcher) {
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
