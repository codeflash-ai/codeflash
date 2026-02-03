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

// Try to load jest-runner from the PROJECT's node_modules, not from codeflash package
// This ensures we use the same version of jest-runner that the project uses
let TestRunner;
let runTest;
let jestRunnerAvailable = false;
let jestVersion = 0;

try {
    // Resolve jest-runner from the current working directory (project root)
    // This is important because the codeflash package may bundle a different version
    const projectRoot = process.cwd();
    const projectRequire = createRequire(path.join(projectRoot, 'node_modules', 'package.json'));

    let jestRunnerPath;
    try {
        // First try to resolve from project's node_modules
        jestRunnerPath = projectRequire.resolve('jest-runner');
    } catch (e) {
        // Fall back to default resolution (codeflash's bundled version)
        jestRunnerPath = require.resolve('jest-runner');
    }

    const internalRequire = createRequire(jestRunnerPath);

    // Try to get the TestRunner class (Jest 30+)
    const jestRunner = internalRequire(jestRunnerPath);
    TestRunner = jestRunner.default || jestRunner.TestRunner;

    if (TestRunner && TestRunner.prototype && typeof TestRunner.prototype.runTests === 'function') {
        // Jest 30+ - use TestRunner class
        jestVersion = 30;
        jestRunnerAvailable = true;
    } else {
        // Try Jest 29 style import
        try {
            runTest = internalRequire('./runTest').default;
            if (typeof runTest === 'function') {
                jestVersion = 29;
                jestRunnerAvailable = true;
            }
        } catch (e29) {
            // Neither Jest 29 nor 30 style import worked
            jestRunnerAvailable = false;
        }
    }
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
                'Vitest projects use external looping handled by the Python runner.'
            );
        }
        this._globalConfig = globalConfig;
        this._context = context || {};
        this._eventEmitter = new SimpleEventEmitter();

        // For Jest 30+, create an instance of the base TestRunner for delegation
        if (jestVersion >= 30 && TestRunner) {
            this._baseRunner = new TestRunner(globalConfig, context);
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
     * Run tests with batched looping for fair distribution.
     */
    async runTests(tests, watcher, options) {
        const startTime = Date.now();
        let batchCount = 0;
        let hasFailure = false;
        let allConsoleOutput = '';

        // Time limit check - must use local time tracking because Jest runs tests
        // in worker processes, so shared state from capture.js isn't accessible here
        const checkTimeLimit = () => {
            const elapsed = Date.now() - startTime;
            return elapsed >= TARGET_DURATION_MS && batchCount >= MIN_BATCHES;
        };

        // Batched looping: run all test files multiple times
        while (batchCount < MAX_BATCHES) {
            batchCount++;

            // Check time limit BEFORE each batch
            if (batchCount > MIN_BATCHES && checkTimeLimit()) {
                console.log(`[codeflash] Time limit reached after ${batchCount - 1} batches (${Date.now() - startTime}ms elapsed)`);
                break;
            }

            // Check if interrupted
            if (watcher.isInterrupted()) {
                break;
            }

            // Set env var for batch number - persists across Jest module resets
            process.env.CODEFLASH_PERF_CURRENT_BATCH = String(batchCount);

            // Run all test files in this batch
            const batchResult = await this._runAllTestsOnce(tests, watcher, options);
            allConsoleOutput += batchResult.consoleOutput;

            if (batchResult.hasFailure) {
                hasFailure = true;
                break;
            }

            // Check time limit AFTER each batch
            if (checkTimeLimit()) {
                console.log(`[codeflash] Time limit reached after ${batchCount} batches (${Date.now() - startTime}ms elapsed)`);
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
     * Uses different approaches for Jest 29 vs Jest 30.
     */
    async _runAllTestsOnce(tests, watcher, options) {
        if (jestVersion >= 30) {
            return this._runAllTestsOnceJest30(tests, watcher, options);
        } else {
            return this._runAllTestsOnceJest29(tests, watcher);
        }
    }

    /**
     * Jest 30+ implementation - delegates to base TestRunner and collects results.
     */
    async _runAllTestsOnceJest30(tests, watcher, options) {
        let hasFailure = false;
        let allConsoleOutput = '';

        // For Jest 30, we need to collect results through event listeners
        const resultsCollector = [];

        // Subscribe to events from the base runner
        const unsubscribeSuccess = this._baseRunner.on('test-file-success', (testData) => {
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

        const unsubscribeFailure = this._baseRunner.on('test-file-failure', (testData) => {
            const [test, error] = testData;
            resultsCollector.push({ test, error, success: false });
            hasFailure = true;

            // Forward to our event emitter
            this._eventEmitter.emit('test-file-failure', testData);
        });

        const unsubscribeStart = this._baseRunner.on('test-file-start', (testData) => {
            // Forward to our event emitter
            this._eventEmitter.emit('test-file-start', testData);
        });

        try {
            // Run tests using the base runner (always serial for benchmarking)
            await this._baseRunner.runTests(tests, watcher, { ...options, serial: true });
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
