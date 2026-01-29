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
 */

'use strict';

const { createRequire } = require('module');
const path = require('path');

const jestRunnerPath = require.resolve('jest-runner');
const internalRequire = createRequire(jestRunnerPath);
const runTest = internalRequire('./runTest').default;

// Configuration
const MAX_BATCHES = parseInt(process.env.CODEFLASH_PERF_LOOP_COUNT || '10000', 10);
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
