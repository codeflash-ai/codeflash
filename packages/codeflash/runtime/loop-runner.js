/**
 * Codeflash Loop Runner - Custom Jest Test Runner for Performance Benchmarking
 *
 * This runner executes all tests multiple times within a single Jest process
 * to eliminate process startup overhead. It implements session-level looping
 * similar to Python's pytest_plugin behavior.
 *
 * Configuration via environment variables:
 *   CODEFLASH_LOOP_COUNT - Maximum number of loop iterations (default: 100)
 *   CODEFLASH_MIN_LOOPS - Minimum loops before stopping (default: 5)
 *   CODEFLASH_TARGET_DURATION_MS - Target total duration in ms (default: 10000)
 *   CODEFLASH_STABILITY_CHECK - Enable stability-based early stopping (default: true)
 *
 * Usage in jest.config.js:
 *   module.exports = {
 *     runner: 'codeflash/loop-runner',
 *   };
 *
 * Or via CLI:
 *   npx jest --runner=codeflash/loop-runner
 */

'use strict';

// Import runTest from jest-runner using createRequire to bypass exports restriction
// jest-runner doesn't export ./build/runTest in its package.json exports field
const { createRequire } = require('module');
const path = require('path');

// Create a require function that can access jest-runner's internals
const jestRunnerPath = require.resolve('jest-runner');
const jestRunnerDir = path.dirname(jestRunnerPath);
const internalRequire = createRequire(jestRunnerPath);

// Load the internal runTest module
const runTest = internalRequire('./runTest').default;

// Configuration from environment
const LOOP_COUNT = parseInt(process.env.CODEFLASH_LOOP_COUNT || '200', 10);
const MIN_LOOPS = parseInt(process.env.CODEFLASH_MIN_LOOPS || '5', 10);
const TARGET_DURATION_MS = parseInt(process.env.CODEFLASH_TARGET_DURATION_MS || '10000', 10);
const STABILITY_CHECK = (process.env.CODEFLASH_STABILITY_CHECK || 'true').toLowerCase() === 'true';

// Stability constants (matching Python's config_consts.py)
const STABILITY_WINDOW_SIZE = 0.35;  // 35% of estimated total loops
const STABILITY_CENTER_TOLERANCE = 0.0025;  // +/-0.25% around median
const STABILITY_SPREAD_TOLERANCE = 0.0025;  // 0.25% (max-min)/min spread

/**
 * Simple event emitter for Jest compatibility.
 * Jest runners need to emit events for test-file-start, test-file-success, test-file-failure.
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
        // Return unsubscribe function
        return () => {
            const set = this.listeners.get(eventName);
            if (set) {
                set.delete(listener);
            }
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
 * Deep copy utility to avoid memory leaks from shared references.
 * Simplified version of jest-util's deepCyclicCopy.
 */
function deepCopy(obj, seen = new WeakMap()) {
    if (obj === null || typeof obj !== 'object') {
        return obj;
    }
    if (seen.has(obj)) {
        return seen.get(obj);
    }
    if (Array.isArray(obj)) {
        const copy = [];
        seen.set(obj, copy);
        for (let i = 0; i < obj.length; i++) {
            copy[i] = deepCopy(obj[i], seen);
        }
        return copy;
    }
    if (obj instanceof Date) {
        return new Date(obj.getTime());
    }
    if (obj instanceof RegExp) {
        return new RegExp(obj.source, obj.flags);
    }
    const copy = {};
    seen.set(obj, copy);
    for (const key of Object.keys(obj)) {
        copy[key] = deepCopy(obj[key], seen);
    }
    return copy;
}

/**
 * Check if performance has stabilized.
 * Matches Python's pytest_plugin.should_stop() exactly.
 *
 * @param {number[]} runtimes - List of aggregate runtimes
 * @param {number} window - Size of the window to check
 * @param {number} minWindowSize - Minimum data points required
 * @returns {boolean} - True if performance has stabilized
 */
function shouldStopStability(runtimes, window, minWindowSize) {
    if (runtimes.length < window || runtimes.length < minWindowSize) {
        return false;
    }

    const recent = runtimes.slice(-window);
    const recentSorted = [...recent].sort((a, b) => a - b);
    const mid = Math.floor(window / 2);
    const median = window % 2 ? recentSorted[mid] : (recentSorted[mid - 1] + recentSorted[mid]) / 2;

    // 1) All recent points close to the median
    for (const r of recent) {
        if (Math.abs(r - median) / median > STABILITY_CENTER_TOLERANCE) {
            return false;
        }
    }

    // 2) Window spread is small
    const rMin = recentSorted[0];
    const rMax = recentSorted[recentSorted.length - 1];
    if (rMin === 0) {
        return false;
    }
    const spreadOk = (rMax - rMin) / rMin <= STABILITY_SPREAD_TOLERANCE;

    return spreadOk;
}

/**
 * Parse timing data from test console output.
 * Extracts timing markers printed by capturePerf.
 *
 * @param {string} stdout - Console output containing timing markers
 * @returns {Map<string, number>} - Map of test IDs to duration in nanoseconds
 */
function parseTimingFromOutput(stdout) {
    const pattern = /!######([^:]+):([^:]*):([^:]+):([^:]+):([^:]+):(\d+)######!/g;
    const timings = new Map();

    let match;
    while ((match = pattern.exec(stdout)) !== null) {
        const [, module, testClass, funcName, _loopIndex, invocationId, durationNs] = match;
        const testId = `${module}:${testClass}:${funcName}:${invocationId}`;
        timings.set(testId, parseInt(durationNs, 10));
    }

    return timings;
}

/**
 * Codeflash Loop Runner
 *
 * Custom Jest test runner that implements session-level looping
 * for performance benchmarking within a single Jest process.
 */
class CodeflashLoopRunner {
    constructor(globalConfig, context) {
        this._globalConfig = globalConfig;
        this._context = context || {};
        this._eventEmitter = new SimpleEventEmitter();
    }

    // Required: Tell Jest this runner supports event emitters
    get supportsEventEmitters() {
        return true;
    }

    // Force serial execution for consistent timing
    get isSerial() {
        return true;
    }

    /**
     * Subscribe to test events.
     * Required by Jest's EmittingTestRunner interface.
     *
     * @param {string} eventName - Event name
     * @param {Function} listener - Event listener
     * @returns {Function} - Unsubscribe function
     */
    on(eventName, listener) {
        return this._eventEmitter.on(eventName, listener);
    }

    /**
     * Run tests with session-level looping.
     *
     * @param {Array} tests - Array of test objects
     * @param {TestWatcher} watcher - Jest test watcher
     * @param {Object} options - Test runner options
     */
    async runTests(tests, watcher, options) {
        const startTime = Date.now();
        const runtimeDataByTest = new Map();
        const aggregateRuntimes = [];

        let loopCount = 0;

        // Session-level looping: run ALL tests multiple times
        while (loopCount < LOOP_COUNT) {
            loopCount++;

            // Update loop index for timing markers
            process.env.CODEFLASH_LOOP_INDEX = String(loopCount);

            // Run all test files in this iteration
            const { consoleOutput, hasFailure } = await this._runAllTestsOnce(tests, watcher);

            // Parse timing data from this iteration
            const timingData = parseTimingFromOutput(consoleOutput || '');
            for (const [testId, durationNs] of timingData) {
                if (!runtimeDataByTest.has(testId)) {
                    runtimeDataByTest.set(testId, []);
                }
                runtimeDataByTest.get(testId).push(durationNs);
            }

            // Stop if tests failed
            if (hasFailure) {
                console.log(`[codeflash] Stopping at loop ${loopCount}: test failure detected`);
                break;
            }

            // Check if interrupted
            if (watcher.isInterrupted()) {
                console.log(`[codeflash] Stopping at loop ${loopCount}: interrupted`);
                break;
            }

            // Check stopping conditions
            const elapsedMs = Date.now() - startTime;

            // Stop if reached min loops AND exceeded time limit
            if (loopCount >= MIN_LOOPS && elapsedMs >= TARGET_DURATION_MS) {
                console.log(`[codeflash] Stopping at loop ${loopCount}: time limit reached (${elapsedMs}ms >= ${TARGET_DURATION_MS}ms)`);
                break;
            }

            // Stability check
            if (STABILITY_CHECK && runtimeDataByTest.size > 0) {
                // Calculate best runtime (sum of min per test case)
                let bestRuntime = 0;
                for (const data of runtimeDataByTest.values()) {
                    if (data.length > 0) {
                        bestRuntime += Math.min(...data);
                    }
                }

                if (bestRuntime > 0) {
                    aggregateRuntimes.push(bestRuntime);

                    // Estimate window size based on loop rate
                    const elapsedNs = elapsedMs * 1e6;
                    if (elapsedNs > 0) {
                        const rate = loopCount / elapsedNs;
                        const totalTimeNs = TARGET_DURATION_MS * 1e6;
                        const estimatedTotalLoops = Math.floor(rate * totalTimeNs);
                        const windowSize = Math.round(STABILITY_WINDOW_SIZE * estimatedTotalLoops);

                        if (shouldStopStability(aggregateRuntimes, windowSize, MIN_LOOPS)) {
                            console.log(`[codeflash] Stopping at loop ${loopCount}: stability reached`);
                            break;
                        }
                    }
                }
            }
        }

        // Log summary
        const totalTimeMs = Date.now() - startTime;
        console.log(`[codeflash] Loop runner completed: ${loopCount} loops in ${totalTimeMs}ms, ${runtimeDataByTest.size} test cases tracked`);
    }

    /**
     * Run all tests once.
     *
     * @param {Array} tests - Array of test objects
     * @param {TestWatcher} watcher - Jest test watcher
     * @returns {Object} - { consoleOutput, hasFailure }
     */
    async _runAllTestsOnce(tests, watcher) {
        let hasFailure = false;
        let allConsoleOutput = '';

        for (const test of tests) {
            if (watcher.isInterrupted()) {
                break;
            }

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

                // Collect console output for timing marker parsing
                if (result.console && Array.isArray(result.console)) {
                    const output = result.console
                        .map(entry => entry.message || '')
                        .join('\n');
                    allConsoleOutput += output + '\n';
                }

                // Check for test failures
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
