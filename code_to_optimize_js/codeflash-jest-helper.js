/**
 * Codeflash Jest Helper - Unified Test Instrumentation
 *
 * This module provides a unified approach to instrumenting JavaScript tests
 * for both behavior verification and performance measurement.
 *
 * The instrumentation mirrors Python's codeflash implementation:
 * - Static identifiers (testModule, testFunction, lineId) are passed at instrumentation time
 * - Dynamic invocation counter increments only when same call site is seen again (e.g., in loops)
 * - Uses hrtime for nanosecond precision timing
 * - SQLite for consistent data format with Python implementation
 *
 * Usage:
 *   const codeflash = require('./codeflash-jest-helper');
 *
 *   // For behavior verification (writes to SQLite):
 *   const result = codeflash.capture('functionName', lineId, targetFunction, arg1, arg2);
 *
 *   // For performance benchmarking (stdout only):
 *   const result = codeflash.capturePerf('functionName', lineId, targetFunction, arg1, arg2);
 *
 * Environment Variables:
 *   CODEFLASH_OUTPUT_FILE - Path to write results SQLite file
 *   CODEFLASH_LOOP_INDEX - Current benchmark loop iteration (default: 1)
 *   CODEFLASH_TEST_ITERATION - Test iteration number (default: 0)
 *   CODEFLASH_TEST_MODULE - Test module path
 */

const fs = require('fs');
const path = require('path');

// Load the codeflash serializer for robust value serialization
const serializer = require('./codeflash-serializer');

// Try to load better-sqlite3, fall back to JSON if not available
let Database;
let useSqlite = false;
try {
    Database = require('better-sqlite3');
    useSqlite = true;
} catch (e) {
    // better-sqlite3 not available, will use JSON fallback
    console.warn('[codeflash] better-sqlite3 not found, using JSON fallback');
}

// Configuration from environment
const OUTPUT_FILE = process.env.CODEFLASH_OUTPUT_FILE || '/tmp/codeflash_results.sqlite';
const LOOP_INDEX = parseInt(process.env.CODEFLASH_LOOP_INDEX || '1', 10);
const TEST_ITERATION = process.env.CODEFLASH_TEST_ITERATION || '0';
const TEST_MODULE = process.env.CODEFLASH_TEST_MODULE || '';

// Looping configuration for performance benchmarking
const MIN_LOOPS = parseInt(process.env.CODEFLASH_MIN_LOOPS || '5', 10);
const MAX_LOOPS = parseInt(process.env.CODEFLASH_MAX_LOOPS || '100000', 10);
const TARGET_DURATION_MS = parseInt(process.env.CODEFLASH_TARGET_DURATION_MS || '10000', 10);
const STABILITY_CHECK = process.env.CODEFLASH_STABILITY_CHECK !== 'false';

// Stability checking constants (matching Python's pytest_plugin.py)
const STABILITY_WINDOW_SIZE = 0.35;        // 35% of estimated total loops
const STABILITY_CENTER_TOLERANCE = 0.0025;  // ±0.25% around median
const STABILITY_SPREAD_TOLERANCE = 0.0025;  // 0.25% window spread

// Current test context (set by Jest hooks)
let currentTestName = null;
let currentTestPath = null;  // Test file path from Jest

// Invocation counter map: tracks how many times each testId has been seen
// Key: testId (testModule:testClass:testFunction:lineId:loopIndex)
// Value: count (starts at 0, increments each time same key is seen)
const invocationCounterMap = new Map();

// Results buffer (for JSON fallback)
const results = [];

// SQLite database (lazy initialized)
let db = null;

/**
 * Get high-resolution time in nanoseconds.
 * Prefers process.hrtime.bigint() for nanosecond precision,
 * falls back to performance.now() * 1e6 for non-Node environments.
 *
 * @returns {bigint|number} - Time in nanoseconds
 */
function getTimeNs() {
    if (typeof process !== 'undefined' && process.hrtime && process.hrtime.bigint) {
        return process.hrtime.bigint();
    }
    // Fallback to performance.now() in milliseconds, converted to nanoseconds
    const { performance } = require('perf_hooks');
    return BigInt(Math.floor(performance.now() * 1_000_000));
}

/**
 * Calculate duration in nanoseconds.
 *
 * @param {bigint} start - Start time in nanoseconds
 * @param {bigint} end - End time in nanoseconds
 * @returns {number} - Duration in nanoseconds (as Number for SQLite compatibility)
 */
function getDurationNs(start, end) {
    const duration = end - start;
    // Convert to Number for SQLite storage (SQLite INTEGER is 64-bit)
    return Number(duration);
}

/**
 * Sanitize a string for use in test IDs.
 * Replaces special characters that could conflict with regex extraction
 * during stdout parsing.
 *
 * Characters replaced with '_': ! # : (space) ( ) [ ] { } | \ / * ? ^ $ . + -
 *
 * @param {string} str - String to sanitize
 * @returns {string} - Sanitized string safe for test IDs
 */
function sanitizeTestId(str) {
    if (!str) return str;
    // Replace characters that could conflict with our delimiter pattern (######)
    // or the colon-separated format, or general regex metacharacters
    return str.replace(/[!#: ()\[\]{}|\\/*?^$.+\-]/g, '_');
}

/**
 * Get or create invocation index for a testId.
 * This mirrors Python's index tracking per wrapper function.
 *
 * @param {string} testId - Unique test identifier
 * @returns {number} - Current invocation index (0-based)
 */
function getInvocationIndex(testId) {
    const currentIndex = invocationCounterMap.get(testId);
    if (currentIndex === undefined) {
        invocationCounterMap.set(testId, 0);
        return 0;
    }
    invocationCounterMap.set(testId, currentIndex + 1);
    return currentIndex + 1;
}

/**
 * Reset invocation counter for a test.
 * Called at the start of each test to ensure consistent indexing.
 */
function resetInvocationCounters() {
    invocationCounterMap.clear();
}

/**
 * Initialize the SQLite database.
 */
function initDatabase() {
    if (!useSqlite || db) return;

    try {
        db = new Database(OUTPUT_FILE);
        db.exec(`
            CREATE TABLE IF NOT EXISTS test_results (
                test_module_path TEXT,
                test_class_name TEXT,
                test_function_name TEXT,
                function_getting_tested TEXT,
                loop_index INTEGER,
                iteration_id TEXT,
                runtime INTEGER,
                return_value BLOB,
                verification_type TEXT
            )
        `);
    } catch (e) {
        console.error('[codeflash] Failed to initialize SQLite:', e.message);
        useSqlite = false;
    }
}

/**
 * Safely serialize a value for storage.
 *
 * @param {any} value - Value to serialize
 * @returns {Buffer} - Serialized value as Buffer
 */
function safeSerialize(value) {
    try {
        return serializer.serialize(value);
    } catch (e) {
        console.warn('[codeflash] Serialization failed:', e.message);
        return Buffer.from(JSON.stringify({ __type: 'SerializationError', error: e.message }));
    }
}

/**
 * Safely deserialize a buffer back to a value.
 *
 * @param {Buffer|Uint8Array} buffer - Serialized buffer
 * @returns {any} - Deserialized value
 */
function safeDeserialize(buffer) {
    try {
        return serializer.deserialize(buffer);
    } catch (e) {
        console.warn('[codeflash] Deserialization failed:', e.message);
        return { __type: 'DeserializationError', error: e.message };
    }
}

/**
 * Record a test result to SQLite or JSON buffer.
 *
 * @param {string} testModulePath - Test module path
 * @param {string|null} testClassName - Test class name (null for Jest)
 * @param {string} testFunctionName - Test function name
 * @param {string} funcName - Name of the function being tested
 * @param {string} invocationId - Unique invocation identifier (lineId_index)
 * @param {Array} args - Arguments passed to the function
 * @param {any} returnValue - Return value from the function
 * @param {Error|null} error - Error thrown by the function (if any)
 * @param {number} durationNs - Execution time in nanoseconds
 */
function recordResult(testModulePath, testClassName, testFunctionName, funcName, invocationId, args, returnValue, error, durationNs) {
    // Serialize the return value (args, kwargs (empty for JS), return_value) like Python does
    const serializedValue = error
        ? safeSerialize(error)
        : safeSerialize([args, {}, returnValue]);

    if (useSqlite && db) {
        try {
            const stmt = db.prepare(`
                INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            `);
            stmt.run(
                testModulePath,           // test_module_path
                testClassName,            // test_class_name
                testFunctionName,         // test_function_name
                funcName,                 // function_getting_tested
                LOOP_INDEX,               // loop_index
                invocationId,             // iteration_id
                durationNs,               // runtime (nanoseconds) - no rounding
                serializedValue,          // return_value (serialized)
                'function_call'           // verification_type
            );
        } catch (e) {
            console.error('[codeflash] Failed to write to SQLite:', e.message);
            // Fall back to JSON
            results.push({
                testModulePath,
                testClassName,
                testFunctionName,
                funcName,
                loopIndex: LOOP_INDEX,
                iterationId: invocationId,
                durationNs,
                returnValue: error ? null : returnValue,
                error: error ? { name: error.name, message: error.message } : null,
                verificationType: 'function_call'
            });
        }
    } else {
        // JSON fallback
        results.push({
            testModulePath,
            testClassName,
            testFunctionName,
            funcName,
            loopIndex: LOOP_INDEX,
            iterationId: invocationId,
            durationNs,
            returnValue: error ? null : returnValue,
            error: error ? { name: error.name, message: error.message } : null,
            verificationType: 'function_call'
        });
    }
}

/**
 * Capture a function call with full behavior tracking.
 *
 * This is the main API for instrumenting function calls for BEHAVIOR verification.
 * It captures inputs, outputs, errors, and timing.
 * Results are written to SQLite for comparison between original and optimized code.
 *
 * Static parameters (funcName, lineId) are determined at instrumentation time.
 * The lineId enables tracking when the same call site is invoked multiple times (e.g., in loops).
 *
 * @param {string} funcName - Name of the function being tested (static)
 * @param {string} lineId - Line number identifier in test file (static)
 * @param {Function} fn - The function to call
 * @param {...any} args - Arguments to pass to the function
 * @returns {any} - The function's return value
 * @throws {Error} - Re-throws any error from the function
 */
function capture(funcName, lineId, fn, ...args) {
    // Initialize database on first capture
    initDatabase();

    // Get test context (raw values for SQLite storage)
    // Use TEST_MODULE env var if set, otherwise derive from test file path
    let testModulePath;
    if (TEST_MODULE) {
        testModulePath = TEST_MODULE;
    } else if (currentTestPath) {
        // Get relative path from cwd and convert to module-style path
        const path = require('path');
        const relativePath = path.relative(process.cwd(), currentTestPath);
        // Convert to Python module-style path (e.g., "tests/test_foo.test.js" -> "tests.test_foo.test")
        // This matches what Jest's junit XML produces
        testModulePath = relativePath
            .replace(/\\/g, '/')       // Handle Windows paths
            .replace(/\.js$/, '')       // Remove .js extension
            .replace(/\.test$/, '.test') // Keep .test suffix
            .replace(/\//g, '.');       // Convert path separators to dots
    } else {
        testModulePath = currentTestName || 'unknown';
    }
    const testClassName = null;  // Jest doesn't use classes like Python
    const testFunctionName = currentTestName || 'unknown';

    // Sanitized versions for stdout tags (avoid regex conflicts)
    const safeModulePath = sanitizeTestId(testModulePath);
    const safeTestFunctionName = sanitizeTestId(testFunctionName);

    // Create testId for invocation tracking (matches Python format)
    const testId = `${safeModulePath}:${testClassName}:${safeTestFunctionName}:${lineId}:${LOOP_INDEX}`;

    // Get invocation index (increments if same testId seen again)
    const invocationIndex = getInvocationIndex(testId);
    const invocationId = `${lineId}_${invocationIndex}`;

    // Format stdout tag (matches Python format, uses sanitized names)
    const testStdoutTag = `${safeModulePath}:${testClassName ? testClassName + '.' : ''}${safeTestFunctionName}:${funcName}:${LOOP_INDEX}:${invocationId}`;

    // Print start tag
    console.log(`!$######${testStdoutTag}######$!`);

    // Timing with nanosecond precision
    const startTime = getTimeNs();
    let returnValue;
    let error = null;

    try {
        returnValue = fn(...args);

        // Handle promises (async functions)
        if (returnValue instanceof Promise) {
            return returnValue.then(
                (resolved) => {
                    const endTime = getTimeNs();
                    const durationNs = getDurationNs(startTime, endTime);
                    recordResult(testModulePath, testClassName, testFunctionName, funcName, invocationId, args, resolved, null, durationNs);
                    // Print end tag (no duration for behavior mode)
                    console.log(`!######${testStdoutTag}######!`);
                    return resolved;
                },
                (err) => {
                    const endTime = getTimeNs();
                    const durationNs = getDurationNs(startTime, endTime);
                    recordResult(testModulePath, testClassName, testFunctionName, funcName, invocationId, args, null, err, durationNs);
                    console.log(`!######${testStdoutTag}######!`);
                    throw err;
                }
            );
        }
    } catch (e) {
        error = e;
    }

    const endTime = getTimeNs();
    const durationNs = getDurationNs(startTime, endTime);
    recordResult(testModulePath, testClassName, testFunctionName, funcName, invocationId, args, returnValue, error, durationNs);

    // Print end tag (no duration for behavior mode, matching Python)
    console.log(`!######${testStdoutTag}######!`);

    if (error) throw error;
    return returnValue;
}

/**
 * Capture a function call for PERFORMANCE benchmarking only.
 *
 * This is a lightweight instrumentation that only measures timing.
 * It prints start/end tags to stdout (no SQLite writes, no serialization overhead).
 * Used when we've already verified behavior and just need accurate timing.
 *
 * The timing measurement is done exactly around the function call for accuracy.
 *
 * Output format matches Python's codeflash_performance wrapper:
 * Start: !$######test_module:test_class.test_name:func_name:loop_index:invocation_id######$!
 * End:   !######test_module:test_class.test_name:func_name:loop_index:invocation_id:duration_ns######!
 *
 * @param {string} funcName - Name of the function being tested (static)
 * @param {string} lineId - Line number identifier in test file (static)
 * @param {Function} fn - The function to call
 * @param {...any} args - Arguments to pass to the function
 * @returns {any} - The function's return value
 * @throws {Error} - Re-throws any error from the function
 */
function capturePerf(funcName, lineId, fn, ...args) {
    // Get test context
    // Use TEST_MODULE env var if set, otherwise derive from test file path
    let testModulePath;
    if (TEST_MODULE) {
        testModulePath = TEST_MODULE;
    } else if (currentTestPath) {
        // Get relative path from cwd and convert to module-style path
        const path = require('path');
        const relativePath = path.relative(process.cwd(), currentTestPath);
        // Convert to Python module-style path (e.g., "tests/test_foo.test.js" -> "tests.test_foo.test")
        testModulePath = relativePath
            .replace(/\\/g, '/')
            .replace(/\.js$/, '')
            .replace(/\.test$/, '.test')
            .replace(/\//g, '.');
    } else {
        testModulePath = currentTestName || 'unknown';
    }
    const testClassName = null;  // Jest doesn't use classes like Python
    const testFunctionName = currentTestName || 'unknown';

    // Sanitized versions for stdout tags (avoid regex conflicts)
    const safeModulePath = sanitizeTestId(testModulePath);
    const safeTestFunctionName = sanitizeTestId(testFunctionName);

    // Create testId for invocation tracking (matches Python format)
    const testId = `${safeModulePath}:${testClassName}:${safeTestFunctionName}:${lineId}:${LOOP_INDEX}`;

    // Get invocation index (increments if same testId seen again)
    const invocationIndex = getInvocationIndex(testId);
    const invocationId = `${lineId}_${invocationIndex}`;

    // Format stdout tag (matches Python format, uses sanitized names)
    const testStdoutTag = `${safeModulePath}:${testClassName ? testClassName + '.' : ''}${safeTestFunctionName}:${funcName}:${LOOP_INDEX}:${invocationId}`;

    // Print start tag
    console.log(`!$######${testStdoutTag}######$!`);

    // Timing with nanosecond precision - exactly around the function call
    let returnValue;
    let error = null;
    let durationNs;

    try {
        const startTime = getTimeNs();
        returnValue = fn(...args);
        const endTime = getTimeNs();
        durationNs = getDurationNs(startTime, endTime);

        // Handle promises (async functions)
        if (returnValue instanceof Promise) {
            return returnValue.then(
                (resolved) => {
                    // For async, we measure until resolution
                    const asyncEndTime = getTimeNs();
                    const asyncDurationNs = getDurationNs(startTime, asyncEndTime);
                    // Print end tag with timing
                    console.log(`!######${testStdoutTag}:${asyncDurationNs}######!`);
                    return resolved;
                },
                (err) => {
                    const asyncEndTime = getTimeNs();
                    const asyncDurationNs = getDurationNs(startTime, asyncEndTime);
                    // Print end tag with timing even on error
                    console.log(`!######${testStdoutTag}:${asyncDurationNs}######!`);
                    throw err;
                }
            );
        }
    } catch (e) {
        const endTime = getTimeNs();
        // For sync errors, we still need to calculate duration
        // Use a fallback if we didn't capture startTime yet
        durationNs = 0;
        error = e;
    }

    // Print end tag with timing (no rounding)
    console.log(`!######${testStdoutTag}:${durationNs}######!`);

    if (error) throw error;
    return returnValue;
}

/**
 * Check if performance measurements have stabilized.
 * Implements the same stability criteria as Python's pytest_plugin.py.
 *
 * @param {number[]} runtimes - Array of runtime measurements
 * @param {number} windowSize - Size of the window to check
 * @returns {boolean} - True if performance has stabilized
 */
function checkStability(runtimes, windowSize) {
    if (runtimes.length < windowSize || windowSize < 3) {
        return false;
    }

    // Get recent window
    const window = runtimes.slice(-windowSize);

    // Check center tolerance (all values within ±0.25% of median)
    const sorted = [...window].sort((a, b) => a - b);
    const medianIndex = Math.floor(sorted.length / 2);
    const median = sorted[medianIndex];
    const centerTolerance = median * STABILITY_CENTER_TOLERANCE;

    const withinCenter = window.every(v => Math.abs(v - median) <= centerTolerance);
    if (!withinCenter) return false;

    // Check spread tolerance (max-min ≤ 0.25% of min)
    const minVal = Math.min(...window);
    const maxVal = Math.max(...window);
    const spreadTolerance = minVal * STABILITY_SPREAD_TOLERANCE;

    return (maxVal - minVal) <= spreadTolerance;
}

/**
 * Capture a function call with internal looping for stable performance measurement.
 *
 * This function runs the target function multiple times within a single test execution,
 * similar to Python's pytest_plugin behavior. It provides stable timing by:
 * - Running multiple iterations to warm up JIT
 * - Continuing until timing stabilizes or time limit is reached
 * - Outputting timing data for each iteration
 *
 * Environment Variables:
 *   CODEFLASH_MIN_LOOPS - Minimum number of loops (default: 5)
 *   CODEFLASH_MAX_LOOPS - Maximum number of loops (default: 100000)
 *   CODEFLASH_TARGET_DURATION_MS - Target duration in ms (default: 10000)
 *   CODEFLASH_STABILITY_CHECK - Enable stability checking (default: true)
 *
 * @param {string} funcName - Name of the function being tested (static)
 * @param {string} lineId - Line number identifier in test file (static)
 * @param {Function} fn - The function to call
 * @param {...any} args - Arguments to pass to the function
 * @returns {any} - The function's return value from the last iteration
 * @throws {Error} - Re-throws any error from the function
 */
function capturePerfLooped(funcName, lineId, fn, ...args) {
    // Get test context
    // Use TEST_MODULE env var if set, otherwise derive from test file path
    let testModulePath;
    if (TEST_MODULE) {
        testModulePath = TEST_MODULE;
    } else if (currentTestPath) {
        // Get relative path from cwd and convert to module-style path
        const path = require('path');
        const relativePath = path.relative(process.cwd(), currentTestPath);
        // Convert to Python module-style path (e.g., "tests/test_foo.test.js" -> "tests.test_foo.test")
        testModulePath = relativePath
            .replace(/\\/g, '/')
            .replace(/\.js$/, '')
            .replace(/\.test$/, '.test')
            .replace(/\//g, '.');
    } else {
        testModulePath = currentTestName || 'unknown';
    }
    const testClassName = null;  // Jest doesn't use classes like Python
    const testFunctionName = currentTestName || 'unknown';

    // Sanitized versions for stdout tags (avoid regex conflicts)
    const safeModulePath = sanitizeTestId(testModulePath);
    const safeTestFunctionName = sanitizeTestId(testFunctionName);

    // Create base testId for invocation tracking
    const baseTestId = `${safeModulePath}:${testClassName}:${safeTestFunctionName}:${lineId}`;

    // Get invocation index (same call site in loops within test)
    const invocationIndex = getInvocationIndex(baseTestId + ':base');
    const invocationId = `${lineId}_${invocationIndex}`;

    // Track runtimes for stability checking
    const runtimes = [];
    let returnValue;
    let error = null;

    const loopStartTime = Date.now();
    let loopCount = 0;

    while (true) {
        loopCount++;

        // Create per-loop stdout tag (uses sanitized names)
        const testStdoutTag = `${safeModulePath}:${testClassName ? testClassName + '.' : ''}${safeTestFunctionName}:${funcName}:${loopCount}:${invocationId}`;

        // Print start tag
        console.log(`!$######${testStdoutTag}######$!`);

        // Timing with nanosecond precision
        let durationNs;
        try {
            const startTime = getTimeNs();
            returnValue = fn(...args);
            const endTime = getTimeNs();
            durationNs = getDurationNs(startTime, endTime);

            // Handle promises - for async, we can't easily loop internally
            // Fall back to single execution for async functions
            if (returnValue instanceof Promise) {
                return returnValue.then(
                    (resolved) => {
                        const asyncEndTime = getTimeNs();
                        const asyncDurationNs = getDurationNs(startTime, asyncEndTime);
                        console.log(`!######${testStdoutTag}:${asyncDurationNs}######!`);
                        return resolved;
                    },
                    (err) => {
                        const asyncEndTime = getTimeNs();
                        const asyncDurationNs = getDurationNs(startTime, asyncEndTime);
                        console.log(`!######${testStdoutTag}:${asyncDurationNs}######!`);
                        throw err;
                    }
                );
            }
        } catch (e) {
            durationNs = 0;
            error = e;
            // Print end tag even on error
            console.log(`!######${testStdoutTag}:${durationNs}######!`);
            throw error;
        }

        // Print end tag with timing
        console.log(`!######${testStdoutTag}:${durationNs}######!`);

        // Track runtime for stability
        runtimes.push(durationNs);

        // Check stopping conditions
        const elapsedMs = Date.now() - loopStartTime;

        // Stop if we've reached max loops
        if (loopCount >= MAX_LOOPS) {
            break;
        }

        // EARLY EXIT: Stop if we've exceeded time limit (even before MIN_LOOPS)
        // This handles slow functions that would otherwise timeout
        if (elapsedMs >= TARGET_DURATION_MS) {
            break;
        }

        // Stop if we've reached min loops AND exceeded a reasonable portion of time
        if (loopCount >= MIN_LOOPS && elapsedMs >= TARGET_DURATION_MS * 0.8) {
            break;
        }

        // Stability check
        if (STABILITY_CHECK && loopCount >= MIN_LOOPS) {
            // Estimate total loops based on current rate
            const rate = loopCount / elapsedMs;
            const estimatedTotalLoops = Math.floor(rate * TARGET_DURATION_MS);
            const windowSize = Math.max(3, Math.floor(STABILITY_WINDOW_SIZE * estimatedTotalLoops));

            if (checkStability(runtimes, windowSize)) {
                // Performance has stabilized
                break;
            }
        }
    }

    return returnValue;
}

/**
 * Capture multiple invocations for benchmarking.
 *
 * @param {string} funcName - Name of the function being tested
 * @param {string} lineId - Line number identifier
 * @param {Function} fn - The function to call
 * @param {Array<Array>} argsList - List of argument arrays to test
 * @returns {Array} - Array of return values
 */
function captureMultiple(funcName, lineId, fn, argsList) {
    return argsList.map(args => capture(funcName, lineId, fn, ...args));
}

/**
 * Write remaining JSON results to file (fallback mode).
 * Called automatically via Jest afterAll hook.
 */
function writeResults() {
    // Close SQLite connection if open
    if (db) {
        try {
            db.close();
        } catch (e) {
            // Ignore close errors
        }
        db = null;
        return;
    }

    // Write JSON fallback if SQLite wasn't used
    if (results.length === 0) return;

    try {
        // Write as JSON for fallback parsing
        const jsonPath = OUTPUT_FILE.replace('.sqlite', '.json');
        const output = {
            version: '1.0.0',
            loopIndex: LOOP_INDEX,
            timestamp: Date.now(),
            results
        };
        fs.writeFileSync(jsonPath, JSON.stringify(output, null, 2));
    } catch (e) {
        console.error('[codeflash] Error writing JSON results:', e.message);
    }
}

/**
 * Clear all recorded results.
 * Useful for resetting between test files.
 */
function clearResults() {
    results.length = 0;
    resetInvocationCounters();
}

/**
 * Get the current results buffer.
 * Useful for debugging or custom result handling.
 *
 * @returns {Array} - Current results buffer
 */
function getResults() {
    return results;
}

/**
 * Set the current test name.
 * Called automatically via Jest beforeEach hook.
 *
 * @param {string} name - Test name
 */
function setTestName(name) {
    currentTestName = name;
    resetInvocationCounters();
}

// Jest lifecycle hooks - these run automatically when this module is imported
if (typeof beforeEach !== 'undefined') {
    beforeEach(() => {
        // Get current test name and path from Jest's expect state
        try {
            const state = expect.getState();
            currentTestName = state.currentTestName || 'unknown';
            // testPath is the absolute path to the test file
            currentTestPath = state.testPath || null;
        } catch (e) {
            currentTestName = 'unknown';
            currentTestPath = null;
        }
        // Reset invocation counters for each test
        resetInvocationCounters();
    });
}

if (typeof afterAll !== 'undefined') {
    afterAll(() => {
        writeResults();
    });
}

// Export public API
module.exports = {
    capture,           // Behavior verification (writes to SQLite)
    capturePerf,       // Performance benchmarking (prints to stdout only, single run)
    capturePerfLooped, // Performance benchmarking with internal looping
    captureMultiple,
    writeResults,
    clearResults,
    getResults,
    setTestName,
    safeSerialize,
    safeDeserialize,
    initDatabase,
    resetInvocationCounters,
    getInvocationIndex,
    checkStability,
    sanitizeTestId,    // Sanitize test names for stdout tags
    // Serializer info
    getSerializerType: serializer.getSerializerType,
    // Constants
    LOOP_INDEX,
    OUTPUT_FILE,
    TEST_ITERATION,
    MIN_LOOPS,
    MAX_LOOPS,
    TARGET_DURATION_MS,
    STABILITY_CHECK
};
