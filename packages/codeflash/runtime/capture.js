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
 *   const { capture } = require('@codeflash/jest-runtime');
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
const Database = require('better-sqlite3');

// Load the codeflash serializer for robust value serialization
const serializer = require('./serializer');

// Try to load better-sqlite3, fall back to JSON if not available
let useSqlite = false;

// Configuration from environment
const OUTPUT_FILE = process.env.CODEFLASH_OUTPUT_FILE;
const LOOP_INDEX = parseInt(process.env.CODEFLASH_LOOP_INDEX || '1', 10);
const TEST_ITERATION = process.env.CODEFLASH_TEST_ITERATION;
const TEST_MODULE = process.env.CODEFLASH_TEST_MODULE;

// Performance loop configuration - controls batched looping in capturePerf
// Batched looping ensures fair distribution across all test invocations:
//   Batch 1: Test1(5 loops) → Test2(5 loops) → Test3(5 loops)
//   Batch 2: Test1(5 loops) → Test2(5 loops) → Test3(5 loops)
//   ...until time budget exhausted
const PERF_LOOP_COUNT = parseInt(process.env.CODEFLASH_PERF_LOOP_COUNT || '1', 10);
const PERF_MIN_LOOPS = parseInt(process.env.CODEFLASH_PERF_MIN_LOOPS || '5', 10);
const PERF_TARGET_DURATION_MS = parseInt(process.env.CODEFLASH_PERF_TARGET_DURATION_MS || '10000', 10);
const PERF_BATCH_SIZE = parseInt(process.env.CODEFLASH_PERF_BATCH_SIZE || '10', 10);
const PERF_STABILITY_CHECK = (process.env.CODEFLASH_PERF_STABILITY_CHECK || 'false').toLowerCase() === 'true';
// Current batch number - set by loop-runner before each batch
// This allows continuous loop indices even when Jest resets module state
const PERF_CURRENT_BATCH = parseInt(process.env.CODEFLASH_PERF_CURRENT_BATCH || '0', 10);

// Stability constants (matching Python's config_consts.py)
const STABILITY_WINDOW_SIZE = 0.35;
const STABILITY_CENTER_TOLERANCE = 0.0025;
const STABILITY_SPREAD_TOLERANCE = 0.0025;

// Shared state for coordinating batched looping across all capturePerf calls
// Uses process object to persist across Jest's module reloads per test file
const PERF_STATE_KEY = '__codeflash_perf_state__';
if (!process[PERF_STATE_KEY]) {
    process[PERF_STATE_KEY] = {
        startTime: null,           // When benchmarking started
        totalLoopsCompleted: 0,    // Total loops across all invocations
        shouldStop: false,         // Flag to stop all further looping
        currentBatch: 0,           // Current batch number (incremented by runner)
        invocationLoopCounts: {},  // Track loops per invocation: {invocationKey: loopCount}
    };
}
const sharedPerfState = process[PERF_STATE_KEY];

/**
 * Check if the shared time budget has been exceeded.
 * @returns {boolean} True if we should stop looping
 */
function checkSharedTimeLimit() {
    if (sharedPerfState.shouldStop) return true;
    if (sharedPerfState.startTime === null) {
        sharedPerfState.startTime = Date.now();
        return false;
    }
    const elapsed = Date.now() - sharedPerfState.startTime;
    if (elapsed >= PERF_TARGET_DURATION_MS && sharedPerfState.totalLoopsCompleted >= PERF_MIN_LOOPS) {
        sharedPerfState.shouldStop = true;
        return true;
    }
    return false;
}

/**
 * Get the current loop index for a specific invocation.
 * Each invocation tracks its own loop count independently within a batch.
 * The actual loop index is computed as: (batch - 1) * BATCH_SIZE + localIndex
 * This ensures continuous loop indices even when Jest resets module state.
 * @param {string} invocationKey - Unique key for this test invocation
 * @returns {number} The next global loop index for this invocation
 */
function getInvocationLoopIndex(invocationKey) {
    // Track local loop count within this batch (starts at 0)
    if (!sharedPerfState.invocationLoopCounts[invocationKey]) {
        sharedPerfState.invocationLoopCounts[invocationKey] = 0;
    }
    const localIndex = ++sharedPerfState.invocationLoopCounts[invocationKey];

    // Calculate global loop index using batch number from environment
    // PERF_CURRENT_BATCH is 1-based (set by loop-runner before each batch)
    const currentBatch = parseInt(process.env.CODEFLASH_PERF_CURRENT_BATCH || '1', 10);
    const globalIndex = (currentBatch - 1) * PERF_BATCH_SIZE + localIndex;

    return globalIndex;
}

/**
 * Increment the batch counter and reset per-invocation loop counts.
 * Called by loop-runner between test file runs.
 *
 * IMPORTANT: We must reset invocationLoopCounts here because the globalIndex
 * formula (currentBatch - 1) * PERF_BATCH_SIZE + localIndex assumes localIndex
 * starts at 1 for each batch. Without this reset, localIndex would continue
 * incrementing across batches, causing globalIndex to grow too fast and
 * triggering early loop termination.
 */
function incrementBatch() {
    sharedPerfState.currentBatch++;
    // Reset per-invocation loop counts for the new batch
    // This ensures localIndex starts at 1 for each batch, making the
    // globalIndex calculation correct: (batch-1)*batchSize + localIndex
    sharedPerfState.invocationLoopCounts = {};
}

/**
 * Get current batch number.
 */
function getCurrentBatch() {
    return sharedPerfState.currentBatch;
}

// Random seed for reproducible test runs
// Both original and optimized runs use the same seed to get identical "random" values
const RANDOM_SEED = parseInt(process.env.CODEFLASH_RANDOM_SEED, 10);

/**
 * Seeded random number generator using mulberry32 algorithm.
 * This provides reproducible "random" numbers given a fixed seed.
 */
function createSeededRandom(seed) {
    let state = seed;
    return function() {
        state |= 0;
        state = state + 0x6D2B79F5 | 0;
        let t = Math.imul(state ^ state >>> 15, 1 | state);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

// Override non-deterministic APIs with seeded versions if seed is provided
// NOTE: We do NOT seed performance.now() or process.hrtime() as those are used
// internally by this script for timing measurements.
if (RANDOM_SEED !== 0) {
    // Seed Math.random
    const seededRandom = createSeededRandom(RANDOM_SEED);
    Math.random = seededRandom;

    // Seed Date.now() and new Date() - use fixed base timestamp that increments
    const SEEDED_BASE_TIME = 1700000000000; // Nov 14, 2023 - fixed reference point
    let dateOffset = 0;
    const OriginalDate = Date;
    const originalDateNow = Date.now;

    Date.now = function() {
        return SEEDED_BASE_TIME + (dateOffset++);
    };

    // Override Date constructor to use seeded time when called without arguments
    function SeededDate(...args) {
        if (args.length === 0) {
            // No arguments: use seeded current time
            return new OriginalDate(SEEDED_BASE_TIME + (dateOffset++));
        }
        // With arguments: use original behavior
        return new OriginalDate(...args);
    }
    SeededDate.prototype = OriginalDate.prototype;
    SeededDate.now = Date.now;
    SeededDate.parse = OriginalDate.parse;
    SeededDate.UTC = OriginalDate.UTC;
    global.Date = SeededDate;

    // Seed crypto.randomUUID() and crypto.getRandomValues()
    try {
        const crypto = require('crypto');
        const randomForCrypto = createSeededRandom(RANDOM_SEED + 1000); // Different seed to avoid correlation

        // Seed crypto.randomUUID()
        if (crypto.randomUUID) {
            const originalRandomUUID = crypto.randomUUID.bind(crypto);
            crypto.randomUUID = function() {
                // Generate a deterministic UUID v4 format
                const hex = () => Math.floor(randomForCrypto() * 16).toString(16);
                const bytes = Array.from({ length: 32 }, hex).join('');
                return `${bytes.slice(0, 8)}-${bytes.slice(8, 12)}-4${bytes.slice(13, 16)}-${(8 + Math.floor(randomForCrypto() * 4)).toString(16)}${bytes.slice(17, 20)}-${bytes.slice(20, 32)}`;
            };
        }

        // Seed crypto.getRandomValues() - used by uuid libraries
        const seededGetRandomValues = function(array) {
            for (let i = 0; i < array.length; i++) {
                if (array instanceof Uint8Array) {
                    array[i] = Math.floor(randomForCrypto() * 256);
                } else if (array instanceof Uint16Array) {
                    array[i] = Math.floor(randomForCrypto() * 65536);
                } else if (array instanceof Uint32Array) {
                    array[i] = Math.floor(randomForCrypto() * 4294967296);
                } else {
                    array[i] = Math.floor(randomForCrypto() * 256);
                }
            }
            return array;
        };

        if (crypto.getRandomValues) {
            crypto.getRandomValues = seededGetRandomValues;
        }

        // Also seed webcrypto if available (Node 18+)
        // Use the same seeded function to avoid circular references
        if (crypto.webcrypto) {
            if (crypto.webcrypto.getRandomValues) {
                crypto.webcrypto.getRandomValues = seededGetRandomValues;
            }
            if (crypto.webcrypto.randomUUID) {
                crypto.webcrypto.randomUUID = crypto.randomUUID;
            }
        }
    } catch (e) {
        // crypto module not available, skip seeding
    }
}

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
 * Check if performance has stabilized (for internal looping).
 * Matches Python's pytest_plugin.should_stop() logic.
 */
function shouldStopStability(runtimes, window, minWindowSize) {
    if (runtimes.length < window || runtimes.length < minWindowSize) {
        return false;
    }
    const recent = runtimes.slice(-window);
    const recentSorted = [...recent].sort((a, b) => a - b);
    const mid = Math.floor(window / 2);
    const median = window % 2 ? recentSorted[mid] : (recentSorted[mid - 1] + recentSorted[mid]) / 2;

    for (const r of recent) {
        if (Math.abs(r - median) / median > STABILITY_CENTER_TOLERANCE) {
            return false;
        }
    }
    const rMin = recentSorted[0];
    const rMax = recentSorted[recentSorted.length - 1];
    if (rMin === 0) return false;
    return (rMax - rMin) / rMin <= STABILITY_SPREAD_TOLERANCE;
}

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
    // Validate that fn is actually a function
    if (typeof fn !== 'function') {
        const fnType = fn === null ? 'null' : (fn === undefined ? 'undefined' : typeof fn);
        throw new TypeError(
            `codeflash.capture: Expected function '${funcName}' but got ${fnType}. ` +
            `This usually means the function was not imported correctly. ` +
            `Check that the import statement matches how the module exports the function ` +
            `(e.g., default export vs named export, CommonJS vs ES modules).`
        );
    }

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
 * When CODEFLASH_PERF_LOOP_COUNT > 1, this function loops internally to avoid
 * Jest environment overhead per iteration. This dramatically improves utilization
 * (time spent in actual function execution vs overhead).
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
    // Check if we should skip looping entirely (shared time budget exceeded)
    const shouldLoop = PERF_LOOP_COUNT > 1 && !checkSharedTimeLimit();

    // Get test context (computed once, reused across batch)
    let testModulePath;
    if (TEST_MODULE) {
        testModulePath = TEST_MODULE;
    } else if (currentTestPath) {
        const path = require('path');
        const relativePath = path.relative(process.cwd(), currentTestPath);
        testModulePath = relativePath
            .replace(/\\/g, '/')
            .replace(/\.js$/, '')
            .replace(/\.test$/, '.test')
            .replace(/\//g, '.');
    } else {
        testModulePath = currentTestName || 'unknown';
    }
    const testClassName = null;
    const testFunctionName = currentTestName || 'unknown';

    const safeModulePath = sanitizeTestId(testModulePath);
    const safeTestFunctionName = sanitizeTestId(testFunctionName);

    // Create unique key for this invocation (identifies this specific capturePerf call site)
    const invocationKey = `${safeModulePath}:${testClassName}:${safeTestFunctionName}:${funcName}:${lineId}`;

    // Check if we've already completed all loops for this invocation
    // If so, just execute the function once without timing (for test assertions)
    const peekLoopIndex = (sharedPerfState.invocationLoopCounts[invocationKey] || 0);
    const currentBatch = parseInt(process.env.CODEFLASH_PERF_CURRENT_BATCH || '1', 10);
    const nextGlobalIndex = (currentBatch - 1) * PERF_BATCH_SIZE + peekLoopIndex + 1;

    if (shouldLoop && nextGlobalIndex > PERF_LOOP_COUNT) {
        // All loops completed, just execute once for test assertion
        return fn(...args);
    }

    let lastReturnValue;
    let lastError = null;

    // Batched looping: run BATCH_SIZE loops per capturePerf call
    // This ensures fair distribution across all test invocations
    const batchSize = shouldLoop ? PERF_BATCH_SIZE : 1;

    for (let batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        // Check shared time limit BEFORE each iteration
        if (shouldLoop && checkSharedTimeLimit()) {
            break;
        }

        // Get the global loop index for this invocation (increments across batches)
        const loopIndex = getInvocationLoopIndex(invocationKey);

        // Check if we've exceeded max loops for this invocation
        if (loopIndex > PERF_LOOP_COUNT) {
            break;
        }

        // Get invocation index for the timing marker
        const testId = `${safeModulePath}:${testClassName}:${safeTestFunctionName}:${lineId}:${loopIndex}`;
        const invocationIndex = getInvocationIndex(testId);
        const invocationId = `${lineId}_${invocationIndex}`;

        // Format stdout tag with current loop index
        const testStdoutTag = `${safeModulePath}:${testClassName ? testClassName + '.' : ''}${safeTestFunctionName}:${funcName}:${loopIndex}:${invocationId}`;

        // Timing with nanosecond precision
        let durationNs;
        try {
            const startTime = getTimeNs();
            lastReturnValue = fn(...args);
            const endTime = getTimeNs();
            durationNs = getDurationNs(startTime, endTime);

            // Handle promises - for async functions, run once and return
            if (lastReturnValue instanceof Promise) {
                return lastReturnValue.then(
                    (resolved) => {
                        const asyncEndTime = getTimeNs();
                        const asyncDurationNs = getDurationNs(startTime, asyncEndTime);
                        console.log(`!######${testStdoutTag}:${asyncDurationNs}######!`);
                        sharedPerfState.totalLoopsCompleted++;
                        return resolved;
                    },
                    (err) => {
                        const asyncEndTime = getTimeNs();
                        const asyncDurationNs = getDurationNs(startTime, asyncEndTime);
                        console.log(`!######${testStdoutTag}:${asyncDurationNs}######!`);
                        sharedPerfState.totalLoopsCompleted++;
                        throw err;
                    }
                );
            }

            lastError = null;
        } catch (e) {
            durationNs = 0;
            lastError = e;
        }

        // Print end tag with timing
        console.log(`!######${testStdoutTag}:${durationNs}######!`);

        // Update shared loop counter
        sharedPerfState.totalLoopsCompleted++;

        // If we had an error, stop looping
        if (lastError) {
            break;
        }
    }

    if (lastError) throw lastError;

    // If we never executed (e.g., hit loop limit on first iteration), run once for assertion
    if (lastReturnValue === undefined && !lastError) {
        return fn(...args);
    }

    return lastReturnValue;
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
 * Reset shared performance state.
 * Should be called at the start of each test file to reset timing.
 */
function resetPerfState() {
    sharedPerfState.startTime = null;
    sharedPerfState.totalLoopsCompleted = 0;
    sharedPerfState.shouldStop = false;
}

/**
 * Clear all recorded results.
 * Useful for resetting between test files.
 */
function clearResults() {
    results.length = 0;
    resetInvocationCounters();
    resetPerfState();
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
    capturePerf,       // Performance benchmarking (prints to stdout only)
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
    sanitizeTestId,    // Sanitize test names for stdout tags
    // Batch looping control (used by loop-runner)
    incrementBatch,
    getCurrentBatch,
    checkSharedTimeLimit,
    // Serializer info
    getSerializerType: serializer.getSerializerType,
    // Constants
    LOOP_INDEX,
    OUTPUT_FILE,
    TEST_ITERATION,
    // Batch configuration
    PERF_BATCH_SIZE,
    PERF_LOOP_COUNT,
};
