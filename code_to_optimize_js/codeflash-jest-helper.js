/**
 * Codeflash Jest Helper - Unified Test Instrumentation
 *
 * This module provides a unified approach to instrumenting JavaScript tests
 * for both behavior verification and performance measurement.
 *
 * Unlike Python which has separate instrumentation methods for generated
 * vs existing tests, this helper works identically for ALL JavaScript tests.
 *
 * Uses SQLite for consistent data format with Python implementation.
 *
 * Usage:
 *   const codeflash = require('./codeflash-jest-helper');
 *
 *   // Wrap function calls to capture behavior
 *   const result = codeflash.capture('functionName', targetFunction, arg1, arg2);
 *
 * Environment Variables:
 *   CODEFLASH_OUTPUT_FILE - Path to write results SQLite file
 *   CODEFLASH_LOOP_INDEX - Current benchmark loop iteration (default: 1)
 *   CODEFLASH_TEST_ITERATION - Test iteration number (default: 0)
 *   CODEFLASH_TEST_MODULE - Test module path
 */

const fs = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');

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

// Current test context
let currentTestName = null;
let invocationCounter = 0;
let lineId = '0';

// Results buffer (for JSON fallback)
const results = [];

// SQLite database (lazy initialized)
let db = null;

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
 * Uses JSON serialization with special handling for complex types.
 *
 * @param {any} value - Value to serialize
 * @returns {Buffer} - Serialized value as Buffer
 */
function safeSerialize(value) {
    const seen = new WeakSet();

    function serialize(val) {
        // Handle primitives
        if (val === null || val === undefined) return val;
        if (typeof val === 'number') {
            if (Number.isNaN(val)) return { __type: 'NaN' };
            if (!Number.isFinite(val)) return { __type: val > 0 ? 'Infinity' : '-Infinity' };
            return val;
        }
        if (typeof val === 'string' || typeof val === 'boolean') return val;
        if (typeof val === 'bigint') return { __type: 'BigInt', value: val.toString() };
        if (typeof val === 'symbol') return { __type: 'Symbol', description: val.description };
        if (typeof val === 'function') return { __type: 'Function', name: val.name || 'anonymous' };

        // Handle special objects
        if (val instanceof Date) return { __type: 'Date', value: val.toISOString() };
        if (val instanceof RegExp) return { __type: 'RegExp', source: val.source, flags: val.flags };
        if (val instanceof Error) return { __type: 'Error', name: val.name, message: val.message };
        if (val instanceof Map) return { __type: 'Map', entries: Array.from(val.entries()).map(([k, v]) => [serialize(k), serialize(v)]) };
        if (val instanceof Set) return { __type: 'Set', values: Array.from(val).map(serialize) };
        if (ArrayBuffer.isView(val)) return { __type: val.constructor.name, data: Array.from(val) };
        if (val instanceof ArrayBuffer) return { __type: 'ArrayBuffer', byteLength: val.byteLength };
        if (val instanceof Promise) return { __type: 'Promise' };

        // Handle arrays
        if (Array.isArray(val)) {
            if (seen.has(val)) return { __type: 'CircularReference' };
            seen.add(val);
            return val.map(serialize);
        }

        // Handle objects
        if (typeof val === 'object') {
            if (seen.has(val)) return { __type: 'CircularReference' };
            seen.add(val);
            const result = {};
            for (const key of Object.keys(val)) {
                try {
                    result[key] = serialize(val[key]);
                } catch (e) {
                    result[key] = { __type: 'UnserializableProperty', error: e.message };
                }
            }
            return result;
        }

        return { __type: 'Unknown', typeof: typeof val };
    }

    try {
        const serialized = serialize(value);
        return Buffer.from(JSON.stringify(serialized));
    } catch (e) {
        return Buffer.from(JSON.stringify({ __type: 'SerializationError', error: e.message }));
    }
}

/**
 * Record a test result to SQLite or JSON buffer.
 *
 * @param {string} funcName - Name of the function being tested
 * @param {Array} args - Arguments passed to the function
 * @param {any} returnValue - Return value from the function
 * @param {Error|null} error - Error thrown by the function (if any)
 * @param {number} durationNs - Execution time in nanoseconds
 */
function recordResult(funcName, args, returnValue, error, durationNs) {
    const invocationId = `${lineId}_${invocationCounter}`;
    invocationCounter++;

    // Get test module path from file being tested or env
    const testModulePath = TEST_MODULE || currentTestName || 'unknown';

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
                null,                     // test_class_name (Jest doesn't use classes like Python)
                currentTestName,          // test_function_name
                funcName,                 // function_getting_tested
                LOOP_INDEX,               // loop_index
                invocationId,             // iteration_id
                Math.round(durationNs),   // runtime (nanoseconds)
                serializedValue,          // return_value (serialized)
                'function_call'           // verification_type
            );
        } catch (e) {
            console.error('[codeflash] Failed to write to SQLite:', e.message);
            // Fall back to JSON
            results.push({
                testModulePath,
                testClassName: null,
                testFunctionName: currentTestName,
                funcName,
                loopIndex: LOOP_INDEX,
                iterationId: invocationId,
                durationNs: Math.round(durationNs),
                returnValue: error ? null : returnValue,
                error: error ? { name: error.name, message: error.message } : null,
                verificationType: 'function_call'
            });
        }
    } else {
        // JSON fallback
        results.push({
            testModulePath,
            testClassName: null,
            testFunctionName: currentTestName,
            funcName,
            loopIndex: LOOP_INDEX,
            iterationId: invocationId,
            durationNs: Math.round(durationNs),
            returnValue: error ? null : returnValue,
            error: error ? { name: error.name, message: error.message } : null,
            verificationType: 'function_call'
        });
    }

    // Print stdout tag like Python does for test identification
    const testClassName = '';
    const testStdoutTag = `${testModulePath}:${testClassName}${currentTestName}:${funcName}:${LOOP_INDEX}:${invocationId}`;
    console.log(`!$######${testStdoutTag}######$!`);
}

/**
 * Capture a function call with full behavior tracking.
 *
 * This is the main API for instrumenting function calls.
 * It captures inputs, outputs, errors, and timing for every call.
 *
 * @param {string} funcName - Name of the function being tested
 * @param {Function} fn - The function to call
 * @param {...any} args - Arguments to pass to the function
 * @returns {any} - The function's return value
 * @throws {Error} - Re-throws any error from the function
 */
function capture(funcName, fn, ...args) {
    // Initialize database on first capture
    initDatabase();

    const startTime = performance.now();
    let returnValue;
    let error = null;

    try {
        returnValue = fn(...args);

        // Handle promises (async functions)
        if (returnValue instanceof Promise) {
            return returnValue.then(
                (resolved) => {
                    const endTime = performance.now();
                    const durationNs = (endTime - startTime) * 1_000_000;
                    recordResult(funcName, args, resolved, null, durationNs);
                    return resolved;
                },
                (err) => {
                    const endTime = performance.now();
                    const durationNs = (endTime - startTime) * 1_000_000;
                    recordResult(funcName, args, null, err, durationNs);
                    throw err;
                }
            );
        }
    } catch (e) {
        error = e;
    }

    const endTime = performance.now();
    const durationNs = (endTime - startTime) * 1_000_000;
    recordResult(funcName, args, returnValue, error, durationNs);

    if (error) throw error;
    return returnValue;
}

/**
 * Capture multiple invocations for benchmarking.
 *
 * @param {string} funcName - Name of the function being tested
 * @param {Function} fn - The function to call
 * @param {Array<Array>} argsList - List of argument arrays to test
 * @returns {Array} - Array of return values
 */
function captureMultiple(funcName, fn, argsList) {
    return argsList.map(args => capture(funcName, fn, ...args));
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
    invocationCounter = 0;
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
    invocationCounter = 0;
}

// Jest lifecycle hooks - these run automatically when this module is imported
if (typeof beforeEach !== 'undefined') {
    beforeEach(() => {
        // Get current test name from Jest's expect state
        try {
            currentTestName = expect.getState().currentTestName || 'unknown';
        } catch (e) {
            currentTestName = 'unknown';
        }
        invocationCounter = 0;
        lineId = String(Date.now() % 1000000); // Unique line ID per test
    });
}

if (typeof afterAll !== 'undefined') {
    afterAll(() => {
        writeResults();
    });
}

// Export public API
module.exports = {
    capture,
    captureMultiple,
    writeResults,
    clearResults,
    getResults,
    setTestName,
    safeSerialize,
    initDatabase,
    // Constants
    LOOP_INDEX,
    OUTPUT_FILE,
    TEST_ITERATION
};
