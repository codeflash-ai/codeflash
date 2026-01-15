/**
 * Codeflash Jest Helper - Unified Test Instrumentation
 *
 * This module provides a unified approach to instrumenting JavaScript tests
 * for both behavior verification and performance measurement.
 *
 * Unlike Python which has separate instrumentation methods for generated
 * vs existing tests, this helper works identically for ALL JavaScript tests.
 *
 * Usage:
 *   const codeflash = require('codeflash-jest-helper');
 *
 *   // Wrap function calls to capture behavior
 *   const result = codeflash.capture('functionName', targetFunction, arg1, arg2);
 *
 * Environment Variables:
 *   CODEFLASH_OUTPUT_FILE - Path to write results (default: /tmp/codeflash_results.bin)
 *   CODEFLASH_LOOP_INDEX - Current benchmark loop iteration (default: 0)
 *   CODEFLASH_MODE - Testing mode: 'behavior' or 'performance' (default: 'behavior')
 */

const fs = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');

// Configuration from environment
const OUTPUT_FILE = process.env.CODEFLASH_OUTPUT_FILE || '/tmp/codeflash_results.bin';
const LOOP_INDEX = parseInt(process.env.CODEFLASH_LOOP_INDEX || '0', 10);
const MODE = process.env.CODEFLASH_MODE || 'behavior';

// Current test context
let currentTestName = null;
let invocationCounter = 0;

// Results buffer
const results = [];

/**
 * Safely serialize a value to JSON.
 * Handles circular references and special types.
 *
 * @param {any} value - Value to serialize
 * @returns {any} - Serializable representation
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
        return serialize(value);
    } catch (e) {
        return { __type: 'SerializationError', error: e.message };
    }
}

/**
 * Record a test result.
 *
 * @param {string} funcName - Name of the function being tested
 * @param {Array} args - Arguments passed to the function
 * @param {any} returnValue - Return value from the function
 * @param {Error|null} error - Error thrown by the function (if any)
 * @param {number} durationNs - Execution time in nanoseconds
 */
function recordResult(funcName, args, returnValue, error, durationNs) {
    const result = {
        testName: currentTestName,
        funcName,
        args: safeSerialize(args),
        returnValue: safeSerialize(returnValue),
        error: error ? {
            name: error.name,
            message: error.message,
            stack: error.stack
        } : null,
        durationNs: Math.round(durationNs),
        invocationId: invocationCounter++,
        loopIndex: LOOP_INDEX,
        mode: MODE,
        timestamp: Date.now()
    };
    results.push(result);
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
 * Write results to output file.
 * Called automatically via Jest afterAll hook.
 */
function writeResults() {
    if (results.length === 0) return;

    try {
        const output = {
            version: '1.0.0',
            mode: MODE,
            loopIndex: LOOP_INDEX,
            timestamp: Date.now(),
            results
        };
        const buffer = Buffer.from(JSON.stringify(output, null, 2));
        fs.writeFileSync(OUTPUT_FILE, buffer);
    } catch (e) {
        console.error('[codeflash] Error writing results:', e.message);
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
    // Constants
    MODE,
    LOOP_INDEX,
    OUTPUT_FILE
};
