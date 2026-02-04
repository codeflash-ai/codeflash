/**
 * Codeflash JavaScript Function Tracer
 *
 * This module provides function tracing instrumentation that captures:
 * - Function inputs (arguments)
 * - Return values
 * - Exceptions thrown
 * - Execution time (nanosecond precision)
 *
 * Traces are stored in SQLite database for later replay test generation.
 * This mirrors the Python tracer functionality in codeflash/tracing/.
 *
 * Database Schema (matches Python tracer):
 * - function_calls: Main trace data (type, function, classname, filename, line_number, time_ns, args)
 * - metadata: Key-value metadata about the trace session
 * - pstats: Profiling statistics (optional)
 *
 * Usage:
 *   const tracer = require('codeflash/tracer');
 *   tracer.init('/path/to/output.sqlite', ['/path/to/project']);
 *
 *   // Wrap a function for tracing
 *   const tracedFunc = tracer.wrap(originalFunc, 'funcName', '/path/to/file.js', 10);
 *
 *   // Or use the decorator pattern
 *   tracer.trace('funcName', '/path/to/file.js', 10, () => {
 *       // function body
 *   });
 *
 * Environment Variables:
 *   CODEFLASH_TRACE_DB - Path to SQLite database for storing traces
 *   CODEFLASH_PROJECT_ROOT - Project root for relative path calculation
 *   CODEFLASH_FUNCTIONS - JSON array of functions to trace (optional, traces all if not set)
 *   CODEFLASH_MAX_FUNCTION_COUNT - Maximum traces per function (default: 256)
 *   CODEFLASH_TRACER_TIMEOUT - Timeout in seconds for tracing (optional)
 */

'use strict';

const path = require('path');
const fs = require('fs');

// Load the codeflash serializer for robust value serialization
const serializer = require('./serializer');

// ============================================================================
// CONFIGURATION
// ============================================================================

// Configuration from environment
const TRACE_DB = process.env.CODEFLASH_TRACE_DB;
const PROJECT_ROOT = process.env.CODEFLASH_PROJECT_ROOT || process.cwd();
const MAX_FUNCTION_COUNT = parseInt(process.env.CODEFLASH_MAX_FUNCTION_COUNT || '256', 10);
const TRACER_TIMEOUT = process.env.CODEFLASH_TRACER_TIMEOUT
    ? parseFloat(process.env.CODEFLASH_TRACER_TIMEOUT) * 1000
    : null;

// Parse functions to trace from environment
let FUNCTIONS_TO_TRACE = null;
try {
    if (process.env.CODEFLASH_FUNCTIONS) {
        FUNCTIONS_TO_TRACE = JSON.parse(process.env.CODEFLASH_FUNCTIONS);
    }
} catch (e) {
    console.error('[codeflash-tracer] Failed to parse CODEFLASH_FUNCTIONS:', e.message);
}

// ============================================================================
// STATE
// ============================================================================

// SQLite database (lazy initialized)
let db = null;
let dbInitialized = false;

// Track function call counts for MAX_FUNCTION_COUNT limit
const functionCallCounts = new Map();

// Track start time for timeout
let tracingStartTime = null;

// Track if tracing is enabled
let tracingEnabled = true;

// Address counter for unique call identification
let lastFrameAddress = 0;

// Prepared statements (cached for performance)
let insertCallStmt = null;
let insertMetadataStmt = null;

// ============================================================================
// DATABASE INITIALIZATION
// ============================================================================

/**
 * Initialize the SQLite database for storing traces.
 *
 * @param {string} dbPath - Path to the SQLite database file
 * @returns {boolean} - True if initialization succeeded
 */
function initDatabase(dbPath) {
    if (dbInitialized) return true;
    if (!dbPath) {
        console.error('[codeflash-tracer] No database path provided');
        return false;
    }

    try {
        const Database = require('better-sqlite3');

        // Ensure directory exists
        const dbDir = path.dirname(dbPath);
        if (!fs.existsSync(dbDir)) {
            fs.mkdirSync(dbDir, { recursive: true });
        }

        db = new Database(dbPath);

        // Create tables matching Python tracer schema
        db.exec(`
            CREATE TABLE IF NOT EXISTS function_calls (
                type TEXT,
                function TEXT,
                classname TEXT,
                filename TEXT,
                line_number INTEGER,
                last_frame_address INTEGER,
                time_ns INTEGER,
                args BLOB
            );

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS pstats (
                filename TEXT,
                line_number INTEGER,
                function TEXT,
                class_name TEXT,
                call_count_nonrecursive INTEGER,
                num_callers INTEGER,
                total_time_ns INTEGER,
                cumulative_time_ns INTEGER,
                callers BLOB
            );

            CREATE INDEX IF NOT EXISTS idx_function_calls_function
                ON function_calls(function, filename);
            CREATE INDEX IF NOT EXISTS idx_function_calls_time
                ON function_calls(time_ns);
        `);

        // Prepare statements for performance
        insertCallStmt = db.prepare(`
            INSERT INTO function_calls (type, function, classname, filename, line_number, last_frame_address, time_ns, args)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        `);

        insertMetadataStmt = db.prepare(`
            INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)
        `);

        // Record metadata
        insertMetadataStmt.run('tracer_version', '1.0.0');
        insertMetadataStmt.run('language', 'javascript');
        insertMetadataStmt.run('project_root', PROJECT_ROOT);
        insertMetadataStmt.run('node_version', process.version);
        insertMetadataStmt.run('start_time', new Date().toISOString());

        dbInitialized = true;
        tracingStartTime = Date.now();

        return true;
    } catch (e) {
        console.error('[codeflash-tracer] Failed to initialize database:', e.message);
        return false;
    }
}

/**
 * Close the database and finalize traces.
 */
function closeDatabase() {
    if (db) {
        try {
            // Record end time
            if (insertMetadataStmt) {
                insertMetadataStmt.run('end_time', new Date().toISOString());
                insertMetadataStmt.run('total_traces', getTotalTraceCount());
            }
            db.close();
        } catch (e) {
            console.error('[codeflash-tracer] Error closing database:', e.message);
        }
        db = null;
        dbInitialized = false;
        insertCallStmt = null;
        insertMetadataStmt = null;
    }
}

// ============================================================================
// TIMING UTILITIES
// ============================================================================

/**
 * Get high-resolution time in nanoseconds.
 *
 * @returns {bigint} - Time in nanoseconds
 */
function getTimeNs() {
    return process.hrtime.bigint();
}

/**
 * Calculate duration in nanoseconds.
 *
 * @param {bigint} start - Start time in nanoseconds
 * @param {bigint} end - End time in nanoseconds
 * @returns {number} - Duration in nanoseconds (as Number for SQLite compatibility)
 */
function getDurationNs(start, end) {
    return Number(end - start);
}

// ============================================================================
// TRACING UTILITIES
// ============================================================================

/**
 * Check if tracing is still enabled (not timed out or disabled).
 *
 * @returns {boolean} - True if tracing should continue
 */
function isTracingEnabled() {
    if (!tracingEnabled) return false;

    if (TRACER_TIMEOUT && tracingStartTime) {
        const elapsed = Date.now() - tracingStartTime;
        if (elapsed >= TRACER_TIMEOUT) {
            console.log('[codeflash-tracer] Tracing timeout reached, stopping tracer');
            tracingEnabled = false;
            return false;
        }
    }

    return true;
}

/**
 * Check if a function should be traced based on configuration.
 *
 * @param {string} funcName - Function name
 * @param {string} fileName - File path
 * @param {string|null} className - Class name (for methods)
 * @returns {boolean} - True if function should be traced
 */
function shouldTraceFunction(funcName, fileName, className = null) {
    if (!isTracingEnabled()) return false;

    // Check if we've exceeded the max call count for this function
    const key = `${fileName}:${className || ''}:${funcName}`;
    const count = functionCallCounts.get(key) || 0;
    if (count >= MAX_FUNCTION_COUNT) {
        return false;
    }

    // Check if function is in the filter list
    if (FUNCTIONS_TO_TRACE && FUNCTIONS_TO_TRACE.length > 0) {
        // Check by function name only, or by full qualified name
        const matchesName = FUNCTIONS_TO_TRACE.some(f => {
            if (typeof f === 'string') {
                return f === funcName || f === `${className}.${funcName}`;
            }
            // Support object format: { function: 'name', file: 'path', class: 'className' }
            if (typeof f === 'object') {
                if (f.function && f.function !== funcName) return false;
                if (f.file && !fileName.includes(f.file)) return false;
                if (f.class && f.class !== className) return false;
                return true;
            }
            return false;
        });
        if (!matchesName) return false;
    }

    return true;
}

/**
 * Increment the call count for a function.
 *
 * @param {string} funcName - Function name
 * @param {string} fileName - File path
 * @param {string|null} className - Class name (for methods)
 */
function incrementCallCount(funcName, fileName, className = null) {
    const key = `${fileName}:${className || ''}:${funcName}`;
    const count = functionCallCounts.get(key) || 0;
    functionCallCounts.set(key, count + 1);
}

/**
 * Get total trace count across all functions.
 *
 * @returns {number} - Total number of traces
 */
function getTotalTraceCount() {
    let total = 0;
    for (const count of functionCallCounts.values()) {
        total += count;
    }
    return total;
}

/**
 * Safely serialize arguments for storage.
 *
 * @param {Array} args - Arguments to serialize
 * @returns {Buffer} - Serialized arguments
 */
function serializeArgs(args) {
    try {
        return serializer.serialize(args);
    } catch (e) {
        console.warn('[codeflash-tracer] Serialization failed:', e.message);
        return Buffer.from(JSON.stringify({ __error__: 'SerializationError', message: e.message }));
    }
}

// ============================================================================
// CORE TRACING API
// ============================================================================

/**
 * Record a function call trace.
 *
 * @param {string} type - Event type ('call' or 'return')
 * @param {string} funcName - Function name
 * @param {string} fileName - File path
 * @param {number} lineNumber - Line number
 * @param {string|null} className - Class name (for methods)
 * @param {bigint} timeNs - Timestamp in nanoseconds
 * @param {any} argsOrResult - Arguments (for 'call') or return value (for 'return')
 */
function recordTrace(type, funcName, fileName, lineNumber, className, timeNs, argsOrResult) {
    if (!dbInitialized) {
        if (!initDatabase(TRACE_DB)) {
            return;
        }
    }

    try {
        const serializedData = serializeArgs(argsOrResult);
        const frameAddress = lastFrameAddress++;

        insertCallStmt.run(
            type,
            funcName,
            className,
            fileName,
            lineNumber,
            frameAddress,
            Number(timeNs),
            serializedData
        );
    } catch (e) {
        console.error('[codeflash-tracer] Failed to record trace:', e.message);
    }
}

/**
 * Wrap a function with tracing instrumentation.
 *
 * @param {Function} fn - The function to wrap
 * @param {string} funcName - Function name
 * @param {string} fileName - File path
 * @param {number} lineNumber - Line number
 * @param {string|null} className - Class name (for methods)
 * @returns {Function} - Wrapped function
 */
function wrap(fn, funcName, fileName, lineNumber, className = null) {
    // Don't wrap if function shouldn't be traced
    if (typeof fn !== 'function') {
        return fn;
    }

    // Check if it's an async function
    const isAsync = fn.constructor.name === 'AsyncFunction';

    if (isAsync) {
        return async function codeflashTracedAsync(...args) {
            if (!shouldTraceFunction(funcName, fileName, className)) {
                return fn.apply(this, args);
            }

            incrementCallCount(funcName, fileName, className);
            const startTime = getTimeNs();

            // Record call
            recordTrace('call', funcName, fileName, lineNumber, className, startTime, args);

            try {
                const result = await fn.apply(this, args);
                return result;
            } catch (error) {
                throw error;
            }
        };
    }

    return function codeflashTraced(...args) {
        if (!shouldTraceFunction(funcName, fileName, className)) {
            return fn.apply(this, args);
        }

        incrementCallCount(funcName, fileName, className);
        const startTime = getTimeNs();

        // Record call
        recordTrace('call', funcName, fileName, lineNumber, className, startTime, args);

        try {
            const result = fn.apply(this, args);

            // Handle promise returns from non-async functions
            if (result instanceof Promise) {
                return result.then(
                    (resolved) => resolved,
                    (error) => { throw error; }
                );
            }

            return result;
        } catch (error) {
            throw error;
        }
    };
}

/**
 * Create a wrapper function factory for use with Babel transformation.
 *
 * @param {string} funcName - Function name
 * @param {string} fileName - File path
 * @param {number} lineNumber - Line number
 * @param {string|null} className - Class name (for methods)
 * @returns {Function} - A function that takes the original function and returns a wrapped version
 */
function createWrapper(funcName, fileName, lineNumber, className = null) {
    return function(fn) {
        return wrap(fn, funcName, fileName, lineNumber, className);
    };
}

/**
 * Initialize the tracer.
 *
 * @param {string} dbPath - Path to SQLite database
 * @param {string} projectRoot - Project root path
 */
function init(dbPath, projectRoot) {
    if (projectRoot) {
        process.env.CODEFLASH_PROJECT_ROOT = projectRoot;
    }
    initDatabase(dbPath || TRACE_DB);
}

/**
 * Disable tracing.
 */
function disable() {
    tracingEnabled = false;
}

/**
 * Enable tracing.
 */
function enable() {
    tracingEnabled = true;
}

/**
 * Get tracing statistics.
 *
 * @returns {Object} - Statistics object
 */
function getStats() {
    return {
        totalTraces: getTotalTraceCount(),
        functionCounts: Object.fromEntries(functionCallCounts),
        tracingEnabled,
        dbInitialized,
    };
}

// ============================================================================
// PROCESS EXIT HANDLER
// ============================================================================

// Ensure database is closed on process exit
process.on('exit', () => {
    closeDatabase();
});

process.on('SIGINT', () => {
    closeDatabase();
    process.exit(0);
});

process.on('SIGTERM', () => {
    closeDatabase();
    process.exit(0);
});

// Also handle uncaught exceptions
process.on('uncaughtException', (err) => {
    console.error('[codeflash-tracer] Uncaught exception:', err);
    closeDatabase();
    process.exit(1);
});

// ============================================================================
// EXPORTS
// ============================================================================

module.exports = {
    // Core API
    init,
    wrap,
    createWrapper,
    recordTrace,

    // Control
    disable,
    enable,
    isTracingEnabled,

    // Database
    initDatabase,
    closeDatabase,

    // Utilities
    shouldTraceFunction,
    getStats,
    serializeArgs,
    getTimeNs,
    getDurationNs,

    // Configuration
    TRACE_DB,
    PROJECT_ROOT,
    MAX_FUNCTION_COUNT,
    FUNCTIONS_TO_TRACE,
};
