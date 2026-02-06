/**
 * Codeflash Replay Test Utilities
 *
 * This module provides utilities for generating and running replay tests
 * from traced function calls. Replay tests allow verifying that optimized
 * code produces the same results as the original code.
 *
 * Usage:
 *   const { getNextArg, createReplayTest } = require('codeflash/replay');
 *
 *   // In a test file:
 *   describe('Replay tests', () => {
 *       test.each(getNextArg(traceFile, 'myFunction', '/path/file.js', 25))
 *       ('myFunction replay %#', (args) => {
 *           myFunction(...args);
 *       });
 *   });
 *
 * The module supports both Jest and Vitest test frameworks.
 */

'use strict';

const path = require('path');
const fs = require('fs');

// Load the codeflash serializer for argument deserialization
const serializer = require('./serializer');

// ============================================================================
// DATABASE ACCESS
// ============================================================================

/**
 * Open a SQLite database connection.
 *
 * @param {string} dbPath - Path to the SQLite database
 * @returns {Object|null} - Database connection or null if failed
 */
function openDatabase(dbPath) {
    try {
        const Database = require('better-sqlite3');
        return new Database(dbPath, { readonly: true });
    } catch (e) {
        console.error('[codeflash-replay] Failed to open database:', e.message);
        return null;
    }
}

/**
 * Get traced function calls from the database.
 *
 * @param {string} traceFile - Path to the trace SQLite database
 * @param {string} functionName - Name of the function
 * @param {string} fileName - Path to the source file
 * @param {string|null} className - Class name (for methods)
 * @param {number} limit - Maximum number of traces to retrieve
 * @returns {Array} - Array of traced arguments
 */
function getNextArg(traceFile, functionName, fileName, limit = 25, className = null) {
    const db = openDatabase(traceFile);
    if (!db) {
        return [];
    }

    try {
        let stmt;
        let rows;

        if (className) {
            stmt = db.prepare(`
                SELECT args FROM function_calls
                WHERE function = ? AND filename = ? AND classname = ? AND type = 'call'
                ORDER BY time_ns ASC
                LIMIT ?
            `);
            rows = stmt.all(functionName, fileName, className, limit);
        } else {
            stmt = db.prepare(`
                SELECT args FROM function_calls
                WHERE function = ? AND filename = ? AND type = 'call'
                ORDER BY time_ns ASC
                LIMIT ?
            `);
            rows = stmt.all(functionName, fileName, limit);
        }

        db.close();

        // Deserialize arguments
        return rows.map((row, index) => {
            try {
                const args = serializer.deserialize(row.args);
                return args;
            } catch (e) {
                console.warn(`[codeflash-replay] Failed to deserialize args at index ${index}:`, e.message);
                return [];
            }
        });
    } catch (e) {
        console.error('[codeflash-replay] Database query failed:', e.message);
        db.close();
        return [];
    }
}

/**
 * Get traced function calls with full metadata.
 *
 * @param {string} traceFile - Path to the trace SQLite database
 * @param {string} functionName - Name of the function
 * @param {string} fileName - Path to the source file
 * @param {string|null} className - Class name (for methods)
 * @param {number} limit - Maximum number of traces to retrieve
 * @returns {Array} - Array of trace objects with args and metadata
 */
function getTracesWithMetadata(traceFile, functionName, fileName, limit = 25, className = null) {
    const db = openDatabase(traceFile);
    if (!db) {
        return [];
    }

    try {
        let stmt;
        let rows;

        if (className) {
            stmt = db.prepare(`
                SELECT type, function, classname, filename, line_number, time_ns, args
                FROM function_calls
                WHERE function = ? AND filename = ? AND classname = ? AND type = 'call'
                ORDER BY time_ns ASC
                LIMIT ?
            `);
            rows = stmt.all(functionName, fileName, className, limit);
        } else {
            stmt = db.prepare(`
                SELECT type, function, classname, filename, line_number, time_ns, args
                FROM function_calls
                WHERE function = ? AND filename = ? AND type = 'call'
                ORDER BY time_ns ASC
                LIMIT ?
            `);
            rows = stmt.all(functionName, fileName, limit);
        }

        db.close();

        // Deserialize arguments and return with metadata
        return rows.map((row, index) => {
            let args;
            try {
                args = serializer.deserialize(row.args);
            } catch (e) {
                console.warn(`[codeflash-replay] Failed to deserialize args at index ${index}:`, e.message);
                args = [];
            }

            return {
                args,
                function: row.function,
                className: row.classname,
                fileName: row.filename,
                lineNumber: row.line_number,
                timeNs: row.time_ns,
            };
        });
    } catch (e) {
        console.error('[codeflash-replay] Database query failed:', e.message);
        db.close();
        return [];
    }
}

/**
 * Get all traced functions from the database.
 *
 * @param {string} traceFile - Path to the trace SQLite database
 * @returns {Array} - Array of { function, fileName, className, count } objects
 */
function getTracedFunctions(traceFile) {
    const db = openDatabase(traceFile);
    if (!db) {
        return [];
    }

    try {
        const stmt = db.prepare(`
            SELECT function, filename, classname, COUNT(*) as count
            FROM function_calls
            WHERE type = 'call'
            GROUP BY function, filename, classname
            ORDER BY count DESC
        `);
        const rows = stmt.all();
        db.close();

        return rows.map(row => ({
            function: row.function,
            fileName: row.filename,
            className: row.classname,
            count: row.count,
        }));
    } catch (e) {
        console.error('[codeflash-replay] Failed to get traced functions:', e.message);
        db.close();
        return [];
    }
}

/**
 * Get metadata from the trace database.
 *
 * @param {string} traceFile - Path to the trace SQLite database
 * @returns {Object} - Metadata key-value pairs
 */
function getTraceMetadata(traceFile) {
    const db = openDatabase(traceFile);
    if (!db) {
        return {};
    }

    try {
        const stmt = db.prepare('SELECT key, value FROM metadata');
        const rows = stmt.all();
        db.close();

        const metadata = {};
        for (const row of rows) {
            metadata[row.key] = row.value;
        }
        return metadata;
    } catch (e) {
        console.error('[codeflash-replay] Failed to get metadata:', e.message);
        db.close();
        return {};
    }
}

// ============================================================================
// TEST GENERATION
// ============================================================================

/**
 * Generate a Jest/Vitest replay test file.
 *
 * @param {string} traceFile - Path to the trace SQLite database
 * @param {Array} functions - Array of { function, fileName, className, modulePath } to test
 * @param {Object} options - Generation options
 * @returns {string} - Generated test file content
 */
function generateReplayTest(traceFile, functions, options = {}) {
    const {
        framework = 'jest', // 'jest' or 'vitest'
        maxRunCount = 100,
        outputPath = null,
    } = options;

    const isVitest = framework === 'vitest';

    // Build imports section
    const imports = [];

    if (isVitest) {
        imports.push("import { describe, test } from 'vitest';");
    }

    imports.push("const { getNextArg } = require('codeflash/replay');");
    imports.push('');

    // Build function imports
    for (const func of functions) {
        const alias = getFunctionAlias(func.modulePath, func.function, func.className);

        if (func.className) {
            // Import class for method testing
            imports.push(`const { ${func.className}: ${alias}_class } = require('${func.modulePath}');`);
        } else {
            // Import function directly
            imports.push(`const { ${func.function}: ${alias} } = require('${func.modulePath}');`);
        }
    }

    imports.push('');

    // Metadata
    const metadata = [
        `const traceFilePath = '${traceFile}';`,
        `const functions = ${JSON.stringify(functions.map(f => f.function))};`,
        '',
    ];

    // Build test cases
    const testCases = [];

    for (const func of functions) {
        const alias = getFunctionAlias(func.modulePath, func.function, func.className);
        const testName = func.className
            ? `${func.className}.${func.function}`
            : func.function;

        if (func.className) {
            // Method test
            testCases.push(`
describe('Replay: ${testName}', () => {
    const traces = getNextArg(traceFilePath, '${func.function}', '${func.fileName}', ${maxRunCount}, '${func.className}');

    test.each(traces.map((args, i) => [i, args]))('call %i', (index, args) => {
        // For instance methods, first arg is 'this' context
        const [thisArg, ...methodArgs] = args;
        const instance = thisArg || new ${alias}_class();
        instance.${func.function}(...methodArgs);
    });
});
`);
        } else {
            // Function test
            testCases.push(`
describe('Replay: ${testName}', () => {
    const traces = getNextArg(traceFilePath, '${func.function}', '${func.fileName}', ${maxRunCount});

    test.each(traces.map((args, i) => [i, args]))('call %i', (index, args) => {
        ${alias}(...args);
    });
});
`);
        }
    }

    // Combine all parts
    const content = [
        '// Auto-generated replay test by Codeflash',
        '// Do not edit this file directly',
        '',
        ...imports,
        ...metadata,
        ...testCases,
    ].join('\n');

    // Write to file if outputPath provided
    if (outputPath) {
        const dir = path.dirname(outputPath);
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
        fs.writeFileSync(outputPath, content);
        console.log(`[codeflash-replay] Generated test file: ${outputPath}`);
    }

    return content;
}

/**
 * Create a function alias for imports to avoid naming conflicts.
 *
 * @param {string} modulePath - Module path
 * @param {string} functionName - Function name
 * @param {string|null} className - Class name
 * @returns {string} - Alias name
 */
function getFunctionAlias(modulePath, functionName, className = null) {
    // Normalize module path to valid identifier
    const moduleAlias = modulePath
        .replace(/[^a-zA-Z0-9]/g, '_')
        .replace(/^_+|_+$/g, '');

    if (className) {
        return `${moduleAlias}_${className}_${functionName}`;
    }
    return `${moduleAlias}_${functionName}`;
}

/**
 * Create replay tests from a trace file.
 * This is the main entry point for Python integration.
 *
 * @param {string} traceFile - Path to the trace SQLite database
 * @param {string} outputPath - Path to write the test file
 * @param {Object} options - Generation options
 * @returns {Object} - { success, outputPath, functions }
 */
function createReplayTestFromTrace(traceFile, outputPath, options = {}) {
    const {
        framework = 'jest',
        maxRunCount = 100,
        projectRoot = process.cwd(),
    } = options;

    // Get all traced functions
    const tracedFunctions = getTracedFunctions(traceFile);

    if (tracedFunctions.length === 0) {
        console.warn('[codeflash-replay] No traced functions found in database');
        return { success: false, outputPath: null, functions: [] };
    }

    // Convert to the format expected by generateReplayTest
    const functions = tracedFunctions.map(tf => {
        // Calculate module path from file name
        let modulePath = tf.fileName;

        // Make relative to project root
        if (path.isAbsolute(modulePath)) {
            modulePath = path.relative(projectRoot, modulePath);
        }

        // Convert to module path (remove .js extension, use forward slashes)
        modulePath = './' + modulePath
            .replace(/\\/g, '/')
            .replace(/\.js$/, '')
            .replace(/\.ts$/, '');

        return {
            function: tf.function,
            fileName: tf.fileName,
            className: tf.className,
            modulePath,
        };
    });

    // Generate the test file
    const testContent = generateReplayTest(traceFile, functions, {
        framework,
        maxRunCount,
        outputPath,
    });

    return {
        success: true,
        outputPath,
        functions: functions.map(f => f.function),
        content: testContent,
    };
}

// ============================================================================
// EXPORTS
// ============================================================================

module.exports = {
    // Core API
    getNextArg,
    getTracesWithMetadata,
    getTracedFunctions,
    getTraceMetadata,

    // Test generation
    generateReplayTest,
    createReplayTestFromTrace,
    getFunctionAlias,

    // Database utilities
    openDatabase,
};
