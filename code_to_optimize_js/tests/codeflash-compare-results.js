#!/usr/bin/env node
/**
 * Codeflash Result Comparator
 *
 * This script compares test results between original and optimized code runs.
 * It reads serialized behavior data from SQLite databases and compares them
 * using the codeflash-comparator in JavaScript land.
 *
 * Usage:
 *   node codeflash-compare-results.js <original_db> <candidate_db>
 *   node codeflash-compare-results.js --json <json_input>
 *
 * Output (JSON):
 *   {
 *     "equivalent": true/false,
 *     "diffs": [
 *       {
 *         "invocation_id": "...",
 *         "scope": "return_value|stdout|did_pass",
 *         "original": "...",
 *         "candidate": "..."
 *       }
 *     ],
 *     "error": null | "error message"
 *   }
 */

const fs = require('fs');
const path = require('path');

// Import our modules
const { deserialize } = require('./codeflash-serializer');
const { comparator } = require('./codeflash-comparator');

// Try to load better-sqlite3
let Database;
try {
    Database = require('better-sqlite3');
} catch (e) {
    // Use console.log (stdout) for JSON output, not console.error (stderr)
    // Exit code 2 indicates a setup error (distinct from 1 = "not equivalent")
    console.log(JSON.stringify({
        equivalent: false,
        diffs: [],
        error: 'better-sqlite3 not installed. Run: npm install better-sqlite3'
    }));
    process.exit(2);
}

/**
 * Read test results from a SQLite database.
 *
 * @param {string} dbPath - Path to SQLite database
 * @returns {Map<string, object>} Map of invocation_id -> result object
 */
function readTestResults(dbPath) {
    const results = new Map();

    if (!fs.existsSync(dbPath)) {
        throw new Error(`Database not found: ${dbPath}`);
    }

    const db = new Database(dbPath, { readonly: true });

    try {
        const stmt = db.prepare(`
            SELECT
                test_module_path,
                test_class_name,
                test_function_name,
                function_getting_tested,
                loop_index,
                iteration_id,
                runtime,
                return_value,
                verification_type
            FROM test_results
            WHERE loop_index = 1
        `);

        for (const row of stmt.iterate()) {
            // Build unique invocation ID (matches Python's format)
            const invocationId = `${row.loop_index}:${row.test_module_path}:${row.test_class_name || ''}:${row.test_function_name}:${row.function_getting_tested}:${row.iteration_id}`;

            // Deserialize the return value
            let returnValue = null;
            if (row.return_value) {
                try {
                    returnValue = deserialize(row.return_value);
                } catch (e) {
                    console.error(`Failed to deserialize result for ${invocationId}: ${e.message}`);
                }
            }

            results.set(invocationId, {
                testModulePath: row.test_module_path,
                testClassName: row.test_class_name,
                testFunctionName: row.test_function_name,
                functionGettingTested: row.function_getting_tested,
                loopIndex: row.loop_index,
                iterationId: row.iteration_id,
                runtime: row.runtime,
                returnValue,
                verificationType: row.verification_type,
            });
        }
    } finally {
        db.close();
    }

    return results;
}

/**
 * Compare two sets of test results.
 *
 * @param {Map<string, object>} originalResults - Results from original code
 * @param {Map<string, object>} candidateResults - Results from optimized code
 * @returns {object} Comparison result
 */
function compareResults(originalResults, candidateResults) {
    const diffs = [];
    let allEquivalent = true;

    // Get all unique invocation IDs
    const allIds = new Set([...originalResults.keys(), ...candidateResults.keys()]);

    for (const invocationId of allIds) {
        const original = originalResults.get(invocationId);
        const candidate = candidateResults.get(invocationId);

        // If candidate has extra results not in original, that's OK
        if (candidate && !original) {
            continue;
        }

        // If original has results not in candidate, that's a diff
        if (original && !candidate) {
            allEquivalent = false;
            diffs.push({
                invocation_id: invocationId,
                scope: 'missing',
                original: summarizeValue(original.returnValue),
                candidate: null,
                test_info: {
                    test_module_path: original.testModulePath,
                    test_function_name: original.testFunctionName,
                    function_getting_tested: original.functionGettingTested,
                }
            });
            continue;
        }

        // Compare return values using the JavaScript comparator
        // The return value format is [args, kwargs, returnValue] (behavior tuple)
        const originalValue = original.returnValue;
        const candidateValue = candidate.returnValue;

        const isEqual = comparator(originalValue, candidateValue);

        if (!isEqual) {
            allEquivalent = false;
            diffs.push({
                invocation_id: invocationId,
                scope: 'return_value',
                original: summarizeValue(originalValue),
                candidate: summarizeValue(candidateValue),
                test_info: {
                    test_module_path: original.testModulePath,
                    test_function_name: original.testFunctionName,
                    function_getting_tested: original.functionGettingTested,
                }
            });
        }
    }

    return {
        equivalent: allEquivalent,
        diffs,
        total_invocations: allIds.size,
        original_count: originalResults.size,
        candidate_count: candidateResults.size,
    };
}

/**
 * Create a summary of a value for diff reporting.
 * Truncates long values to avoid huge output.
 *
 * @param {any} value - Value to summarize
 * @returns {string} String representation
 */
function summarizeValue(value, maxLength = 200) {
    try {
        let str;
        if (value === undefined) {
            str = 'undefined';
        } else if (value === null) {
            str = 'null';
        } else if (typeof value === 'function') {
            str = `[Function: ${value.name || 'anonymous'}]`;
        } else if (value instanceof Map) {
            str = `Map(${value.size}) { ${[...value.entries()].slice(0, 3).map(([k, v]) => `${summarizeValue(k, 50)} => ${summarizeValue(v, 50)}`).join(', ')}${value.size > 3 ? ', ...' : ''} }`;
        } else if (value instanceof Set) {
            str = `Set(${value.size}) { ${[...value].slice(0, 3).map(v => summarizeValue(v, 50)).join(', ')}${value.size > 3 ? ', ...' : ''} }`;
        } else if (value instanceof Date) {
            str = value.toISOString();
        } else if (Array.isArray(value)) {
            if (value.length <= 5) {
                str = JSON.stringify(value);
            } else {
                str = `[${value.slice(0, 3).map(v => summarizeValue(v, 50)).join(', ')}, ... (${value.length} items)]`;
            }
        } else if (typeof value === 'object') {
            str = JSON.stringify(value);
        } else {
            str = String(value);
        }

        if (str.length > maxLength) {
            return str.slice(0, maxLength - 3) + '...';
        }
        return str;
    } catch (e) {
        return `[Unable to stringify: ${e.message}]`;
    }
}

/**
 * Compare results from serialized buffers directly (for stdin input).
 *
 * @param {Buffer} originalBuffer - Serialized original result
 * @param {Buffer} candidateBuffer - Serialized candidate result
 * @returns {boolean} True if equivalent
 */
function compareBuffers(originalBuffer, candidateBuffer) {
    try {
        const original = deserialize(originalBuffer);
        const candidate = deserialize(candidateBuffer);
        return comparator(original, candidate);
    } catch (e) {
        console.error(`Comparison error: ${e.message}`);
        return false;
    }
}

/**
 * Main entry point.
 */
function main() {
    const args = process.argv.slice(2);

    if (args.length === 0) {
        console.error('Usage: node codeflash-compare-results.js <original_db> <candidate_db>');
        console.error('       node codeflash-compare-results.js --stdin (reads JSON from stdin)');
        process.exit(1);
    }

    // Handle stdin mode for programmatic use
    if (args[0] === '--stdin') {
        let input = '';
        process.stdin.setEncoding('utf8');
        process.stdin.on('data', chunk => input += chunk);
        process.stdin.on('end', () => {
            try {
                const data = JSON.parse(input);
                const originalBuffer = Buffer.from(data.original, 'base64');
                const candidateBuffer = Buffer.from(data.candidate, 'base64');
                const isEqual = compareBuffers(originalBuffer, candidateBuffer);
                console.log(JSON.stringify({ equivalent: isEqual, error: null }));
            } catch (e) {
                console.log(JSON.stringify({ equivalent: false, error: e.message }));
            }
        });
        return;
    }

    // Standard mode: compare two SQLite databases
    if (args.length < 2) {
        console.error('Usage: node codeflash-compare-results.js <original_db> <candidate_db>');
        process.exit(1);
    }

    const [originalDb, candidateDb] = args;

    try {
        const originalResults = readTestResults(originalDb);
        const candidateResults = readTestResults(candidateDb);

        const comparison = compareResults(originalResults, candidateResults);

        // Limit the number of diffs to avoid huge output
        const MAX_DIFFS = 50;
        if (comparison.diffs.length > MAX_DIFFS) {
            const truncatedCount = comparison.diffs.length - MAX_DIFFS;
            comparison.diffs = comparison.diffs.slice(0, MAX_DIFFS);
            comparison.diffs_truncated = truncatedCount;
        }

        // Use compact JSON (no pretty-printing) to reduce output size
        console.log(JSON.stringify(comparison));
        process.exit(comparison.equivalent ? 0 : 1);
    } catch (e) {
        console.log(JSON.stringify({
            equivalent: false,
            diffs: [],
            error: e.message
        }));
        process.exit(1);
    }
}

// Export for programmatic use
module.exports = {
    readTestResults,
    compareResults,
    compareBuffers,
    summarizeValue,
};

// Run if called directly
if (require.main === module) {
    main();
}
