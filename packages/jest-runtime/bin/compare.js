#!/usr/bin/env node
/**
 * Codeflash Result Comparator CLI
 *
 * Compares test results between original and optimized code runs.
 *
 * Usage:
 *   codeflash-compare <original_db> <candidate_db>
 *   codeflash-compare --stdin
 */

'use strict';

const {
    readTestResults,
    compareResults,
    compareBuffers,
} = require('../src/compare-results');

// Main entry point
function main() {
    const args = process.argv.slice(2);

    if (args.length === 0) {
        console.error('Usage: codeflash-compare <original_db> <candidate_db>');
        console.error('       codeflash-compare --stdin (reads JSON from stdin)');
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
        console.error('Usage: codeflash-compare <original_db> <candidate_db>');
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

main();
