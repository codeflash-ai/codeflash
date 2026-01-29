/**
 * Custom Line Profiler Implementation
 *
 * This profiler instruments JavaScript code by inserting timing calls
 * between each line to measure execution time per line.
 *
 * Approach: Insert process.hrtime.bigint() calls before and after each statement.
 */

const fs = require('fs');
const path = require('path');

// Global timing data storage
const lineTimings = new Map();  // Map<filename, Map<lineNumber, {count, totalNs}>>

// High-resolution timer
function startTimer() {
    return process.hrtime.bigint();
}

function endTimer(start) {
    return process.hrtime.bigint() - start;
}

/**
 * Record timing for a specific line.
 */
function recordLineTiming(filename, lineNumber, durationNs) {
    if (!lineTimings.has(filename)) {
        lineTimings.set(filename, new Map());
    }
    const fileTimings = lineTimings.get(filename);
    if (!fileTimings.has(lineNumber)) {
        fileTimings.set(lineNumber, { count: 0, totalNs: BigInt(0) });
    }
    const timing = fileTimings.get(lineNumber);
    timing.count++;
    timing.totalNs += durationNs;
}

/**
 * Get all recorded timings.
 */
function getTimings() {
    const result = {};
    for (const [filename, fileTimings] of lineTimings) {
        result[filename] = {};
        for (const [lineNumber, data] of fileTimings) {
            result[filename][lineNumber] = {
                count: data.count,
                totalNs: Number(data.totalNs),
                avgNs: data.count > 0 ? Number(data.totalNs / BigInt(data.count)) : 0
            };
        }
    }
    return result;
}

/**
 * Clear all recorded timings.
 */
function clearTimings() {
    lineTimings.clear();
}

/**
 * Simple AST-free instrumentation using regex.
 * This is a simplified approach that works for common patterns.
 */
function instrumentFunction(funcSource, funcName, filename) {
    const lines = funcSource.split('\n');
    const instrumentedLines = [];

    // Track block depth for proper instrumentation
    let inFunction = false;
    let braceDepth = 0;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const lineNum = i + 1;
        const trimmed = line.trim();

        // Skip empty lines and comments
        if (!trimmed || trimmed.startsWith('//') || trimmed.startsWith('/*') || trimmed.startsWith('*')) {
            instrumentedLines.push(line);
            continue;
        }

        // Detect function start
        if (trimmed.includes('function') || trimmed.match(/^\s*(const|let|var)\s+\w+\s*=\s*(async\s*)?\(/)) {
            inFunction = true;
        }

        // Track braces
        const openBraces = (line.match(/{/g) || []).length;
        const closeBraces = (line.match(/}/g) || []).length;
        braceDepth += openBraces - closeBraces;

        // Skip lines that are just braces, function declarations, or control structures without body
        if (trimmed === '{' || trimmed === '}' ||
            trimmed.match(/^(function|if|else|for|while|switch|try|catch|finally)\s*[\({]?$/) ||
            trimmed.match(/^}\s*(else|catch|finally)/) ||
            trimmed.endsWith('{')) {
            instrumentedLines.push(line);
            continue;
        }

        // Don't instrument return statements that are just `return;`
        if (trimmed === 'return;') {
            instrumentedLines.push(line);
            continue;
        }

        // Add timing instrumentation
        const indent = line.match(/^(\s*)/)[1];
        const timerVar = `__t${lineNum}`;

        // Wrap the line with timing
        instrumentedLines.push(`${indent}const ${timerVar} = __profiler.startTimer();`);
        instrumentedLines.push(line);
        instrumentedLines.push(`${indent}__profiler.recordLineTiming('${filename}', ${lineNum}, __profiler.endTimer(${timerVar}));`);
    }

    return instrumentedLines.join('\n');
}

/**
 * More sophisticated instrumentation using a proper parser approach.
 * This creates wrapper functions that time each statement.
 */
function createProfiledVersion(originalFunc, funcName, filename) {
    // Get the source code
    const source = originalFunc.toString();

    // Parse out the function body (simplified)
    const bodyMatch = source.match(/\{([\s\S]*)\}$/);
    if (!bodyMatch) {
        console.error('Could not parse function body');
        return originalFunc;
    }

    const body = bodyMatch[1];
    const lines = body.split('\n');
    const instrumentedLines = [];

    // Get the function signature
    const sigMatch = source.match(/^((?:async\s+)?function\s*\w*\s*\([^)]*\)|(?:async\s+)?\([^)]*\)\s*=>|\([^)]*\)\s*=>)/);
    const signature = sigMatch ? sigMatch[1] : 'function()';

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const lineNum = i + 1;
        const trimmed = line.trim();

        // Skip empty lines, comments, braces only
        if (!trimmed || trimmed.startsWith('//') || trimmed === '{' || trimmed === '}') {
            instrumentedLines.push(line);
            continue;
        }

        // Check if this is a statement that should be timed
        if (isTimableStatement(trimmed)) {
            const indent = line.match(/^(\s*)/)[1];
            const timerVar = `__t${lineNum}`;

            // Handle return statements specially
            if (trimmed.startsWith('return ')) {
                const returnExpr = trimmed.slice(7).replace(/;$/, '');
                instrumentedLines.push(`${indent}const ${timerVar} = __profiler.startTimer();`);
                instrumentedLines.push(`${indent}const __retVal${lineNum} = ${returnExpr};`);
                instrumentedLines.push(`${indent}__profiler.recordLineTiming('${filename}', ${lineNum}, __profiler.endTimer(${timerVar}));`);
                instrumentedLines.push(`${indent}return __retVal${lineNum};`);
            } else {
                instrumentedLines.push(`${indent}const ${timerVar} = __profiler.startTimer();`);
                instrumentedLines.push(line);
                instrumentedLines.push(`${indent}__profiler.recordLineTiming('${filename}', ${lineNum}, __profiler.endTimer(${timerVar}));`);
            }
        } else {
            instrumentedLines.push(line);
        }
    }

    // Reconstruct the function
    const instrumentedBody = instrumentedLines.join('\n');
    const instrumentedSource = `${signature} {\n${instrumentedBody}\n}`;

    // Create the new function with profiler in scope
    try {
        const wrappedFunc = new Function('__profiler', `return ${instrumentedSource}`);
        return wrappedFunc({
            startTimer,
            endTimer,
            recordLineTiming
        });
    } catch (e) {
        console.error('Failed to create instrumented function:', e.message);
        return originalFunc;
    }
}

function isTimableStatement(line) {
    // Skip control flow keywords (will time the body instead)
    if (line.match(/^(if|else|for|while|switch|case|default|try|catch|finally|do)\s*[\({]?/)) {
        return false;
    }
    // Skip braces and empty returns
    if (line === '{' || line === '}' || line === 'return;') {
        return false;
    }
    // Time everything else
    return true;
}

/**
 * Alternative approach: Manual instrumentation with explicit timing points.
 * This is the most accurate but requires more setup.
 */
function createManuallyInstrumentedFibonacci() {
    return function fibonacci_instrumented(n) {
        const timings = {};
        let t;

        // Line 1: if (n <= 1) return n;
        t = process.hrtime.bigint();
        const cond1 = n <= 1;
        recordLineTiming('fibonacci', 1, process.hrtime.bigint() - t);
        if (cond1) {
            t = process.hrtime.bigint();
            const ret = n;
            recordLineTiming('fibonacci', 1, process.hrtime.bigint() - t);
            return ret;
        }

        // Line 2: let a = 0;
        t = process.hrtime.bigint();
        let a = 0;
        recordLineTiming('fibonacci', 2, process.hrtime.bigint() - t);

        // Line 3: let b = 1;
        t = process.hrtime.bigint();
        let b = 1;
        recordLineTiming('fibonacci', 3, process.hrtime.bigint() - t);

        // Line 4-7: for loop
        t = process.hrtime.bigint();
        for (let i = 2; i <= n; i++) {
            recordLineTiming('fibonacci', 4, process.hrtime.bigint() - t);

            // Line 5: const temp = a + b;
            t = process.hrtime.bigint();
            const temp = a + b;
            recordLineTiming('fibonacci', 5, process.hrtime.bigint() - t);

            // Line 6: a = b;
            t = process.hrtime.bigint();
            a = b;
            recordLineTiming('fibonacci', 6, process.hrtime.bigint() - t);

            // Line 7: b = temp;
            t = process.hrtime.bigint();
            b = temp;
            recordLineTiming('fibonacci', 7, process.hrtime.bigint() - t);

            // Loop iteration timing
            t = process.hrtime.bigint();
        }
        recordLineTiming('fibonacci', 4, process.hrtime.bigint() - t);

        // Line 8: return b;
        t = process.hrtime.bigint();
        const result = b;
        recordLineTiming('fibonacci', 8, process.hrtime.bigint() - t);
        return result;
    };
}

/**
 * Manual instrumentation for reverseString
 */
function createManuallyInstrumentedReverseString() {
    return function reverseString_instrumented(str) {
        let t;

        // Line 1: let result = '';
        t = process.hrtime.bigint();
        let result = '';
        recordLineTiming('reverseString', 1, process.hrtime.bigint() - t);

        // Line 2-4: for loop
        t = process.hrtime.bigint();
        for (let i = str.length - 1; i >= 0; i--) {
            recordLineTiming('reverseString', 2, process.hrtime.bigint() - t);

            // Line 3: result += str[i];
            t = process.hrtime.bigint();
            result += str[i];
            recordLineTiming('reverseString', 3, process.hrtime.bigint() - t);

            t = process.hrtime.bigint();
        }
        recordLineTiming('reverseString', 2, process.hrtime.bigint() - t);

        // Line 5: return result;
        t = process.hrtime.bigint();
        const ret = result;
        recordLineTiming('reverseString', 5, process.hrtime.bigint() - t);
        return ret;
    };
}

/**
 * Manual instrumentation for bubbleSort
 */
function createManuallyInstrumentedBubbleSort() {
    return function bubbleSort_instrumented(arr) {
        let t;

        // Line 1: const n = arr.length;
        t = process.hrtime.bigint();
        const n = arr.length;
        recordLineTiming('bubbleSort', 1, process.hrtime.bigint() - t);

        // Line 2: const sorted = [...arr];
        t = process.hrtime.bigint();
        const sorted = [...arr];
        recordLineTiming('bubbleSort', 2, process.hrtime.bigint() - t);

        // Line 3: outer for loop
        t = process.hrtime.bigint();
        for (let i = 0; i < n - 1; i++) {
            recordLineTiming('bubbleSort', 3, process.hrtime.bigint() - t);

            // Line 4: inner for loop
            t = process.hrtime.bigint();
            for (let j = 0; j < n - i - 1; j++) {
                recordLineTiming('bubbleSort', 4, process.hrtime.bigint() - t);

                // Line 5: if (sorted[j] > sorted[j + 1])
                t = process.hrtime.bigint();
                if (sorted[j] > sorted[j + 1]) {
                    recordLineTiming('bubbleSort', 5, process.hrtime.bigint() - t);

                    // Line 6: const temp = sorted[j];
                    t = process.hrtime.bigint();
                    const temp = sorted[j];
                    recordLineTiming('bubbleSort', 6, process.hrtime.bigint() - t);

                    // Line 7: sorted[j] = sorted[j + 1];
                    t = process.hrtime.bigint();
                    sorted[j] = sorted[j + 1];
                    recordLineTiming('bubbleSort', 7, process.hrtime.bigint() - t);

                    // Line 8: sorted[j + 1] = temp;
                    t = process.hrtime.bigint();
                    sorted[j + 1] = temp;
                    recordLineTiming('bubbleSort', 8, process.hrtime.bigint() - t);
                } else {
                    recordLineTiming('bubbleSort', 5, process.hrtime.bigint() - t);
                }

                t = process.hrtime.bigint();
            }
            recordLineTiming('bubbleSort', 4, process.hrtime.bigint() - t);

            t = process.hrtime.bigint();
        }
        recordLineTiming('bubbleSort', 3, process.hrtime.bigint() - t);

        // Line 12: return sorted;
        t = process.hrtime.bigint();
        const ret = sorted;
        recordLineTiming('bubbleSort', 12, process.hrtime.bigint() - t);
        return ret;
    };
}

module.exports = {
    startTimer,
    endTimer,
    recordLineTiming,
    getTimings,
    clearTimings,
    instrumentFunction,
    createProfiledVersion,
    createManuallyInstrumentedFibonacci,
    createManuallyInstrumentedReverseString,
    createManuallyInstrumentedBubbleSort
};
