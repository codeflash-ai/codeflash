#!/usr/bin/env node
/**
 * Codeflash Jest Loop Runner
 *
 * This script runs Jest tests multiple times to collect stable performance measurements.
 * It mimics the Python pytest_plugin.py looping behavior.
 *
 * Usage:
 *   node loop-runner.js <test-file> [options]
 *
 * Options:
 *   --min-loops=N       Minimum loops to run (default: 5)
 *   --max-loops=N       Maximum loops to run (default: 100000)
 *   --duration=N        Target duration in seconds (default: 10)
 *   --stability-check   Enable stability-based early stopping
 */

const { spawn } = require('child_process');
const path = require('path');

// Configuration
const DEFAULT_MIN_LOOPS = 5;
const DEFAULT_MAX_LOOPS = 100000;
const DEFAULT_DURATION_SECONDS = 10;
const STABILITY_WINDOW_SIZE = 0.35;
const STABILITY_CENTER_TOLERANCE = 0.0025;
const STABILITY_SPREAD_TOLERANCE = 0.0025;

/**
 * Parse timing data from Jest stdout.
 * Looks for patterns like: !######test:func:1:lineId_0:123456######!
 * where 123456 is the duration in nanoseconds.
 */
function parseTimingFromStdout(stdout) {
    const timings = new Map(); // Map<testId, number[]>
    const pattern = /!######([^:]+):([^:]*):([^:]+):([^:]+):(\d+_\d+):(\d+)######!/g;

    let match;
    while ((match = pattern.exec(stdout)) !== null) {
        const [, testModule, testClass, testFunc, funcName, invocationId, durationNs] = match;
        const testId = `${testModule}:${testClass}:${testFunc}:${funcName}:${invocationId}`;

        if (!timings.has(testId)) {
            timings.set(testId, []);
        }
        timings.get(testId).push(parseInt(durationNs, 10));
    }

    return timings;
}

/**
 * Run Jest once and return timing data.
 */
async function runJestOnce(testFile, loopIndex, timeout, cwd) {
    return new Promise((resolve, reject) => {
        const env = {
            ...process.env,
            CODEFLASH_LOOP_INDEX: String(loopIndex),
        };

        const jestArgs = [
            'jest',
            testFile,
            '--runInBand',
            '--forceExit',
            `--testTimeout=${timeout * 1000}`,
        ];

        const proc = spawn('npx', jestArgs, {
            cwd,
            env,
            stdio: ['pipe', 'pipe', 'pipe'],
        });

        let stdout = '';
        let stderr = '';

        proc.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        proc.stderr.on('data', (data) => {
            stderr += data.toString();
        });

        proc.on('close', (code) => {
            resolve({
                code,
                stdout,
                stderr,
                timings: parseTimingFromStdout(stdout),
            });
        });

        proc.on('error', reject);
    });
}

/**
 * Check if performance has stabilized.
 * Implements the same stability check as Python's pytest_plugin.
 */
function shouldStopForStability(allTimings, windowSize) {
    // Get total runtime for each loop
    const loopTotals = [];
    for (const [loopIndex, timings] of allTimings.entries()) {
        let total = 0;
        for (const durations of timings.values()) {
            total += Math.min(...durations);
        }
        loopTotals.push(total);
    }

    if (loopTotals.length < windowSize) {
        return false;
    }

    // Get recent window
    const window = loopTotals.slice(-windowSize);

    // Check center tolerance (all values within ±0.25% of median)
    const sorted = [...window].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const centerTolerance = median * STABILITY_CENTER_TOLERANCE;

    const withinCenter = window.every(v => Math.abs(v - median) <= centerTolerance);

    // Check spread tolerance (max-min ≤ 0.25% of min)
    const minVal = Math.min(...window);
    const maxVal = Math.max(...window);
    const spreadTolerance = minVal * STABILITY_SPREAD_TOLERANCE;
    const withinSpread = (maxVal - minVal) <= spreadTolerance;

    return withinCenter && withinSpread;
}

/**
 * Main loop runner.
 */
async function runLoopedTests(testFile, options = {}) {
    const minLoops = options.minLoops || DEFAULT_MIN_LOOPS;
    const maxLoops = options.maxLoops || DEFAULT_MAX_LOOPS;
    const durationSeconds = options.durationSeconds || DEFAULT_DURATION_SECONDS;
    const stabilityCheck = options.stabilityCheck !== false;
    const timeout = options.timeout || 15;
    const cwd = options.cwd || process.cwd();

    console.log(`[codeflash-loop-runner] Starting looped test execution`);
    console.log(`  Test file: ${testFile}`);
    console.log(`  Min loops: ${minLoops}`);
    console.log(`  Max loops: ${maxLoops}`);
    console.log(`  Duration: ${durationSeconds}s`);
    console.log(`  Stability check: ${stabilityCheck}`);
    console.log('');

    const startTime = Date.now();
    const allTimings = new Map(); // Map<loopIndex, Map<testId, number[]>>
    let loopCount = 0;
    let lastExitCode = 0;

    while (true) {
        loopCount++;
        const loopStart = Date.now();

        console.log(`[loop ${loopCount}] Running...`);

        const result = await runJestOnce(testFile, loopCount, timeout, cwd);
        lastExitCode = result.code;

        // Store timings for this loop
        allTimings.set(loopCount, result.timings);

        const loopDuration = Date.now() - loopStart;
        const totalElapsed = (Date.now() - startTime) / 1000;

        // Count timing entries
        let timingCount = 0;
        for (const durations of result.timings.values()) {
            timingCount += durations.length;
        }

        console.log(`[loop ${loopCount}] Completed in ${loopDuration}ms, ${timingCount} timing entries`);

        // Check stopping conditions
        if (loopCount >= maxLoops) {
            console.log(`[codeflash-loop-runner] Reached max loops (${maxLoops})`);
            break;
        }

        if (loopCount >= minLoops && totalElapsed >= durationSeconds) {
            console.log(`[codeflash-loop-runner] Reached duration limit (${durationSeconds}s)`);
            break;
        }

        // Stability check
        if (stabilityCheck && loopCount >= minLoops) {
            const estimatedTotalLoops = Math.floor((durationSeconds / totalElapsed) * loopCount);
            const windowSize = Math.max(3, Math.floor(STABILITY_WINDOW_SIZE * estimatedTotalLoops));

            if (shouldStopForStability(allTimings, windowSize)) {
                console.log(`[codeflash-loop-runner] Performance stabilized after ${loopCount} loops`);
                break;
            }
        }
    }

    // Aggregate results
    const aggregatedTimings = new Map(); // Map<testId, {min, max, avg, count}>

    for (const [loopIndex, timings] of allTimings.entries()) {
        for (const [testId, durations] of timings.entries()) {
            if (!aggregatedTimings.has(testId)) {
                aggregatedTimings.set(testId, { values: [], min: Infinity, max: 0, sum: 0, count: 0 });
            }
            const agg = aggregatedTimings.get(testId);
            for (const d of durations) {
                agg.values.push(d);
                agg.min = Math.min(agg.min, d);
                agg.max = Math.max(agg.max, d);
                agg.sum += d;
                agg.count++;
            }
        }
    }

    // Print summary
    console.log('');
    console.log('=== Performance Summary ===');
    console.log(`Total loops: ${loopCount}`);
    console.log(`Total time: ${((Date.now() - startTime) / 1000).toFixed(2)}s`);
    console.log('');

    for (const [testId, agg] of aggregatedTimings.entries()) {
        const avg = agg.sum / agg.count;
        console.log(`${testId}:`);
        console.log(`  Min: ${(agg.min / 1000).toFixed(2)} μs`);
        console.log(`  Max: ${(agg.max / 1000).toFixed(2)} μs`);
        console.log(`  Avg: ${(avg / 1000).toFixed(2)} μs`);
        console.log(`  Samples: ${agg.count}`);
    }

    return {
        loopCount,
        allTimings,
        aggregatedTimings,
        exitCode: lastExitCode,
    };
}

// CLI interface
if (require.main === module) {
    const args = process.argv.slice(2);

    if (args.length === 0 || args[0] === '--help') {
        console.log('Usage: node loop-runner.js <test-file> [options]');
        console.log('');
        console.log('Options:');
        console.log('  --min-loops=N       Minimum loops to run (default: 5)');
        console.log('  --max-loops=N       Maximum loops to run (default: 100000)');
        console.log('  --duration=N        Target duration in seconds (default: 10)');
        console.log('  --stability-check   Enable stability-based early stopping');
        console.log('  --cwd=PATH          Working directory for Jest');
        process.exit(0);
    }

    const testFile = args[0];
    const options = {};

    for (const arg of args.slice(1)) {
        if (arg.startsWith('--min-loops=')) {
            options.minLoops = parseInt(arg.split('=')[1], 10);
        } else if (arg.startsWith('--max-loops=')) {
            options.maxLoops = parseInt(arg.split('=')[1], 10);
        } else if (arg.startsWith('--duration=')) {
            options.durationSeconds = parseFloat(arg.split('=')[1]);
        } else if (arg === '--stability-check') {
            options.stabilityCheck = true;
        } else if (arg.startsWith('--cwd=')) {
            options.cwd = arg.split('=')[1];
        }
    }

    runLoopedTests(testFile, options)
        .then((result) => {
            process.exit(result.exitCode);
        })
        .catch((error) => {
            console.error('Error:', error);
            process.exit(1);
        });
}

module.exports = { runLoopedTests, parseTimingFromStdout };
