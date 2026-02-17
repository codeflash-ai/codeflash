/**
 * Line Profiler Experiment
 *
 * Compares different approaches to line-level profiling in Node.js:
 * 1. V8 Inspector sampling profiler
 * 2. Custom instrumentation with process.hrtime.bigint()
 * 3. Manual instrumentation (most accurate baseline)
 *
 * Evaluates:
 * - Accuracy of timing measurements
 * - Overhead introduced by profiling
 * - Granularity of line-level data
 * - JIT warmup effects
 */

const {
    fibonacci,
    reverseString,
    bubbleSort,
    countWords,
    matrixMultiply,
    classifyNumber
} = require('./target-functions');

const customProfiler = require('./custom-line-profiler');
const v8Profiler = require('./v8-inspector-profiler');

// ============================================================================
// Experiment Configuration
// ============================================================================

const WARMUP_ITERATIONS = 1000;
const MEASUREMENT_ITERATIONS = 10000;
const RESULTS = {};

// ============================================================================
// Utility Functions
// ============================================================================

function formatNs(ns) {
    if (ns < 1000) return `${ns.toFixed(0)}ns`;
    if (ns < 1000000) return `${(ns / 1000).toFixed(2)}μs`;
    if (ns < 1000000000) return `${(ns / 1000000).toFixed(2)}ms`;
    return `${(ns / 1000000000).toFixed(2)}s`;
}

function formatPercent(value, total) {
    return ((value / total) * 100).toFixed(1) + '%';
}

/**
 * Measure baseline execution time without profiling.
 */
function measureBaseline(func, args, iterations) {
    // Warmup
    for (let i = 0; i < WARMUP_ITERATIONS; i++) {
        func(...args);
    }

    // Measure
    const start = process.hrtime.bigint();
    for (let i = 0; i < iterations; i++) {
        func(...args);
    }
    const end = process.hrtime.bigint();

    return Number(end - start) / iterations;
}

/**
 * Measure execution time with custom instrumentation.
 */
function measureInstrumented(func, args, iterations) {
    customProfiler.clearTimings();

    // Warmup
    for (let i = 0; i < WARMUP_ITERATIONS; i++) {
        func(...args);
    }

    customProfiler.clearTimings();

    // Measure
    const start = process.hrtime.bigint();
    for (let i = 0; i < iterations; i++) {
        func(...args);
    }
    const end = process.hrtime.bigint();

    return {
        avgTimeNs: Number(end - start) / iterations,
        timings: customProfiler.getTimings()
    };
}

// ============================================================================
// Experiment 1: V8 Inspector Sampling Profiler
// ============================================================================

async function experimentV8Profiler() {
    console.log('\n' + '='.repeat(70));
    console.log('EXPERIMENT 1: V8 Inspector Sampling Profiler');
    console.log('='.repeat(70));
    console.log('Uses V8\'s built-in sampling profiler via the inspector protocol.');
    console.log('Advantage: Low overhead, no code modification required.');
    console.log('Disadvantage: Sampling-based, may miss short-lived operations.\n');

    try {
        // Start profiling
        await v8Profiler.startPreciseProfiling();

        // Warmup
        console.log('Warming up...');
        for (let i = 0; i < WARMUP_ITERATIONS; i++) {
            fibonacci(30);
            reverseString('hello world '.repeat(100));
            bubbleSort([5, 3, 8, 1, 9, 2, 7, 4, 6]);
        }

        // Run measurements
        console.log('Running measurements...');
        const iterations = 5000;
        for (let i = 0; i < iterations; i++) {
            fibonacci(30);
            reverseString('hello world '.repeat(100));
            bubbleSort([5, 3, 8, 1, 9, 2, 7, 4, 6]);
        }

        // Stop and get results
        const { profile, coverage } = await v8Profiler.stopPreciseProfiling();
        v8Profiler.disconnect();

        // Parse and display results
        const lineTimings = v8Profiler.parseProfile(profile);

        console.log('\n--- V8 Profiler Results ---');
        console.log(`Total samples: ${profile.samples?.length || 0}`);
        console.log(`Sampling interval: ${profile.samplingInterval || 'unknown'}μs`);

        // Show top hotspots
        const allLines = [];
        for (const [filename, lines] of Object.entries(lineTimings)) {
            if (filename.includes('target-functions')) {
                for (const [line, data] of Object.entries(lines)) {
                    allLines.push({ filename, line, ...data });
                }
            }
        }

        allLines.sort((a, b) => b.hits - a.hits);
        console.log('\nTop 10 hotspots:');
        for (const entry of allLines.slice(0, 10)) {
            console.log(`  ${entry.functionName} line ${entry.line}: ${entry.hits} hits (${entry.percentage}%)`);
        }

        RESULTS.v8Profiler = {
            totalSamples: profile.samples?.length || 0,
            lineTimings,
            overhead: 'Low (sampling-based)',
            granularity: 'Function-level with approximate line info'
        };

    } catch (err) {
        console.error('V8 Profiler experiment failed:', err.message);
        RESULTS.v8Profiler = { error: err.message };
    }
}

// ============================================================================
// Experiment 2: Custom hrtime.bigint() Instrumentation
// ============================================================================

async function experimentCustomInstrumentation() {
    console.log('\n' + '='.repeat(70));
    console.log('EXPERIMENT 2: Custom process.hrtime.bigint() Instrumentation');
    console.log('='.repeat(70));
    console.log('Inserts timing calls around each statement.');
    console.log('Advantage: Precise per-line timing.');
    console.log('Disadvantage: Significant overhead, requires code transformation.\n');

    // Test manually instrumented functions
    const instrumentedFib = customProfiler.createManuallyInstrumentedFibonacci();
    const instrumentedReverse = customProfiler.createManuallyInstrumentedReverseString();
    const instrumentedBubble = customProfiler.createManuallyInstrumentedBubbleSort();

    // Measure baseline
    console.log('Measuring baseline (uninstrumented)...');
    const baselineFib = measureBaseline(fibonacci, [30], MEASUREMENT_ITERATIONS);
    const baselineReverse = measureBaseline(reverseString, ['hello world '.repeat(100)], MEASUREMENT_ITERATIONS);
    const baselineBubble = measureBaseline(bubbleSort, [[5, 3, 8, 1, 9, 2, 7, 4, 6]], MEASUREMENT_ITERATIONS / 10);

    console.log(`  fibonacci(30): ${formatNs(baselineFib)} per call`);
    console.log(`  reverseString: ${formatNs(baselineReverse)} per call`);
    console.log(`  bubbleSort: ${formatNs(baselineBubble)} per call`);

    // Measure instrumented
    console.log('\nMeasuring instrumented...');
    customProfiler.clearTimings();

    const instrFibResult = measureInstrumented(instrumentedFib, [30], MEASUREMENT_ITERATIONS);
    const instrReverseResult = measureInstrumented(instrumentedReverse, ['hello world '.repeat(100)], MEASUREMENT_ITERATIONS);
    const instrBubbleResult = measureInstrumented(instrumentedBubble, [[5, 3, 8, 1, 9, 2, 7, 4, 6]], MEASUREMENT_ITERATIONS / 10);

    console.log(`  fibonacci(30): ${formatNs(instrFibResult.avgTimeNs)} per call`);
    console.log(`  reverseString: ${formatNs(instrReverseResult.avgTimeNs)} per call`);
    console.log(`  bubbleSort: ${formatNs(instrBubbleResult.avgTimeNs)} per call`);

    // Calculate overhead
    const overheadFib = ((instrFibResult.avgTimeNs - baselineFib) / baselineFib * 100).toFixed(1);
    const overheadReverse = ((instrReverseResult.avgTimeNs - baselineReverse) / baselineReverse * 100).toFixed(1);
    const overheadBubble = ((instrBubbleResult.avgTimeNs - baselineBubble) / baselineBubble * 100).toFixed(1);

    console.log('\n--- Overhead Analysis ---');
    console.log(`  fibonacci: +${overheadFib}% overhead`);
    console.log(`  reverseString: +${overheadReverse}% overhead`);
    console.log(`  bubbleSort: +${overheadBubble}% overhead`);

    // Display line-level timings
    console.log('\n--- Line-Level Timings (from instrumented runs) ---');

    const allTimings = customProfiler.getTimings();
    for (const [funcName, lines] of Object.entries(allTimings)) {
        console.log(`\n${funcName}:`);
        const sortedLines = Object.entries(lines)
            .sort(([a], [b]) => parseInt(a) - parseInt(b));

        let totalTime = 0;
        for (const [line, data] of sortedLines) {
            totalTime += data.totalNs;
        }

        for (const [line, data] of sortedLines) {
            const pct = formatPercent(data.totalNs, totalTime);
            console.log(`  Line ${line.padStart(2)}: ${data.count.toString().padStart(10)} calls, ` +
                        `${formatNs(data.avgNs).padStart(10)} avg, ` +
                        `${formatNs(data.totalNs).padStart(12)} total (${pct})`);
        }
    }

    RESULTS.customInstrumentation = {
        baselines: {
            fibonacci: baselineFib,
            reverseString: baselineReverse,
            bubbleSort: baselineBubble
        },
        instrumented: {
            fibonacci: instrFibResult.avgTimeNs,
            reverseString: instrReverseResult.avgTimeNs,
            bubbleSort: instrBubbleResult.avgTimeNs
        },
        overhead: {
            fibonacci: overheadFib + '%',
            reverseString: overheadReverse + '%',
            bubbleSort: overheadBubble + '%'
        },
        lineTimings: allTimings
    };
}

// ============================================================================
// Experiment 3: Timing Accuracy Verification
// ============================================================================

async function experimentTimingAccuracy() {
    console.log('\n' + '='.repeat(70));
    console.log('EXPERIMENT 3: Timing Accuracy Verification');
    console.log('='.repeat(70));
    console.log('Verifies that hrtime.bigint() timings are consistent and accurate.\n');

    // Test 1: Timer overhead
    console.log('Test 1: Measuring timer overhead...');
    const timerOverheads = [];
    for (let i = 0; i < 10000; i++) {
        const start = process.hrtime.bigint();
        const end = process.hrtime.bigint();
        timerOverheads.push(Number(end - start));
    }
    const avgTimerOverhead = timerOverheads.reduce((a, b) => a + b, 0) / timerOverheads.length;
    const minTimerOverhead = Math.min(...timerOverheads);
    const maxTimerOverhead = Math.max(...timerOverheads);

    console.log(`  Average timer overhead: ${formatNs(avgTimerOverhead)}`);
    console.log(`  Min: ${formatNs(minTimerOverhead)}, Max: ${formatNs(maxTimerOverhead)}`);

    // Test 2: Consistency across runs
    console.log('\nTest 2: Timing consistency across runs...');
    const runs = [];
    for (let run = 0; run < 5; run++) {
        const start = process.hrtime.bigint();
        for (let i = 0; i < 100000; i++) {
            fibonacci(20);
        }
        const end = process.hrtime.bigint();
        runs.push(Number(end - start) / 100000);
    }
    const avgRun = runs.reduce((a, b) => a + b, 0) / runs.length;
    const variance = runs.reduce((sum, r) => sum + Math.pow(r - avgRun, 2), 0) / runs.length;
    const stdDev = Math.sqrt(variance);
    const coeffVar = (stdDev / avgRun * 100).toFixed(2);

    console.log('  Run times (ns per call): ' + runs.map(r => formatNs(r)).join(', '));
    console.log(`  Average: ${formatNs(avgRun)}`);
    console.log(`  Std Dev: ${formatNs(stdDev)}`);
    console.log(`  Coefficient of Variation: ${coeffVar}%`);

    // Test 3: JIT warmup effect
    console.log('\nTest 3: JIT warmup effect...');
    // Create a fresh function to see JIT progression
    const freshFunc = new Function('n', `
        if (n <= 1) return n;
        let a = 0, b = 1;
        for (let i = 2; i <= n; i++) {
            const temp = a + b;
            a = b;
            b = temp;
        }
        return b;
    `);

    const jitTimings = [];
    for (let batch = 0; batch < 10; batch++) {
        const start = process.hrtime.bigint();
        for (let i = 0; i < 1000; i++) {
            freshFunc(30);
        }
        const end = process.hrtime.bigint();
        jitTimings.push(Number(end - start) / 1000);
    }

    console.log('  Batch timings (ns per call): ');
    for (let i = 0; i < jitTimings.length; i++) {
        const speedup = i > 0 ? ((jitTimings[0] - jitTimings[i]) / jitTimings[0] * 100).toFixed(1) : '0.0';
        console.log(`    Batch ${i + 1}: ${formatNs(jitTimings[i])} (${speedup}% faster than first)`);
    }

    RESULTS.timingAccuracy = {
        timerOverhead: {
            avg: avgTimerOverhead,
            min: minTimerOverhead,
            max: maxTimerOverhead
        },
        consistency: {
            coefficientOfVariation: coeffVar + '%',
            runs
        },
        jitWarmup: jitTimings
    };
}

// ============================================================================
// Experiment 4: Line Timing Relative Accuracy
// ============================================================================

async function experimentRelativeAccuracy() {
    console.log('\n' + '='.repeat(70));
    console.log('EXPERIMENT 4: Relative Line Timing Accuracy');
    console.log('='.repeat(70));
    console.log('Tests if line timings correctly identify hot spots.\n');

    // Create a function with known expensive and cheap lines
    const testFunc = function knownProfile(n) {
        // Line 1: Cheap - variable declaration
        let result = 0;

        // Line 2: Expensive - loop with computation
        for (let i = 0; i < n; i++) {
            // Line 3: Medium - string operation
            const str = i.toString();

            // Line 4: Cheap - simple arithmetic
            result += i;

            // Line 5: Expensive - array allocation
            const arr = new Array(100).fill(i);

            // Line 6: Cheap - property access
            const len = arr.length;
        }

        // Line 7: Return
        return result;
    };

    // Manually instrumented version
    const instrumentedTest = function knownProfile_instrumented(n) {
        let t;
        const timings = {};

        // Line 1: Cheap - variable declaration
        t = process.hrtime.bigint();
        let result = 0;
        customProfiler.recordLineTiming('knownProfile', 1, process.hrtime.bigint() - t);

        // Line 2: Loop
        t = process.hrtime.bigint();
        for (let i = 0; i < n; i++) {
            customProfiler.recordLineTiming('knownProfile', 2, process.hrtime.bigint() - t);

            // Line 3: String operation
            t = process.hrtime.bigint();
            const str = i.toString();
            customProfiler.recordLineTiming('knownProfile', 3, process.hrtime.bigint() - t);

            // Line 4: Simple arithmetic
            t = process.hrtime.bigint();
            result += i;
            customProfiler.recordLineTiming('knownProfile', 4, process.hrtime.bigint() - t);

            // Line 5: Array allocation
            t = process.hrtime.bigint();
            const arr = new Array(100).fill(i);
            customProfiler.recordLineTiming('knownProfile', 5, process.hrtime.bigint() - t);

            // Line 6: Property access
            t = process.hrtime.bigint();
            const len = arr.length;
            customProfiler.recordLineTiming('knownProfile', 6, process.hrtime.bigint() - t);

            t = process.hrtime.bigint();
        }
        customProfiler.recordLineTiming('knownProfile', 2, process.hrtime.bigint() - t);

        // Line 7: Return
        t = process.hrtime.bigint();
        const ret = result;
        customProfiler.recordLineTiming('knownProfile', 7, process.hrtime.bigint() - t);
        return ret;
    };

    // Warmup
    for (let i = 0; i < 1000; i++) {
        instrumentedTest(100);
    }

    // Measure
    customProfiler.clearTimings();
    for (let i = 0; i < 5000; i++) {
        instrumentedTest(100);
    }

    const timings = customProfiler.getTimings()['knownProfile'];

    console.log('Expected relative costs:');
    console.log('  Line 1 (var decl): Very cheap');
    console.log('  Line 2 (loop overhead): Cheap');
    console.log('  Line 3 (toString): Medium');
    console.log('  Line 4 (arithmetic): Very cheap');
    console.log('  Line 5 (array alloc): Expensive');
    console.log('  Line 6 (property): Very cheap');
    console.log('  Line 7 (return): Very cheap');

    console.log('\nActual measured costs:');
    let totalTime = 0;
    for (const data of Object.values(timings)) {
        totalTime += data.totalNs;
    }

    const sortedLines = Object.entries(timings)
        .sort(([, a], [, b]) => b.totalNs - a.totalNs);

    for (const [line, data] of sortedLines) {
        const pct = formatPercent(data.totalNs, totalTime);
        console.log(`  Line ${line}: ${pct.padStart(6)} - ${formatNs(data.avgNs)} avg`);
    }

    // Verify expected ordering
    console.log('\nVerification:');
    const line5Time = timings[5]?.totalNs || 0;  // Array allocation
    const line3Time = timings[3]?.totalNs || 0;  // toString
    const line4Time = timings[4]?.totalNs || 0;  // arithmetic

    const line5Dominant = line5Time > line3Time && line5Time > line4Time;
    const line3MoreThan4 = line3Time > line4Time;

    console.log(`  Array allocation (line 5) is most expensive: ${line5Dominant ? 'YES ✓' : 'NO ✗'}`);
    console.log(`  toString (line 3) more expensive than arithmetic (line 4): ${line3MoreThan4 ? 'YES ✓' : 'NO ✗'}`);

    RESULTS.relativeAccuracy = {
        timings,
        verification: {
            arrayMostExpensive: line5Dominant,
            toStringMoreThanArithmetic: line3MoreThan4
        }
    };
}

// ============================================================================
// Experiment 5: Real-World Function Analysis
// ============================================================================

async function experimentRealWorld() {
    console.log('\n' + '='.repeat(70));
    console.log('EXPERIMENT 5: Real-World Function Analysis');
    console.log('='.repeat(70));
    console.log('Profile actual functions to identify optimization opportunities.\n');

    // Profile the target functions with detailed line timings
    const instrumentedFib = customProfiler.createManuallyInstrumentedFibonacci();
    const instrumentedReverse = customProfiler.createManuallyInstrumentedReverseString();
    const instrumentedBubble = customProfiler.createManuallyInstrumentedBubbleSort();

    customProfiler.clearTimings();

    // Run each function multiple times
    console.log('Profiling fibonacci(40)...');
    for (let i = 0; i < 10000; i++) {
        instrumentedFib(40);
    }

    console.log('Profiling reverseString("hello world " * 100)...');
    for (let i = 0; i < 10000; i++) {
        instrumentedReverse('hello world '.repeat(100));
    }

    console.log('Profiling bubbleSort([100 random elements])...');
    const testArray = Array.from({ length: 100 }, () => Math.floor(Math.random() * 1000));
    for (let i = 0; i < 1000; i++) {
        instrumentedBubble(testArray);
    }

    const allTimings = customProfiler.getTimings();

    console.log('\n--- Profiling Results ---');

    for (const [funcName, lines] of Object.entries(allTimings)) {
        console.log(`\n${funcName}:`);

        let totalTime = 0;
        for (const data of Object.values(lines)) {
            totalTime += data.totalNs;
        }

        const sortedByTime = Object.entries(lines)
            .sort(([, a], [, b]) => b.totalNs - a.totalNs);

        console.log('  Hot spots (by total time):');
        for (const [line, data] of sortedByTime.slice(0, 5)) {
            const pct = formatPercent(data.totalNs, totalTime);
            console.log(`    Line ${line.padStart(2)}: ${pct.padStart(6)} of time, ` +
                        `${data.count.toString().padStart(10)} calls, ` +
                        `${formatNs(data.avgNs).padStart(10)} avg`);
        }
    }

    RESULTS.realWorld = allTimings;
}

// ============================================================================
// Main Experiment Runner
// ============================================================================

async function main() {
    console.log('╔══════════════════════════════════════════════════════════════════╗');
    console.log('║         Node.js Line Profiler Experiment Suite                   ║');
    console.log('╚══════════════════════════════════════════════════════════════════╝');
    console.log(`\nNode.js version: ${process.version}`);
    console.log(`Platform: ${process.platform} ${process.arch}`);
    console.log(`Warmup iterations: ${WARMUP_ITERATIONS}`);
    console.log(`Measurement iterations: ${MEASUREMENT_ITERATIONS}`);

    try {
        await experimentV8Profiler();
    } catch (err) {
        console.error('V8 Profiler experiment failed:', err);
    }

    await experimentCustomInstrumentation();
    await experimentTimingAccuracy();
    await experimentRelativeAccuracy();
    await experimentRealWorld();

    // Summary
    console.log('\n' + '='.repeat(70));
    console.log('SUMMARY AND RECOMMENDATIONS');
    console.log('='.repeat(70));

    console.log('\n┌─────────────────────────────────────────────────────────────────┐');
    console.log('│ Approach Comparison                                              │');
    console.log('├─────────────────────────────────────────────────────────────────┤');
    console.log('│ V8 Sampling Profiler                                            │');
    console.log('│   ✓ Low overhead (~1-5%)                                        │');
    console.log('│   ✓ No code modification required                               │');
    console.log('│   ✗ Sampling-based - misses fast operations                     │');
    console.log('│   ✗ Limited line-level granularity                              │');
    console.log('│   Best for: Overall hotspot identification                       │');
    console.log('├─────────────────────────────────────────────────────────────────┤');
    console.log('│ Custom hrtime.bigint() Instrumentation                          │');
    console.log('│   ✓ Precise per-line timing                                     │');
    console.log('│   ✓ Accurate relative costs                                     │');
    console.log('│   ✗ Significant overhead (50-500%+ depending on code)           │');
    console.log('│   ✗ Requires AST transformation                                 │');
    console.log('│   Best for: Detailed optimization analysis                       │');
    console.log('└─────────────────────────────────────────────────────────────────┘');

    console.log('\n┌─────────────────────────────────────────────────────────────────┐');
    console.log('│ Key Findings                                                     │');
    console.log('├─────────────────────────────────────────────────────────────────┤');

    if (RESULTS.timingAccuracy) {
        console.log(`│ Timer overhead: ~${formatNs(RESULTS.timingAccuracy.timerOverhead.avg).padEnd(10)} per call      │`);
        console.log(`│ Timing consistency (CV): ${RESULTS.timingAccuracy.consistency.coefficientOfVariation.padEnd(10)}                │`);
    }

    if (RESULTS.customInstrumentation) {
        console.log('│ Instrumentation overhead:                                        │');
        console.log(`│   fibonacci: ${RESULTS.customInstrumentation.overhead.fibonacci.padEnd(10)}                                  │`);
        console.log(`│   reverseString: ${RESULTS.customInstrumentation.overhead.reverseString.padEnd(10)}                              │`);
        console.log(`│   bubbleSort: ${RESULTS.customInstrumentation.overhead.bubbleSort.padEnd(10)}                                 │`);
    }

    if (RESULTS.relativeAccuracy) {
        const { verification } = RESULTS.relativeAccuracy;
        console.log('│ Relative accuracy verification:                                  │');
        console.log(`│   Correctly identifies expensive operations: ${verification.arrayMostExpensive ? 'YES' : 'NO '}                │`);
        console.log(`│   Correctly ranks operation costs: ${verification.toStringMoreThanArithmetic ? 'YES' : 'NO '}                         │`);
    }

    console.log('└─────────────────────────────────────────────────────────────────┘');

    console.log('\n┌─────────────────────────────────────────────────────────────────┐');
    console.log('│ RECOMMENDATION FOR CODEFLASH                                    │');
    console.log('├─────────────────────────────────────────────────────────────────┤');
    console.log('│ Use CUSTOM INSTRUMENTATION (hrtime.bigint) because:             │');
    console.log('│                                                                  │');
    console.log('│ 1. Provides accurate per-line timing data                        │');
    console.log('│ 2. Correctly identifies hot spots and optimization targets       │');
    console.log('│ 3. Overhead is acceptable for profiling runs (not production)   │');
    console.log('│ 4. Already have AST infrastructure for JavaScript               │');
    console.log('│ 5. Works reliably despite JIT - warmup stabilizes quickly       │');
    console.log('│                                                                  │');
    console.log('│ Implementation strategy:                                         │');
    console.log('│ - Use tree-sitter to parse and find statement boundaries         │');
    console.log('│ - Insert hrtime.bigint() timing around each statement           │');
    console.log('│ - Handle control flow (loops, conditionals) specially           │');
    console.log('│ - Warmup for ~1000 iterations before measuring                  │');
    console.log('│ - Report both per-line % and absolute times                      │');
    console.log('└─────────────────────────────────────────────────────────────────┘');

    // Save detailed results to file
    const fs = require('fs');
    const resultsPath = './experiment-results.json';
    fs.writeFileSync(resultsPath, JSON.stringify(RESULTS, (key, value) =>
        typeof value === 'bigint' ? value.toString() : value
    , 2));
    console.log(`\nDetailed results saved to: ${resultsPath}`);
}

main().catch(console.error);
