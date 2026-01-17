#!/usr/bin/env node
/**
 * Integration Test: Behavior Testing with Different Optimization Indices
 *
 * This script simulates the actual codeflash workflow:
 * 1. Run tests with CODEFLASH_LOOP_INDEX=1 (original code)
 * 2. Run tests with CODEFLASH_LOOP_INDEX=2 (optimized code)
 * 3. Read back both result files
 * 4. Compare using the comparator to verify equivalence
 *
 * Run directly: node tests/integration-behavior-test.js
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Import our modules
const { serialize, deserialize, getSerializerType } = require('../codeflash-serializer');
const { comparator } = require('../codeflash-comparator');

// Test configuration
const TEST_DIR = '/tmp/codeflash_integration_test';
const ORIGINAL_RESULTS = path.join(TEST_DIR, 'original_results.bin');
const OPTIMIZED_RESULTS = path.join(TEST_DIR, 'optimized_results.bin');

// Sample function to test - this simulates the "function being optimized"
function processData(input) {
    // Original implementation
    const result = {
        numbers: input.numbers.map(n => n * 2),
        sum: input.numbers.reduce((a, b) => a + b, 0),
        metadata: new Map([
            ['processed', true],
            ['timestamp', new Date()],
        ]),
        tags: new Set(input.tags || []),
    };
    return result;
}

// "Optimized" version - same behavior, different implementation
function processDataOptimized(input) {
    // Optimized implementation (same behavior)
    const doubled = [];
    let sum = 0;
    for (const n of input.numbers) {
        doubled.push(n * 2);
        sum += n;
    }
    return {
        numbers: doubled,
        sum,
        metadata: new Map([
            ['processed', true],
            ['timestamp', new Date()],
        ]),
        tags: new Set(input.tags || []),
    };
}

// Test cases
const testCases = [
    { numbers: [1, 2, 3], tags: ['a', 'b'] },
    { numbers: [10, 20, 30, 40] },
    { numbers: [-5, 0, 5], tags: ['negative', 'zero', 'positive'] },
    { numbers: [1.5, 2.5, 3.5] },
    { numbers: [] },
];

// Helper to run a function and capture behavior
function captureAllBehaviors(fn, inputs) {
    const results = [];
    for (const input of inputs) {
        try {
            const returnValue = fn(input);
            // Remove timestamp from metadata for comparison (it will differ)
            if (returnValue.metadata) {
                returnValue.metadata.delete('timestamp');
            }
            results.push({
                success: true,
                args: [input],
                kwargs: {},
                returnValue,
            });
        } catch (error) {
            results.push({
                success: false,
                args: [input],
                kwargs: {},
                error: { name: error.name, message: error.message },
            });
        }
    }
    return results;
}

// Main test function
async function runIntegrationTest() {
    console.log('='.repeat(60));
    console.log('Integration Test: Behavior Comparison');
    console.log('='.repeat(60));
    console.log(`Serializer type: ${getSerializerType()}`);
    console.log();

    // Setup
    if (fs.existsSync(TEST_DIR)) {
        fs.rmSync(TEST_DIR, { recursive: true });
    }
    fs.mkdirSync(TEST_DIR, { recursive: true });

    // Phase 1: Run "original" code (LOOP_INDEX=1)
    console.log('Phase 1: Capturing original behavior...');
    const originalBehaviors = captureAllBehaviors(processData, testCases);
    const originalSerialized = serialize(originalBehaviors);
    fs.writeFileSync(ORIGINAL_RESULTS, originalSerialized);
    console.log(`  - Captured ${originalBehaviors.length} invocations`);
    console.log(`  - Serialized size: ${originalSerialized.length} bytes`);
    console.log(`  - Saved to: ${ORIGINAL_RESULTS}`);
    console.log();

    // Phase 2: Run "optimized" code (LOOP_INDEX=2)
    console.log('Phase 2: Capturing optimized behavior...');
    const optimizedBehaviors = captureAllBehaviors(processDataOptimized, testCases);
    const optimizedSerialized = serialize(optimizedBehaviors);
    fs.writeFileSync(OPTIMIZED_RESULTS, optimizedSerialized);
    console.log(`  - Captured ${optimizedBehaviors.length} invocations`);
    console.log(`  - Serialized size: ${optimizedSerialized.length} bytes`);
    console.log(`  - Saved to: ${OPTIMIZED_RESULTS}`);
    console.log();

    // Phase 3: Read back and compare
    console.log('Phase 3: Comparing behaviors...');
    const originalRestored = deserialize(fs.readFileSync(ORIGINAL_RESULTS));
    const optimizedRestored = deserialize(fs.readFileSync(OPTIMIZED_RESULTS));

    console.log(`  - Original results restored: ${originalRestored.length} invocations`);
    console.log(`  - Optimized results restored: ${optimizedRestored.length} invocations`);
    console.log();

    // Compare each invocation
    let allEqual = true;
    const comparisonResults = [];

    for (let i = 0; i < originalRestored.length; i++) {
        const orig = originalRestored[i];
        const opt = optimizedRestored[i];

        // Compare the behavior tuples
        const isEqual = comparator(
            [orig.args, orig.kwargs, orig.returnValue],
            [opt.args, opt.kwargs, opt.returnValue]
        );

        comparisonResults.push({
            invocation: i,
            isEqual,
            args: orig.args,
        });

        if (!isEqual) {
            allEqual = false;
            console.log(`  ❌ Invocation ${i}: DIFFERENT`);
            console.log(`     Args: ${JSON.stringify(orig.args)}`);
        } else {
            console.log(`  ✓ Invocation ${i}: EQUAL`);
        }
    }

    console.log();
    console.log('='.repeat(60));
    if (allEqual) {
        console.log('✅ SUCCESS: All behaviors are equivalent!');
        console.log('   The optimization preserves correctness.');
    } else {
        console.log('❌ FAILURE: Some behaviors differ!');
        console.log('   The optimization changed the behavior.');
    }
    console.log('='.repeat(60));

    // Cleanup
    fs.rmSync(TEST_DIR, { recursive: true });

    // Return result for programmatic use
    return { success: allEqual, results: comparisonResults };
}

// Also test with a "broken" optimization
async function runBrokenOptimizationTest() {
    console.log();
    console.log('='.repeat(60));
    console.log('Testing detection of broken optimization...');
    console.log('='.repeat(60));

    // Setup
    if (!fs.existsSync(TEST_DIR)) {
        fs.mkdirSync(TEST_DIR, { recursive: true });
    }

    // Original function
    const original = (x) => x * 2;

    // "Broken" optimized function
    const brokenOptimized = (x) => x * 2 + 1;  // Bug: adds 1

    const inputs = [1, 5, 10, 100];

    // Capture original
    const originalResults = inputs.map(x => ({
        args: [x],
        kwargs: {},
        returnValue: original(x),
    }));

    // Capture broken optimized
    const brokenResults = inputs.map(x => ({
        args: [x],
        kwargs: {},
        returnValue: brokenOptimized(x),
    }));

    // Serialize
    const originalSerialized = serialize(originalResults);
    const brokenSerialized = serialize(brokenResults);

    // Compare
    const originalRestored = deserialize(originalSerialized);
    const brokenRestored = deserialize(brokenSerialized);

    let detectedBug = false;
    for (let i = 0; i < originalRestored.length; i++) {
        const isEqual = comparator(
            [originalRestored[i].args, {}, originalRestored[i].returnValue],
            [brokenRestored[i].args, {}, brokenRestored[i].returnValue]
        );
        if (!isEqual) {
            detectedBug = true;
            console.log(`  ❌ Invocation ${i}: Difference detected`);
            console.log(`     Input: ${originalRestored[i].args[0]}`);
            console.log(`     Original: ${originalRestored[i].returnValue}`);
            console.log(`     Broken: ${brokenRestored[i].returnValue}`);
        }
    }

    console.log();
    if (detectedBug) {
        console.log('✅ SUCCESS: Bug in optimization was detected!');
    } else {
        console.log('❌ FAILURE: Bug was not detected!');
    }
    console.log('='.repeat(60));

    // Cleanup
    if (fs.existsSync(TEST_DIR)) {
        fs.rmSync(TEST_DIR, { recursive: true });
    }

    return { success: detectedBug };
}

// Run tests
async function main() {
    try {
        const result1 = await runIntegrationTest();
        const result2 = await runBrokenOptimizationTest();

        console.log();
        console.log('='.repeat(60));
        console.log('FINAL SUMMARY');
        console.log('='.repeat(60));
        console.log(`Correct optimization test: ${result1.success ? 'PASS' : 'FAIL'}`);
        console.log(`Broken optimization detection: ${result2.success ? 'PASS' : 'FAIL'}`);

        process.exit(result1.success && result2.success ? 0 : 1);
    } catch (error) {
        console.error('Test failed with error:', error);
        process.exit(1);
    }
}

main();
