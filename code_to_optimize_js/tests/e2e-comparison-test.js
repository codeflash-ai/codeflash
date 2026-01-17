#!/usr/bin/env node
/**
 * End-to-End Comparison Test
 *
 * This test validates the full behavior comparison workflow:
 * 1. Serialize test results to SQLite (simulating codeflash-jest-helper)
 * 2. Run the comparison script
 * 3. Verify results match expectations
 */

const fs = require('fs');
const path = require('path');

// Import our modules
const { serialize } = require('../codeflash-serializer');
const { readTestResults, compareResults } = require('../codeflash-compare-results');

// Try to load better-sqlite3
let Database;
try {
    Database = require('better-sqlite3');
} catch (e) {
    console.error('better-sqlite3 not installed, skipping E2E test');
    process.exit(0);
}

const TEST_DIR = '/tmp/codeflash_e2e_comparison_test';

/**
 * Create a SQLite database with test results.
 */
function createTestDatabase(dbPath, results) {
    // Ensure directory exists
    const dir = path.dirname(dbPath);
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }

    // Remove existing file
    if (fs.existsSync(dbPath)) {
        fs.unlinkSync(dbPath);
    }

    const db = new Database(dbPath);

    // Create table
    db.exec(`
        CREATE TABLE test_results (
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

    // Insert results
    const stmt = db.prepare(`
        INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    for (const result of results) {
        stmt.run(
            result.testModulePath,
            result.testClassName || null,
            result.testFunctionName,
            result.functionGettingTested,
            result.loopIndex,
            result.iterationId,
            result.runtime,
            result.returnValue ? serialize(result.returnValue) : null,
            result.verificationType || 'function_call'
        );
    }

    db.close();
    return dbPath;
}

/**
 * Test 1: Identical results should be equivalent.
 */
function testIdenticalResults() {
    console.log('\n=== Test 1: Identical Results ===');

    const results = [
        {
            testModulePath: 'tests/math.test.js',
            testFunctionName: 'test adds numbers',
            functionGettingTested: 'add',
            loopIndex: 1,
            iterationId: '0_0',
            runtime: 1000,
            returnValue: [[1, 2], {}, 3],  // [args, kwargs, returnValue]
        },
        {
            testModulePath: 'tests/math.test.js',
            testFunctionName: 'test multiplies numbers',
            functionGettingTested: 'multiply',
            loopIndex: 1,
            iterationId: '0_1',
            runtime: 1000,
            returnValue: [[2, 3], {}, 6],
        },
    ];

    const originalDb = createTestDatabase(path.join(TEST_DIR, 'original1.sqlite'), results);
    const candidateDb = createTestDatabase(path.join(TEST_DIR, 'candidate1.sqlite'), results);

    const originalResults = readTestResults(originalDb);
    const candidateResults = readTestResults(candidateDb);
    const comparison = compareResults(originalResults, candidateResults);

    console.log(`  Original invocations: ${originalResults.size}`);
    console.log(`  Candidate invocations: ${candidateResults.size}`);
    console.log(`  Equivalent: ${comparison.equivalent}`);
    console.log(`  Diffs: ${comparison.diffs.length}`);

    if (!comparison.equivalent || comparison.diffs.length > 0) {
        console.log('  ❌ FAILED: Expected identical results to be equivalent');
        return false;
    }
    console.log('  ✅ PASSED');
    return true;
}

/**
 * Test 2: Different return values should NOT be equivalent.
 */
function testDifferentReturnValues() {
    console.log('\n=== Test 2: Different Return Values ===');

    const originalResults = [
        {
            testModulePath: 'tests/math.test.js',
            testFunctionName: 'test adds numbers',
            functionGettingTested: 'add',
            loopIndex: 1,
            iterationId: '0_0',
            runtime: 1000,
            returnValue: [[1, 2], {}, 3],  // Correct: 1 + 2 = 3
        },
    ];

    const candidateResults = [
        {
            testModulePath: 'tests/math.test.js',
            testFunctionName: 'test adds numbers',
            functionGettingTested: 'add',
            loopIndex: 1,
            iterationId: '0_0',
            runtime: 1000,
            returnValue: [[1, 2], {}, 4],  // Wrong: should be 3, not 4
        },
    ];

    const originalDb = createTestDatabase(path.join(TEST_DIR, 'original2.sqlite'), originalResults);
    const candidateDb = createTestDatabase(path.join(TEST_DIR, 'candidate2.sqlite'), candidateResults);

    const original = readTestResults(originalDb);
    const candidate = readTestResults(candidateDb);
    const comparison = compareResults(original, candidate);

    console.log(`  Equivalent: ${comparison.equivalent}`);
    console.log(`  Diffs: ${comparison.diffs.length}`);

    if (comparison.equivalent || comparison.diffs.length === 0) {
        console.log('  ❌ FAILED: Expected different results to NOT be equivalent');
        return false;
    }
    console.log(`  Diff found: ${comparison.diffs[0].scope}`);
    console.log('  ✅ PASSED');
    return true;
}

/**
 * Test 3: Complex JavaScript types (Map, Set, Date) should compare correctly.
 */
function testComplexTypes() {
    console.log('\n=== Test 3: Complex JavaScript Types ===');

    const complexValue = {
        map: new Map([['a', 1], ['b', 2]]),
        set: new Set([1, 2, 3]),
        date: new Date('2024-01-15T00:00:00.000Z'),
        nested: {
            array: [1, 2, 3],
            map: new Map([['nested', true]]),
        },
    };

    const results = [
        {
            testModulePath: 'tests/complex.test.js',
            testFunctionName: 'test complex return',
            functionGettingTested: 'processData',
            loopIndex: 1,
            iterationId: '0_0',
            runtime: 1000,
            returnValue: [[], {}, complexValue],
        },
    ];

    const originalDb = createTestDatabase(path.join(TEST_DIR, 'original3.sqlite'), results);
    const candidateDb = createTestDatabase(path.join(TEST_DIR, 'candidate3.sqlite'), results);

    const original = readTestResults(originalDb);
    const candidate = readTestResults(candidateDb);
    const comparison = compareResults(original, candidate);

    console.log(`  Original invocations: ${original.size}`);
    console.log(`  Equivalent: ${comparison.equivalent}`);
    console.log(`  Diffs: ${comparison.diffs.length}`);

    if (!comparison.equivalent) {
        console.log('  ❌ FAILED: Expected complex types to be equivalent');
        if (comparison.diffs.length > 0) {
            console.log(`  Diff: ${JSON.stringify(comparison.diffs[0])}`);
        }
        return false;
    }
    console.log('  ✅ PASSED');
    return true;
}

/**
 * Test 4: Floating point tolerance should allow small differences.
 */
function testFloatingPointTolerance() {
    console.log('\n=== Test 4: Floating Point Tolerance ===');

    const originalResults = [
        {
            testModulePath: 'tests/float.test.js',
            testFunctionName: 'test float calculation',
            functionGettingTested: 'calculate',
            loopIndex: 1,
            iterationId: '0_0',
            runtime: 1000,
            returnValue: [[], {}, 0.1 + 0.2],  // 0.30000000000000004
        },
    ];

    const candidateResults = [
        {
            testModulePath: 'tests/float.test.js',
            testFunctionName: 'test float calculation',
            functionGettingTested: 'calculate',
            loopIndex: 1,
            iterationId: '0_0',
            runtime: 1000,
            returnValue: [[], {}, 0.3],  // 0.3 (optimized calculation)
        },
    ];

    const originalDb = createTestDatabase(path.join(TEST_DIR, 'original4.sqlite'), originalResults);
    const candidateDb = createTestDatabase(path.join(TEST_DIR, 'candidate4.sqlite'), candidateResults);

    const original = readTestResults(originalDb);
    const candidate = readTestResults(candidateDb);
    const comparison = compareResults(original, candidate);

    console.log(`  Original value: ${0.1 + 0.2}`);
    console.log(`  Candidate value: ${0.3}`);
    console.log(`  Equivalent: ${comparison.equivalent}`);

    if (!comparison.equivalent) {
        console.log('  ❌ FAILED: Expected floating point values to be equivalent within tolerance');
        return false;
    }
    console.log('  ✅ PASSED');
    return true;
}

/**
 * Test 5: NaN values should be equal to each other.
 */
function testNaNEquality() {
    console.log('\n=== Test 5: NaN Equality ===');

    const results = [
        {
            testModulePath: 'tests/nan.test.js',
            testFunctionName: 'test NaN return',
            functionGettingTested: 'divideByZero',
            loopIndex: 1,
            iterationId: '0_0',
            runtime: 1000,
            returnValue: [[], {}, NaN],
        },
    ];

    const originalDb = createTestDatabase(path.join(TEST_DIR, 'original5.sqlite'), results);
    const candidateDb = createTestDatabase(path.join(TEST_DIR, 'candidate5.sqlite'), results);

    const original = readTestResults(originalDb);
    const candidate = readTestResults(candidateDb);
    const comparison = compareResults(original, candidate);

    console.log(`  Equivalent: ${comparison.equivalent}`);

    if (!comparison.equivalent) {
        console.log('  ❌ FAILED: Expected NaN values to be equivalent');
        return false;
    }
    console.log('  ✅ PASSED');
    return true;
}

/**
 * Main test runner.
 */
function main() {
    console.log('='.repeat(60));
    console.log('E2E Comparison Test Suite');
    console.log('='.repeat(60));

    // Setup
    if (fs.existsSync(TEST_DIR)) {
        fs.rmSync(TEST_DIR, { recursive: true });
    }
    fs.mkdirSync(TEST_DIR, { recursive: true });

    const results = [];
    results.push(testIdenticalResults());
    results.push(testDifferentReturnValues());
    results.push(testComplexTypes());
    results.push(testFloatingPointTolerance());
    results.push(testNaNEquality());

    // Cleanup
    fs.rmSync(TEST_DIR, { recursive: true });

    // Summary
    console.log('\n' + '='.repeat(60));
    console.log('Summary');
    console.log('='.repeat(60));
    const passed = results.filter(r => r).length;
    const total = results.length;
    console.log(`Passed: ${passed}/${total}`);

    if (passed === total) {
        console.log('\n✅ ALL TESTS PASSED');
        process.exit(0);
    } else {
        console.log('\n❌ SOME TESTS FAILED');
        process.exit(1);
    }
}

main();
