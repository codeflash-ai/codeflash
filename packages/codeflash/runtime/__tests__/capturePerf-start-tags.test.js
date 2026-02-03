/**
 * Test: capturePerf start tag output
 *
 * This test verifies that capturePerf outputs START tags before timing,
 * which is required for:
 * 1. Throughput calculation (counting completed executions)
 * 2. Timing marker matching in parse_test_output.py
 *
 * Run with: node __tests__/capturePerf-start-tags.test.js
 */

const assert = require('assert');

// Mock console.log to capture output
const originalConsoleLog = console.log;
let capturedOutput = [];

console.log = function(...args) {
    capturedOutput.push(args.join(' '));
    // Uncomment to also print to console for debugging:
    // originalConsoleLog.apply(console, args);
};

// Set up environment for capturePerf
process.env.CODEFLASH_PERF_LOOP_COUNT = '1';

// Load capture module
const capture = require('../capture');

// Set test name (simulating Jest's beforeEach)
capture.setTestName('test_function_name');

console.log = originalConsoleLog;
console.log('Testing capturePerf start tag output...\n');
console.log = function(...args) {
    capturedOutput.push(args.join(' '));
};

// Test function
function testFunction(x) {
    return x * 2;
}

// Clear captured output
capturedOutput = [];

// Test 1: capturePerf should output start tag
console.log = function(...args) {
    capturedOutput.push(args.join(' '));
};

const result = capture.capturePerf('testFunction', '42', testFunction, 5);

console.log = originalConsoleLog;

console.log('Test 1: Start tag output');
console.log('  Captured output:', capturedOutput);

// Check that start tag was output (contains !$###### and ######$!)
const startTags = capturedOutput.filter(line => line.includes('!$######') && line.includes('######$!'));
const endTags = capturedOutput.filter(line => line.includes('!######') && line.includes('######!') && !line.includes('!$'));

assert.strictEqual(startTags.length, 1, 'Should have exactly one start tag');
assert.strictEqual(endTags.length, 1, 'Should have exactly one end tag');
console.log('  PASS: Start and end tags present\n');

// Test 2: Start tag format should match expected pattern
console.log('Test 2: Start tag format');
const startTag = startTags[0];
// Format: !$######module:testName:funcName:loopIndex:invocationId######$!
assert.ok(startTag.includes('testFunction'), 'Start tag should contain function name');
assert.ok(startTag.includes('test_function_name'), 'Start tag should contain test name');
console.log('  PASS: Start tag format correct\n');

// Test 3: End tag should contain timing
console.log('Test 3: End tag contains timing');
const endTag = endTags[0];
// Format: !######module:testName:funcName:loopIndex:invocationId:durationNs######!
const parts = endTag.match(/!######(.*)######!/);
assert.ok(parts, 'End tag should match pattern');
const endTagContent = parts[1];
const colonCount = (endTagContent.match(/:/g) || []).length;
// module:testClass:testName:funcName:loopIndex:invocationId:durationNs = 6 colons
assert.ok(colonCount >= 5, 'End tag should have duration (6+ colons in content)');
console.log('  PASS: End tag contains timing data\n');

// Test 4: Return value should be correct
console.log('Test 4: Return value');
assert.strictEqual(result, 10, 'Return value should be 10 (5 * 2)');
console.log('  PASS: Return value correct\n');

// Cleanup
delete process.env.CODEFLASH_PERF_LOOP_COUNT;

console.log('All tests passed!');
