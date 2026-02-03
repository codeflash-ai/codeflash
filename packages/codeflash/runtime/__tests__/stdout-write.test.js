/**
 * Test: process.stdout.write for timing markers
 *
 * This test verifies that capturePerf uses process.stdout.write instead of
 * console.log for timing markers. This is required because Vitest intercepts
 * console.log output and may not pass it to the subprocess stdout.
 *
 * Run with: node __tests__/stdout-write.test.js
 */

const assert = require('assert');

// Mock process.stdout.write to capture output
const originalStdoutWrite = process.stdout.write;
let stdoutOutput = [];

process.stdout.write = function(data) {
    stdoutOutput.push(data.toString());
    return true;
};

// Also mock console.log to track it
const originalConsoleLog = console.log;
let consoleLogCalls = [];

console.log = function(...args) {
    consoleLogCalls.push(args.join(' '));
};

// Set up environment for capturePerf
process.env.CODEFLASH_PERF_LOOP_COUNT = '1';

// Load capture module
const capture = require('../capture');

// Set test name (simulating Jest's beforeEach)
capture.setTestName('test_function_name');

// Clear output arrays
stdoutOutput = [];
consoleLogCalls = [];

// Test function
function testFunction(x) {
    return x * 2;
}

// Run capturePerf
const result = capture.capturePerf('testFunction', '42', testFunction, 5);

// Restore mocks
process.stdout.write = originalStdoutWrite;
console.log = originalConsoleLog;

console.log('Testing process.stdout.write for timing markers...\n');

// Test 1: End tag should use process.stdout.write
console.log('Test 1: End tag uses process.stdout.write');
console.log('  stdout.write calls:', stdoutOutput);

const stdoutEndTags = stdoutOutput.filter(line =>
    line.includes('!######') && line.includes('######!') && !line.includes('!$')
);
assert.ok(stdoutEndTags.length >= 1, 'End tag should be written to stdout');
console.log('  PASS: End tag written to stdout\n');

// Test 2: End tag should NOT use console.log
console.log('Test 2: End tag should not use console.log');
console.log('  console.log calls:', consoleLogCalls);

const consoleEndTags = consoleLogCalls.filter(line =>
    line.includes('!######') && line.includes('######!') && !line.includes('!$')
);
assert.strictEqual(consoleEndTags.length, 0, 'End tag should NOT be logged via console.log');
console.log('  PASS: End tag not using console.log\n');

// Test 3: stdout.write output should have newline
console.log('Test 3: Output includes newline');
const endTag = stdoutEndTags[0];
assert.ok(endTag.endsWith('\n'), 'Output should end with newline');
console.log('  PASS: Output ends with newline\n');

// Test 4: Return value should be correct
console.log('Test 4: Return value');
assert.strictEqual(result, 10, 'Return value should be 10 (5 * 2)');
console.log('  PASS: Return value correct\n');

// Cleanup
delete process.env.CODEFLASH_PERF_LOOP_COUNT;

console.log('All tests passed!');
