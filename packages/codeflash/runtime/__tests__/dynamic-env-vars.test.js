/**
 * Test: Dynamic environment variable reading
 *
 * This test verifies that the performance configuration functions read
 * environment variables at runtime rather than at module load time.
 *
 * This is critical for Vitest compatibility, where modules may be cached
 * and loaded before environment variables are set.
 *
 * Run with: node __tests__/dynamic-env-vars.test.js
 */

const assert = require('assert');

// Clear any existing env vars before loading the module
delete process.env.CODEFLASH_PERF_LOOP_COUNT;
delete process.env.CODEFLASH_PERF_MIN_LOOPS;
delete process.env.CODEFLASH_PERF_TARGET_DURATION_MS;
delete process.env.CODEFLASH_PERF_BATCH_SIZE;
delete process.env.CODEFLASH_PERF_STABILITY_CHECK;
delete process.env.CODEFLASH_PERF_CURRENT_BATCH;

// Now load the module - at this point env vars are not set
const capture = require('../capture');

console.log('Testing dynamic environment variable reading...\n');

// Test 1: Default values when env vars are not set
console.log('Test 1: Default values');
assert.strictEqual(capture.getPerfLoopCount(), 1, 'getPerfLoopCount default should be 1');
assert.strictEqual(capture.getPerfMinLoops(), 5, 'getPerfMinLoops default should be 5');
assert.strictEqual(capture.getPerfTargetDurationMs(), 10000, 'getPerfTargetDurationMs default should be 10000');
assert.strictEqual(capture.getPerfBatchSize(), 10, 'getPerfBatchSize default should be 10');
assert.strictEqual(capture.getPerfStabilityCheck(), false, 'getPerfStabilityCheck default should be false');
assert.strictEqual(capture.getPerfCurrentBatch(), 0, 'getPerfCurrentBatch default should be 0');
console.log('  PASS: All defaults correct\n');

// Test 2: Values change when env vars are set AFTER module load
// This is the critical test - if these were constants, they would still return defaults
console.log('Test 2: Dynamic reading after module load');
process.env.CODEFLASH_PERF_LOOP_COUNT = '100';
process.env.CODEFLASH_PERF_MIN_LOOPS = '10';
process.env.CODEFLASH_PERF_TARGET_DURATION_MS = '5000';
process.env.CODEFLASH_PERF_BATCH_SIZE = '20';
process.env.CODEFLASH_PERF_STABILITY_CHECK = 'true';
process.env.CODEFLASH_PERF_CURRENT_BATCH = '5';

assert.strictEqual(capture.getPerfLoopCount(), 100, 'getPerfLoopCount should read 100 from env');
assert.strictEqual(capture.getPerfMinLoops(), 10, 'getPerfMinLoops should read 10 from env');
assert.strictEqual(capture.getPerfTargetDurationMs(), 5000, 'getPerfTargetDurationMs should read 5000 from env');
assert.strictEqual(capture.getPerfBatchSize(), 20, 'getPerfBatchSize should read 20 from env');
assert.strictEqual(capture.getPerfStabilityCheck(), true, 'getPerfStabilityCheck should read true from env');
assert.strictEqual(capture.getPerfCurrentBatch(), 5, 'getPerfCurrentBatch should read 5 from env');
console.log('  PASS: Dynamic reading works correctly\n');

// Test 3: Values change again when env vars are modified
console.log('Test 3: Values update when env vars change');
process.env.CODEFLASH_PERF_LOOP_COUNT = '500';
process.env.CODEFLASH_PERF_BATCH_SIZE = '50';

assert.strictEqual(capture.getPerfLoopCount(), 500, 'getPerfLoopCount should update to 500');
assert.strictEqual(capture.getPerfBatchSize(), 50, 'getPerfBatchSize should update to 50');
console.log('  PASS: Values update correctly\n');

// Cleanup
delete process.env.CODEFLASH_PERF_LOOP_COUNT;
delete process.env.CODEFLASH_PERF_MIN_LOOPS;
delete process.env.CODEFLASH_PERF_TARGET_DURATION_MS;
delete process.env.CODEFLASH_PERF_BATCH_SIZE;
delete process.env.CODEFLASH_PERF_STABILITY_CHECK;
delete process.env.CODEFLASH_PERF_CURRENT_BATCH;

console.log('All tests passed!');
