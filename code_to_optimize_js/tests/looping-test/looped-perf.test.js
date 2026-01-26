/**
 * Test for session-level looping performance measurement.
 *
 * Note: Looping is now done at the session level by Python (test_runner.py)
 * which runs Jest multiple times. Each Jest run executes the test once,
 * and timing data is aggregated across runs for stability checking.
 */

// Load the codeflash helper from npm package
const codeflash = require('@codeflash/cli');

// Simple function to test
function fibonacci(n) {
    if (n <= 1) return n;
    let a = 0, b = 1;
    for (let i = 2; i <= n; i++) {
        const temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

describe('Session-Level Looping Performance Test', () => {
    test('fibonacci(20) with session-level looping', () => {
        // Looping is controlled by Python via CODEFLASH_LOOP_INDEX env var
        const result = codeflash.capturePerf('fibonacci', '10', fibonacci, 20);
        expect(result).toBe(6765);
    });

    test('fibonacci(30) with session-level looping', () => {
        const result = codeflash.capturePerf('fibonacci', '16', fibonacci, 30);
        expect(result).toBe(832040);
    });
});
