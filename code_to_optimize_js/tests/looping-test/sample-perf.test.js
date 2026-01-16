/**
 * Sample performance test to verify looping mechanism.
 */

const path = require('path');

// Load the codeflash helper from the project root
const codeflash = require(path.join(__dirname, '..', '..', 'codeflash-jest-helper.js'));

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

describe('Looping Performance Test', () => {
    test('fibonacci(20) timing', () => {
        const result = codeflash.capturePerf('fibonacci', '10', fibonacci, 20);
        expect(result).toBe(6765);
    });

    test('fibonacci(30) timing', () => {
        const result = codeflash.capturePerf('fibonacci', '16', fibonacci, 30);
        expect(result).toBe(832040);
    });

    test('multiple calls in one test', () => {
        // Same lineId, multiple calls - should increment invocation counter
        const r1 = codeflash.capturePerf('fibonacci', '22', fibonacci, 5);
        const r2 = codeflash.capturePerf('fibonacci', '22', fibonacci, 10);
        const r3 = codeflash.capturePerf('fibonacci', '22', fibonacci, 15);

        expect(r1).toBe(5);
        expect(r2).toBe(55);
        expect(r3).toBe(610);
    });
});
