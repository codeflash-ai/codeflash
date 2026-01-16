/**
 * Test for internal looping performance measurement.
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

describe('Looped Performance Test', () => {
    test('fibonacci(20) with internal looping', () => {
        // This will loop internally based on CODEFLASH_* env vars
        const result = codeflash.capturePerfLooped('fibonacci', '10', fibonacci, 20);
        expect(result).toBe(6765);
    });

    test('fibonacci(30) with internal looping', () => {
        const result = codeflash.capturePerfLooped('fibonacci', '16', fibonacci, 30);
        expect(result).toBe(832040);
    });
});
