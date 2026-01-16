/**
 * Fibonacci implementations - intentionally inefficient for optimization testing.
 */

/**
 * Calculate the nth Fibonacci number using naive recursion.
 * This is intentionally slow to demonstrate optimization potential.
 * @param {number} n - The index of the Fibonacci number to calculate
 * @returns {number} - The nth Fibonacci number
 */
function fibonacci(n) {
    if (n <= 1) {
        return n;
    }

    // Fast doubling for integer n >= 2 gives O(log n) time.
    // Iterative implementation to avoid array allocations and recursion overhead
    if (Number.isInteger(n)) {
        // Find the most significant bit position
        let k = n;
        let bitPos = 0;
        while ((1 << (bitPos + 1)) <= k) {
            bitPos++;
        }

        // Start with F(1) = 1, F(2) = 1
        let a = 1;
        let b = 1;

        // Process bits from MSB to LSB (skipping the MSB itself)
        for (let i = bitPos - 1; i >= 0; i--) {
            // Double: F(2k) = F(k)[2*F(k+1) - F(k)], F(2k+1) = F(k)^2 + F(k+1)^2
            const c = a * (2 * b - a);
            const d = a * a + b * b;
            
            if ((k & (1 << i)) !== 0) {
                // Bit is 1: we need F(2k+1) and F(2k+2)
                a = d;
                b = c + d;
            } else {
                // Bit is 0: we need F(2k) and F(2k+1)
                a = c;
                b = d;
            }
        }

        return a;
    }

    // For non-integer n > 1, memoize the recursive definition to avoid exponential blowup.
    const cache = new Map();
    function memo(x) {
        if (x <= 1) {
            return x;
        }
        const cached = cache.get(x);
        if (cached !== undefined) {
            return cached;
        }
        const res = memo(x - 1) + memo(x - 2);
        cache.set(x, res);
        return res;
    }

    return memo(n);
}

/**
 * Check if a number is a Fibonacci number.
 * @param {number} num - The number to check
 * @returns {boolean} - True if num is a Fibonacci number
 */
function isFibonacci(num) {
    // A number is Fibonacci if one of (5*n*n + 4) or (5*n*n - 4) is a perfect square
    const check1 = 5 * num * num + 4;
    const check2 = 5 * num * num - 4;

    return isPerfectSquare(check1) || isPerfectSquare(check2);
}

/**
 * Check if a number is a perfect square.
 * @param {number} n - The number to check
 * @returns {boolean} - True if n is a perfect square
 */
function isPerfectSquare(n) {
    const sqrt = Math.sqrt(n);
    return sqrt === Math.floor(sqrt);
}

/**
 * Generate an array of Fibonacci numbers up to n.
 * @param {number} n - The number of Fibonacci numbers to generate
 * @returns {number[]} - Array of Fibonacci numbers
 */
function fibonacciSequence(n) {
    const result = [];
    for (let i = 0; i < n; i++) {
        result.push(fibonacci(i));
    }
    return result;
}

module.exports = { fibonacci, isFibonacci, isPerfectSquare, fibonacciSequence };
