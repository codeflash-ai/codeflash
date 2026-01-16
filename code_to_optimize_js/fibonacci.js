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

    // Match original behavior where the loop effectively runs up to floor(n)
    let m = Math.floor(n);

    // Fast doubling: maintain (a, b) = (F(k), F(k+1))
    let a = 0;
    let b = 1;

    // Find highest power of two <= m using multiplication to avoid 32-bit shift limits
    let highest = 1;
    while (highest <= m) {
        highest *= 2;
    }
    highest /= 2;

    // Consume bits from highest to lowest: for each bit, double the index;
    // if the bit is set, advance by one.
    while (highest >= 1) {
        // c = F(2k) = F(k) * (2*F(k+1) - F(k))
        // d = F(2k+1) = F(k)^2 + F(k+1)^2
        const c = a * (2 * b - a);
        const d = a * a + b * b;

        if (m >= highest) {
            m -= highest;
            a = d;
            b = c + d;
        } else {
            a = c;
            b = d;
        }

        highest /= 2;
    }

    return a;
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
