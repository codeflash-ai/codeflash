const _fibArray = [0, 1];

/**
 * Fibonacci implementations - ES Module
 * Intentionally inefficient for optimization testing.
 */

/**
 * Calculate the nth Fibonacci number using naive recursion.
 * This is intentionally slow to demonstrate optimization potential.
 * @param {number} n - The index of the Fibonacci number to calculate
 * @returns {number} The nth Fibonacci number
 */
export function fibonacci(n) {
    if (n <= 1) {
        return n;
    }

    if (typeof n === 'number' && Number.isInteger(n) && n >= 0) {
        if (n < _fibArray.length) {
            return _fibArray[n];
        }
        let a = _fibArray[_fibArray.length - 2];
        let b = _fibArray[_fibArray.length - 1];
        for (let i = _fibArray.length; i <= n; i++) {
            const c = a + b;
            _fibArray.push(c);
            a = b;
            b = c;
        }
        return _fibArray[n];
    }

    return fibonacci(n - 1) + fibonacci(n - 2);
}

/**
 * Check if a number is a Fibonacci number.
 * @param {number} num - The number to check
 * @returns {boolean} True if num is a Fibonacci number
 */
export function isFibonacci(num) {
    // A number is Fibonacci if one of (5*n*n + 4) or (5*n*n - 4) is a perfect square
    const check1 = 5 * num * num + 4;
    const check2 = 5 * num * num - 4;
    return isPerfectSquare(check1) || isPerfectSquare(check2);
}

/**
 * Check if a number is a perfect square.
 * @param {number} n - The number to check
 * @returns {boolean} True if n is a perfect square
 */
export function isPerfectSquare(n) {
    const sqrt = Math.sqrt(n);
    return sqrt === Math.floor(sqrt);
}

/**
 * Generate an array of Fibonacci numbers up to n.
 * @param {number} n - The number of Fibonacci numbers to generate
 * @returns {number[]} Array of Fibonacci numbers
 */
export function fibonacciSequence(n) {
    const result = [];
    for (let i = 0; i < n; i++) {
        result.push(fibonacci(i));
    }
    return result;
}
