/**
 * Math utility functions - basic arithmetic operations.
 */

/**
 * Add two numbers.
 * @param a - First number
 * @param b - Second number
 * @returns Sum of a and b
 */
export function add(a, b) {
    return a + b;
}

/**
 * Multiply two numbers.
 * @param a - First number
 * @param b - Second number
 * @returns Product of a and b
 */
export function multiply(a, b) {
    return a * b;
}

/**
 * Calculate factorial recursively.
 * @param n - Non-negative integer
 * @returns Factorial of n
 */
export function factorial(n) {
    // Intentionally inefficient recursive implementation
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

/**
 * Calculate power using repeated multiplication.
 * @param base - Base number
 * @param exp - Exponent
 * @returns base raised to exp
 */
export function power(base, exp) {
    // Inefficient: linear time instead of log time
    let result = 1;
    for (let i = 0; i < exp; i++) {
        result = multiply(result, base);
    }
    return result;
}

module.exports = {
    add,
    multiply,
    factorial,
    power
};