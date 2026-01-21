/**
 * Fibonacci implementations - intentionally inefficient for optimization testing.
 */

/**
 * Calculate the nth Fibonacci number using naive recursion.
 * This is intentionally slow to demonstrate optimization potential.
 * @param n - The index of the Fibonacci number to calculate
 * @returns The nth Fibonacci number
 */
export function fibonacci(n: number): number {
    if (n <= 1) {
        return n;
    }
    // Preserve original recursive semantics for non-integer inputs
    if (!Number.isInteger(n)) {
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
    // Iterative Fibonacci for integer n > 1 — O(n) time, O(1) memory
    let a = 0;
    let b = 1;
    for (let i = 2, len = n; i <= len; i++) {
        const c = a + b;
        a = b;
        b = c;
    }
    return b;
}

/**
 * Check if a number is a Fibonacci number.
 * @param num - The number to check
 * @returns True if num is a Fibonacci number
 */
export function isFibonacci(num: number): boolean {
    // A number is Fibonacci if one of (5*n*n + 4) or (5*n*n - 4) is a perfect square
    const check1 = 5 * num * num + 4;
    const check2 = 5 * num * num - 4;

    return isPerfectSquare(check1) || isPerfectSquare(check2);
}

/**
 * Check if a number is a perfect square.
 * @param n - The number to check
 * @returns True if n is a perfect square
 */
export function isPerfectSquare(n: number): boolean {
    const sqrt = Math.sqrt(n);
    return sqrt === Math.floor(sqrt);
}

/**
 * Generate an array of Fibonacci numbers up to n.
 * @param n - The number of Fibonacci numbers to generate
 * @returns Array of Fibonacci numbers
 */
export function fibonacciSequence(n: number): number[] {
    const result: number[] = [];
    for (let i = 0; i < n; i++) {
        result.push(fibonacci(i));
    }
    return result;
}
