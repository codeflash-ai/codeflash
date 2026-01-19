/**
 * Fibonacci implementations - intentionally inefficient for optimization testing.
 */

/**
 * Calculate the nth Fibonacci number using naive recursion.
 * This is intentionally slow to demonstrate optimization potential.
 * @param n - The index of the Fibonacci number to calculate
 * @returns The nth Fibonacci number
 */
const memo: Map<number, number> = new Map();

/**
 * Optimized fibonacci with memoization and a fast iterative path for integer n.
 */
export function fibonacci(n: number): number {
    if (memo.has(n)) {
        return memo.get(n)!;
    }

    if (n <= 1) {
        memo.set(n, n);
        return n;
    }

    // Fast path for integer inputs: iterative O(n)
    if (Number.isInteger(n)) {
        // compute bottom-up to avoid recursion and repeated work
        let a = 0;
        let b = 1;
        // handle n === 0 or n === 1 already above, so start at 2
        for (let i = 2, len = n; i <= len; i++) {
            const c = a + b;
            a = b;
            b = c;
        }
        memo.set(0, 0);
        memo.set(1, 1);
        memo.set(n, b);
        return b;
    }

    // Non-integer fallback: use memoized recursion to preserve original behavior
    const result = fibonacci(n - 1) + fibonacci(n - 2);
    memo.set(n, result);
    return result;
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
