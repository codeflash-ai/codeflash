/**
 * Calculator class - TypeScript version.
 * Demonstrates class method optimization with typed imports.
 */

import { add, multiply, factorial } from './math_utils';
import { formatNumber, validateInput } from './helpers/format';

interface HistoryEntry {
    type: string;
    result: number;
}

export class Calculator {
    private precision: number;
    private history: HistoryEntry[];

    constructor(precision: number = 2) {
        this.precision = precision;
        this.history = [];
    }

    /**
     * Calculate compound interest with multiple helper dependencies.
     * @param principal - Initial amount
     * @param rate - Interest rate (as decimal)
     * @param time - Time in years
     * @param n - Compounding frequency per year
     * @returns Compound interest result
     */
    calculateCompoundInterest(principal: number, rate: number, time: number, n: number): number {
        validateInput(principal, 'principal');
        validateInput(rate, 'rate');

        // Inefficient: recalculates power multiple times
        let result = principal;
        for (let i = 0; i < n * time; i++) {
            result = multiply(result, add(1, rate / n));
        }

        const interest = result - principal;
        this.history.push({ type: 'compound', result: interest });
        return formatNumber(interest, this.precision);
    }

    /**
     * Calculate permutation using factorial helper.
     * @param n - Total items
     * @param r - Items to choose
     * @returns Permutation result
     */
    permutation(n: number, r: number): number {
        if (n < r) return 0;
        // Inefficient: calculates factorial(n) fully even when not needed
        return factorial(n) / factorial(n - r);
    }

    /**
     * Get calculation history.
     */
    getHistory(): HistoryEntry[] {
        return [...this.history];
    }

    /**
     * Static method for quick calculations.
     */
    static quickAdd(a: number, b: number): number {
        return add(a, b);
    }
}

export default Calculator;