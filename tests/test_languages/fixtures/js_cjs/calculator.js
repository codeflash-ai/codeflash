/**
 * Calculator class - demonstrates class method optimization scenarios.
 * Uses helper functions from math_utils.js.
 */

const { add, multiply, factorial } = require('./math_utils');
const { formatNumber, validateInput } = require('./helpers/format');

class Calculator {
    constructor(precision = 2) {
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
    calculateCompoundInterest(principal, rate, time, n) {
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
    permutation(n, r) {
        if (n < r) return 0;
        // Inefficient: calculates factorial(n) fully even when not needed
        return factorial(n) / factorial(n - r);
    }

    /**
     * Static method for quick calculations.
     */
    static quickAdd(a, b) {
        return add(a, b);
    }
}

module.exports = { Calculator };