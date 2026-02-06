/**
 * Calculator module - demonstrates cross-file function calls.
 * Uses helper functions from math_helpers.js.
 */

const { sumArray, average, findMax, findMin } = require('./math_helpers');


/**
 * Calculate statistics for an array of numbers.
 * @param numbers - Array of numbers to analyze
 * @returns Object containing sum, average, min, max, and range
 */
export function calculateStats(numbers) {
    if (numbers.length === 0) {
        return {
            sum: 0,
            average: 0,
            min: 0,
            max: 0,
            range: 0
        };
    }

    const sum = sumArray(numbers);
    const avg = average(numbers);
    const min = findMin(numbers);
    const max = findMax(numbers);
    const range = max - min;

    return {
        sum,
        average: avg,
        min,
        max,
        range
    };
}

/**
 * Normalize an array of numbers to a 0-1 range.
 * @param numbers - Array of numbers to normalize
 * @returns Normalized array
 */
export function normalizeArray(numbers) {
    if (numbers.length === 0) return [];

    const min = findMin(numbers);
    const max = findMax(numbers);
    const range = max - min;

    if (range === 0) {
        return numbers.map(() => 0.5);
    }

    return numbers.map(n => (n - min) / range);
}

/**
 * Calculate the weighted average of values with corresponding weights.
 * @param values - Array of values
 * @param weights - Array of weights (same length as values)
 * @returns The weighted average
 */
export function weightedAverage(values, weights) {
    if (values.length === 0 || values.length !== weights.length) {
        return 0;
    }

    let weightedSum = 0;
    for (let i = 0; i < values.length; i++) {
        weightedSum += values[i] * weights[i];
    }

    const totalWeight = sumArray(weights);
    if (totalWeight === 0) return 0;

    return weightedSum / totalWeight;
}

module.exports = {
    calculateStats,
    normalizeArray,
    weightedAverage
};
