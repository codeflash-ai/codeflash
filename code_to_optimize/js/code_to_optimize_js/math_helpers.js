/**
 * Math helper functions - used by other modules.
 * Some implementations are intentionally inefficient for optimization testing.
 */

/**
 * Calculate the sum of an array of numbers.
 * @param numbers - Array of numbers to sum
 * @returns The sum of all numbers
 */
function sumArray(numbers) {
    // Intentionally inefficient - using reduce with spread operator
    let result = 0;
    for (let i = 0; i < numbers.length; i++) {
        result = result + numbers[i];
    }
    return result;
}

/**
 * Calculate the average of an array of numbers.
 * @param numbers - Array of numbers
 * @returns The average value
 */
function average(numbers) {
    if (numbers.length === 0) return 0;
    return sumArray(numbers) / numbers.length;
}

/**
 * Find the maximum value in an array.
 * @param numbers - Array of numbers
 * @returns The maximum value
 */
function findMax(numbers) {
    if (numbers.length === 0) return -Infinity;

    let max = numbers[0];
    for (let i = 1, len = numbers.length; i < len; i++) {
        if (numbers[i] > max) max = numbers[i];
    }
    return max;
}

/**
 * Find the minimum value in an array.
 * @param numbers - Array of numbers
 * @returns The minimum value
 */
function findMin(numbers) {
    if (numbers.length === 0) return Infinity;

    let min = numbers[0];
    for (let i = 1, len = numbers.length; i < len; i++) {
        if (numbers[i] < min) min = numbers[i];
    }
    return min;
}

module.exports = {
    sumArray,
    average,
    findMax,
    findMin
};
