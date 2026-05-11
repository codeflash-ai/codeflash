/**
 * Bubble sort implementation - intentionally inefficient for optimization testing.
 */

/**
 * Sort an array using bubble sort algorithm.
 * @param {number[]} arr - The array to sort
 * @returns {number[]} - The sorted array
 */
function bubbleSort(arr) {
    const result = arr.slice();
    const n = result.length;

    if (n <= 1) return result;

    for (let i = 0; i < n - 1; i++) {
        let swapped = false;
        const limit = n - i - 1;
        for (let j = 0; j < limit; j++) {
            const a = result[j];
            const b = result[j + 1];
            if (a > b) {
                result[j] = b;
                result[j + 1] = a;
                swapped = true;
            }
        }
        if (!swapped) break;
    }

    return result;
}


module.exports = { bubbleSort };
