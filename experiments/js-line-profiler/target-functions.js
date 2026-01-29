/**
 * Target functions to profile.
 * These represent different types of code patterns we want to measure.
 */

// Simple arithmetic function - good baseline
function fibonacci(n) {
    if (n <= 1) return n;
    let a = 0;
    let b = 1;
    for (let i = 2; i <= n; i++) {
        const temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

// String manipulation - common pattern
function reverseString(str) {
    let result = '';
    for (let i = str.length - 1; i >= 0; i--) {
        result += str[i];
    }
    return result;
}

// Array operations - heap allocations
function bubbleSort(arr) {
    const n = arr.length;
    const sorted = [...arr];
    for (let i = 0; i < n - 1; i++) {
        for (let j = 0; j < n - i - 1; j++) {
            if (sorted[j] > sorted[j + 1]) {
                const temp = sorted[j];
                sorted[j] = sorted[j + 1];
                sorted[j + 1] = temp;
            }
        }
    }
    return sorted;
}

// Object manipulation
function countWords(text) {
    const words = text.toLowerCase().split(/\s+/);
    const counts = {};
    for (const word of words) {
        if (word) {
            counts[word] = (counts[word] || 0) + 1;
        }
    }
    return counts;
}

// Nested loops - demonstrates hot spots
function matrixMultiply(a, b) {
    const rowsA = a.length;
    const colsA = a[0].length;
    const colsB = b[0].length;
    const result = [];

    for (let i = 0; i < rowsA; i++) {
        result[i] = [];
        for (let j = 0; j < colsB; j++) {
            let sum = 0;
            for (let k = 0; k < colsA; k++) {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

// Function with conditionals - branch coverage
function classifyNumber(n) {
    let result = '';
    if (n < 0) {
        result = 'negative';
    } else if (n === 0) {
        result = 'zero';
    } else if (n < 10) {
        result = 'small';
    } else if (n < 100) {
        result = 'medium';
    } else {
        result = 'large';
    }
    return result;
}

module.exports = {
    fibonacci,
    reverseString,
    bubbleSort,
    countWords,
    matrixMultiply,
    classifyNumber
};
