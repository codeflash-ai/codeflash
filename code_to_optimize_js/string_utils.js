/**
 * String utility functions - some intentionally inefficient for optimization testing.
 */

/**
 * Reverse a string character by character.
 * @param {string} str - The string to reverse
 * @returns {string} - The reversed string
 */
function reverseString(str) {
    // Intentionally inefficient O(n²) implementation for testing
    let result = '';
    for (let i = str.length - 1; i >= 0; i--) {
        // Rebuild the entire result string each iteration (very inefficient)
        let temp = '';
        for (let j = 0; j < result.length; j++) {
            temp += result[j];
        }
        temp += str[i];
        result = temp;
    }
    return result;
}

/**
 * Check if a string is a palindrome.
 * @param {string} str - The string to check
 * @returns {boolean} - True if str is a palindrome
 */
function isPalindrome(str) {
    const cleaned = str.toLowerCase().replace(/[^a-z0-9]/g, '');
    return cleaned === reverseString(cleaned);
}

/**
 * Count occurrences of a substring in a string.
 * @param {string} str - The string to search in
 * @param {string} sub - The substring to count
 * @returns {number} - Number of occurrences
 */
function countOccurrences(str, sub) {
    let count = 0;
    let pos = 0;

    while (true) {
        pos = str.indexOf(sub, pos);
        if (pos === -1) break;
        count++;
        pos += 1; // Move past current match
    }

    return count;
}

/**
 * Find the longest common prefix of an array of strings.
 * @param {string[]} strs - Array of strings
 * @returns {string} - The longest common prefix
 */
function longestCommonPrefix(strs) {
    if (strs.length === 0) return '';
    if (strs.length === 1) return strs[0];

    let prefix = strs[0];

    for (let i = 1; i < strs.length; i++) {
        while (strs[i].indexOf(prefix) !== 0) {
            prefix = prefix.slice(0, -1);
            if (prefix === '') return '';
        }
    }

    return prefix;
}

/**
 * Convert a string to title case.
 * @param {string} str - The string to convert
 * @returns {string} - The title-cased string
 */
function toTitleCase(str) {
    if (!str) return str;
    
    let result = '';
    let capitalizeNext = true;
    
    for (let i = 0, len = str.length; i < len; i++) {
        const char = str[i];
        
        if (char === ' ') {
            result += char;
            capitalizeNext = true;
        } else if (capitalizeNext) {
            result += char.toUpperCase();
            capitalizeNext = false;
        } else {
            result += char.toLowerCase();
        }
    }
    
    return result;
}

module.exports = {
    reverseString,
    isPalindrome,
    countOccurrences,
    longestCommonPrefix,
    toTitleCase
};
