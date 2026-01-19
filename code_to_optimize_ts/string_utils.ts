/**
 * String utility functions - intentionally inefficient for optimization testing.
 */

/**
 * Reverse a string character by character.
 * This is intentionally inefficient - uses string concatenation in a loop.
 * @param str - The string to reverse
 * @returns The reversed string
 */
export function reverseString(str: string): string {
    let result = '';
    for (let i = str.length - 1; i >= 0; i--) {
        result += str[i];
    }
    return result;
}

/**
 * Check if a string is a palindrome.
 * @param str - The string to check
 * @returns True if the string is a palindrome
 */
export function isPalindrome(str: string): boolean {
    const cleaned = str.toLowerCase().replace(/[^a-z0-9]/g, '');
    return cleaned === reverseString(cleaned);
}

/**
 * Count occurrences of a substring in a string.
 * @param str - The string to search in
 * @param substr - The substring to count
 * @returns The number of occurrences
 */
export function countOccurrences(str: string, substr: string): number {
    let count = 0;
    let pos = 0;

    while ((pos = str.indexOf(substr, pos)) !== -1) {
        count++;
        pos += 1; // Move forward to find overlapping occurrences
    }

    return count;
}

/**
 * Find the longest common prefix among an array of strings.
 * @param strs - Array of strings
 * @returns The longest common prefix
 */
export function longestCommonPrefix(strs: string[]): string {
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
 * @param str - The string to convert
 * @returns The title-cased string
 */
export function toTitleCase(str: string): string {
    return str
        .toLowerCase()
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}
