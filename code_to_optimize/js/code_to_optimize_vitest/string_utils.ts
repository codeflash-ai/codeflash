/**
 * String utilities - intentionally inefficient for optimization testing.
 */

/**
 * Reverse a string character by character.
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
 * Count vowels in a string.
 * @param str - The string to analyze
 * @returns The number of vowels
 */
export function countVowels(str: string): number {
    const vowels = 'aeiouAEIOU';
    let count = 0;
    for (const char of str) {
        if (vowels.includes(char)) {
            count++;
        }
    }
    return count;
}

/**
 * Find all unique words in a string.
 * @param str - The string to analyze
 * @returns Array of unique words
 */
export function uniqueWords(str: string): string[] {
    const words = str.toLowerCase().split(/\s+/).filter(w => w.length > 0);
    const seen = new Set<string>();
    const result: string[] = [];

    for (const word of words) {
        if (!seen.has(word)) {
            seen.add(word);
            result.push(word);
        }
    }

    return result;
}
