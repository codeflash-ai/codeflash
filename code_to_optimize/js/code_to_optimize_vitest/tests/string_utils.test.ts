import { describe, test, expect } from 'vitest';
import { reverseString, isPalindrome, countVowels, uniqueWords } from '../string_utils';

describe('reverseString', () => {
    test('reverses a simple string', () => {
        expect(reverseString('hello')).toBe('olleh');
    });

    test('returns empty string for empty input', () => {
        expect(reverseString('')).toBe('');
    });

    test('returns same character for single character', () => {
        expect(reverseString('a')).toBe('a');
    });

    test('handles strings with spaces', () => {
        expect(reverseString('hello world')).toBe('dlrow olleh');
    });

    test('handles palindrome', () => {
        expect(reverseString('racecar')).toBe('racecar');
    });
});

describe('isPalindrome', () => {
    test('returns true for palindrome', () => {
        expect(isPalindrome('racecar')).toBe(true);
    });

    test('returns true for palindrome with spaces', () => {
        expect(isPalindrome('A man a plan a canal Panama')).toBe(true);
    });

    test('returns false for non-palindrome', () => {
        expect(isPalindrome('hello')).toBe(false);
    });

    test('returns true for empty string', () => {
        expect(isPalindrome('')).toBe(true);
    });

    test('returns true for single character', () => {
        expect(isPalindrome('a')).toBe(true);
    });

    test('handles mixed case', () => {
        expect(isPalindrome('RaceCar')).toBe(true);
    });
});

describe('countVowels', () => {
    test('counts vowels in simple string', () => {
        expect(countVowels('hello')).toBe(2);
    });

    test('returns 0 for string with no vowels', () => {
        expect(countVowels('bcdfg')).toBe(0);
    });

    test('returns 0 for empty string', () => {
        expect(countVowels('')).toBe(0);
    });

    test('counts uppercase vowels', () => {
        expect(countVowels('HELLO')).toBe(2);
    });

    test('counts all vowels', () => {
        expect(countVowels('aeiouAEIOU')).toBe(10);
    });
});

describe('uniqueWords', () => {
    test('finds unique words in simple string', () => {
        expect(uniqueWords('hello world')).toEqual(['hello', 'world']);
    });

    test('removes duplicates', () => {
        expect(uniqueWords('hello hello world')).toEqual(['hello', 'world']);
    });

    test('returns empty array for empty string', () => {
        expect(uniqueWords('')).toEqual([]);
    });

    test('handles multiple spaces', () => {
        expect(uniqueWords('hello   world')).toEqual(['hello', 'world']);
    });

    test('normalizes case', () => {
        expect(uniqueWords('Hello hello HELLO')).toEqual(['hello']);
    });
});
