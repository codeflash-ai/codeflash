import { reverseString, isPalindrome, countOccurrences, longestCommonPrefix, toTitleCase } from '../string_utils';

describe('reverseString', () => {
    test('reverses an empty string', () => {
        expect(reverseString('')).toBe('');
    });

    test('reverses a single character', () => {
        expect(reverseString('a')).toBe('a');
    });

    test('reverses a word', () => {
        expect(reverseString('hello')).toBe('olleh');
    });

    test('reverses a sentence', () => {
        expect(reverseString('hello world')).toBe('dlrow olleh');
    });

    test('handles special characters', () => {
        expect(reverseString('a!b@c#')).toBe('#c@b!a');
    });
});

describe('isPalindrome', () => {
    test('returns true for empty string', () => {
        expect(isPalindrome('')).toBe(true);
    });

    test('returns true for single character', () => {
        expect(isPalindrome('a')).toBe(true);
    });

    test('returns true for palindrome word', () => {
        expect(isPalindrome('racecar')).toBe(true);
    });

    test('returns true for palindrome with mixed case', () => {
        expect(isPalindrome('RaceCar')).toBe(true);
    });

    test('returns true for palindrome with spaces', () => {
        expect(isPalindrome('A man a plan a canal Panama')).toBe(true);
    });

    test('returns false for non-palindrome', () => {
        expect(isPalindrome('hello')).toBe(false);
    });
});

describe('countOccurrences', () => {
    test('returns 0 for empty string', () => {
        expect(countOccurrences('', 'a')).toBe(0);
    });

    test('returns 0 when substring not found', () => {
        expect(countOccurrences('hello', 'x')).toBe(0);
    });

    test('counts single occurrence', () => {
        expect(countOccurrences('hello', 'e')).toBe(1);
    });

    test('counts multiple occurrences', () => {
        expect(countOccurrences('hello', 'l')).toBe(2);
    });

    test('counts overlapping occurrences', () => {
        expect(countOccurrences('aaa', 'aa')).toBe(2);
    });

    test('counts multi-character substring', () => {
        expect(countOccurrences('abcabc', 'abc')).toBe(2);
    });
});

describe('longestCommonPrefix', () => {
    test('returns empty for empty array', () => {
        expect(longestCommonPrefix([])).toBe('');
    });

    test('returns the string for single element', () => {
        expect(longestCommonPrefix(['hello'])).toBe('hello');
    });

    test('finds common prefix', () => {
        expect(longestCommonPrefix(['flower', 'flow', 'flight'])).toBe('fl');
    });

    test('returns empty when no common prefix', () => {
        expect(longestCommonPrefix(['dog', 'racecar', 'car'])).toBe('');
    });

    test('handles identical strings', () => {
        expect(longestCommonPrefix(['test', 'test', 'test'])).toBe('test');
    });
});

describe('toTitleCase', () => {
    test('converts single word', () => {
        expect(toTitleCase('hello')).toBe('Hello');
    });

    test('converts multiple words', () => {
        expect(toTitleCase('hello world')).toBe('Hello World');
    });

    test('handles already title case', () => {
        expect(toTitleCase('Hello World')).toBe('Hello World');
    });

    test('handles all uppercase', () => {
        expect(toTitleCase('HELLO WORLD')).toBe('Hello World');
    });

    test('handles empty string', () => {
        expect(toTitleCase('')).toBe('');
    });
});
