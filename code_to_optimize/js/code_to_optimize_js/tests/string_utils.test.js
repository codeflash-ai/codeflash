const {
    reverseString,
    isPalindrome,
    countOccurrences,
    longestCommonPrefix,
    toTitleCase
} = require('../string_utils');

describe('reverseString', () => {
    test('reverses a simple string', () => {
        expect(reverseString('hello')).toBe('olleh');
    });

    test('returns empty string for empty input', () => {
        expect(reverseString('')).toBe('');
    });

    test('handles single character', () => {
        expect(reverseString('a')).toBe('a');
    });

    test('handles palindrome', () => {
        expect(reverseString('radar')).toBe('radar');
    });

    test('handles spaces', () => {
        expect(reverseString('hello world')).toBe('dlrow olleh');
    });

    test('reverses a longer string for performance', () => {
        const input = 'abcdefghijklmnopqrstuvwxyz'.repeat(20);
        const result = reverseString(input);
        expect(result.length).toBe(input.length);
        expect(result[0]).toBe('z');
        expect(result[result.length - 1]).toBe('a');
    });

    test('reverses a medium string', () => {
        const input = 'The quick brown fox jumps over the lazy dog';
        const expected = 'god yzal eht revo spmuj xof nworb kciuq ehT';
        expect(reverseString(input)).toBe(expected);
    });
});

describe('isPalindrome', () => {
    test('returns true for simple palindrome', () => {
        expect(isPalindrome('radar')).toBe(true);
    });

    test('returns true for palindrome with mixed case', () => {
        expect(isPalindrome('RaceCar')).toBe(true);
    });

    test('returns true for palindrome with spaces and punctuation', () => {
        expect(isPalindrome('A man, a plan, a canal: Panama')).toBe(true);
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
});

describe('countOccurrences', () => {
    test('counts single occurrence', () => {
        expect(countOccurrences('hello', 'ell')).toBe(1);
    });

    test('counts multiple occurrences', () => {
        expect(countOccurrences('abababab', 'ab')).toBe(4);
    });

    test('returns 0 for no occurrences', () => {
        expect(countOccurrences('hello', 'xyz')).toBe(0);
    });

    test('handles overlapping matches', () => {
        expect(countOccurrences('aaa', 'aa')).toBe(2);
    });

    test('handles empty substring', () => {
        expect(countOccurrences('hello', '')).toBe(6);
    });
});

describe('longestCommonPrefix', () => {
    test('finds common prefix', () => {
        expect(longestCommonPrefix(['flower', 'flow', 'flight'])).toBe('fl');
    });

    test('returns empty for no common prefix', () => {
        expect(longestCommonPrefix(['dog', 'racecar', 'car'])).toBe('');
    });

    test('returns empty for empty array', () => {
        expect(longestCommonPrefix([])).toBe('');
    });

    test('returns the string for single element array', () => {
        expect(longestCommonPrefix(['hello'])).toBe('hello');
    });

    test('handles identical strings', () => {
        expect(longestCommonPrefix(['test', 'test', 'test'])).toBe('test');
    });
});

describe('toTitleCase', () => {
    test('converts simple string', () => {
        expect(toTitleCase('hello world')).toBe('Hello World');
    });

    test('handles already title case', () => {
        expect(toTitleCase('Hello World')).toBe('Hello World');
    });

    test('handles uppercase input', () => {
        expect(toTitleCase('HELLO WORLD')).toBe('Hello World');
    });

    test('handles single word', () => {
        expect(toTitleCase('hello')).toBe('Hello');
    });

    test('handles empty string', () => {
        expect(toTitleCase('')).toBe('');
    });
});
