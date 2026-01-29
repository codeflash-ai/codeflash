import { bubbleSort, bubbleSortDescending, isSorted } from '../bubble_sort';

describe('bubbleSort', () => {
    test('sorts an empty array', () => {
        expect(bubbleSort([])).toEqual([]);
    });

    test('sorts a single element array', () => {
        expect(bubbleSort([1])).toEqual([1]);
    });

    test('sorts an already sorted array', () => {
        expect(bubbleSort([1, 2, 3, 4, 5])).toEqual([1, 2, 3, 4, 5]);
    });

    test('sorts a reverse sorted array', () => {
        expect(bubbleSort([5, 4, 3, 2, 1])).toEqual([1, 2, 3, 4, 5]);
    });

    test('sorts an unsorted array', () => {
        expect(bubbleSort([3, 1, 4, 1, 5, 9, 2, 6])).toEqual([1, 1, 2, 3, 4, 5, 6, 9]);
    });

    test('handles negative numbers', () => {
        expect(bubbleSort([-3, -1, -4, -1, -5])).toEqual([-5, -4, -3, -1, -1]);
    });

    test('handles mixed positive and negative', () => {
        expect(bubbleSort([3, -1, 4, -1, 5])).toEqual([-1, -1, 3, 4, 5]);
    });

    test('does not mutate original array', () => {
        const original = [3, 1, 2];
        bubbleSort(original);
        expect(original).toEqual([3, 1, 2]);
    });

    test('sorts a larger reverse sorted array for performance', () => {
        const input: number[] = [];
        for (let i = 500; i >= 0; i--) {
            input.push(i);
        }
        const result = bubbleSort(input);
        expect(result[0]).toBe(0);
        expect(result[result.length - 1]).toBe(500);
    });

    test('sorts a larger random array for performance', () => {
        const input = [
            42, 17, 93, 8, 67, 31, 55, 22, 89, 4,
            76, 12, 39, 58, 95, 26, 71, 48, 83, 19,
            64, 3, 88, 37, 52, 11, 79, 46, 91, 28,
            63, 7, 84, 33, 57, 14, 72, 41, 96, 24,
            69, 6, 81, 36, 54, 16, 77, 44, 90, 29
        ];
        const result = bubbleSort(input);
        expect(result[0]).toBe(3);
        expect(result[result.length - 1]).toBe(96);
    });
});

describe('bubbleSortDescending', () => {
    test('sorts in descending order', () => {
        expect(bubbleSortDescending([1, 3, 2, 5, 4])).toEqual([5, 4, 3, 2, 1]);
    });

    test('handles empty array', () => {
        expect(bubbleSortDescending([])).toEqual([]);
    });

    test('handles single element', () => {
        expect(bubbleSortDescending([42])).toEqual([42]);
    });
});

describe('isSorted', () => {
    test('returns true for empty array', () => {
        expect(isSorted([])).toBe(true);
    });

    test('returns true for single element', () => {
        expect(isSorted([1])).toBe(true);
    });

    test('returns true for sorted array', () => {
        expect(isSorted([1, 2, 3, 4, 5])).toBe(true);
    });

    test('returns false for unsorted array', () => {
        expect(isSorted([1, 3, 2, 4, 5])).toBe(false);
    });
});
