const { bubbleSort, bubbleSortDescending } = require('../bubble_sort');

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

    test('sorts an array with duplicates', () => {
        expect(bubbleSort([3, 1, 4, 1, 5, 9, 2, 6])).toEqual([1, 1, 2, 3, 4, 5, 6, 9]);
    });

    test('sorts negative numbers', () => {
        expect(bubbleSort([-3, -1, -4, -1, -5])).toEqual([-5, -4, -3, -1, -1]);
    });

    test('does not mutate original array', () => {
        const original = [3, 1, 2];
        bubbleSort(original);
        expect(original).toEqual([3, 1, 2]);
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
