import { DataProcessor } from '../data_processor';

describe('DataProcessor', () => {
    describe('findDuplicates', () => {
        test('finds duplicates in array with repeated values', () => {
            const processor = new DataProcessor([1, 2, 3, 2, 4, 3, 5]);
            expect(processor.findDuplicates().sort()).toEqual([2, 3]);
        });

        test('returns empty array when no duplicates', () => {
            const processor = new DataProcessor([1, 2, 3, 4, 5]);
            expect(processor.findDuplicates()).toEqual([]);
        });

        test('handles empty array', () => {
            const processor = new DataProcessor<number>([]);
            expect(processor.findDuplicates()).toEqual([]);
        });

        test('handles array with all same values', () => {
            const processor = new DataProcessor([5, 5, 5, 5]);
            expect(processor.findDuplicates()).toEqual([5]);
        });

        test('handles larger arrays with duplicates', () => {
            const data: number[] = [];
            for (let i = 0; i < 100; i++) {
                data.push(i % 20);
            }
            const processor = new DataProcessor(data);
            const duplicates = processor.findDuplicates();
            expect(duplicates.length).toBe(20);
        });
    });

    describe('sortData', () => {
        test('sorts numbers in ascending order', () => {
            const processor = new DataProcessor([5, 2, 8, 1, 9]);
            expect(processor.sortData()).toEqual([1, 2, 5, 8, 9]);
        });

        test('handles already sorted array', () => {
            const processor = new DataProcessor([1, 2, 3, 4, 5]);
            expect(processor.sortData()).toEqual([1, 2, 3, 4, 5]);
        });

        test('handles reverse sorted array', () => {
            const processor = new DataProcessor([5, 4, 3, 2, 1]);
            expect(processor.sortData()).toEqual([1, 2, 3, 4, 5]);
        });

        test('handles array with duplicates', () => {
            const processor = new DataProcessor([3, 1, 4, 1, 5, 9, 2, 6, 5]);
            expect(processor.sortData()).toEqual([1, 1, 2, 3, 4, 5, 5, 6, 9]);
        });

        test('handles larger arrays', () => {
            const data: number[] = [];
            for (let i = 500; i >= 0; i--) {
                data.push(i);
            }
            const processor = new DataProcessor(data);
            const sorted = processor.sortData();
            expect(sorted[0]).toBe(0);
            expect(sorted[sorted.length - 1]).toBe(500);
        });
    });

    describe('getUnique', () => {
        test('returns unique values', () => {
            const processor = new DataProcessor([1, 2, 2, 3, 3, 3, 4]);
            expect(processor.getUnique()).toEqual([1, 2, 3, 4]);
        });

        test('preserves order of first occurrence', () => {
            const processor = new DataProcessor([3, 1, 2, 1, 3, 2]);
            expect(processor.getUnique()).toEqual([3, 1, 2]);
        });

        test('handles empty array', () => {
            const processor = new DataProcessor<number>([]);
            expect(processor.getUnique()).toEqual([]);
        });

        test('handles array with all unique values', () => {
            const processor = new DataProcessor([1, 2, 3, 4, 5]);
            expect(processor.getUnique()).toEqual([1, 2, 3, 4, 5]);
        });

        test('handles strings', () => {
            const processor = new DataProcessor(['a', 'b', 'a', 'c', 'b']);
            expect(processor.getUnique()).toEqual(['a', 'b', 'c']);
        });
    });
});
