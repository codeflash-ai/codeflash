/**
 * Tests for async utility functions - ES Module
 */
import { delay, processItemsSequential, asyncMap, asyncFilter } from '../async_utils.js';

describe('processItemsSequential', () => {
    test('processes all items', async () => {
        const items = [1, 2, 3, 4, 5];
        const processor = async (x) => x * 2;
        const results = await processItemsSequential(items, processor);
        expect(results).toEqual([2, 4, 6, 8, 10]);
    });

    test('handles empty array', async () => {
        const results = await processItemsSequential([], async (x) => x);
        expect(results).toEqual([]);
    });

    test('handles async operations with delays', async () => {
        const items = [1, 2, 3];
        const processor = async (x) => {
            await delay(1);
            return x + 10;
        };
        const results = await processItemsSequential(items, processor);
        expect(results).toEqual([11, 12, 13]);
    });

    test('preserves order', async () => {
        const items = [5, 4, 3, 2, 1];
        const processor = async (x) => x.toString();
        const results = await processItemsSequential(items, processor);
        expect(results).toEqual(['5', '4', '3', '2', '1']);
    });

    test('handles larger arrays', async () => {
        const items = Array.from({ length: 20 }, (_, i) => i);
        const processor = async (x) => x * 2;
        const results = await processItemsSequential(items, processor);
        expect(results.length).toBe(20);
        expect(results[0]).toBe(0);
        expect(results[19]).toBe(38);
    });
});

describe('asyncMap', () => {
    test('maps all items', async () => {
        const items = [1, 2, 3];
        const mapper = async (x) => x * 10;
        const results = await asyncMap(items, mapper);
        expect(results).toEqual([10, 20, 30]);
    });

    test('handles empty array', async () => {
        const results = await asyncMap([], async (x) => x);
        expect(results).toEqual([]);
    });

    test('handles objects', async () => {
        const items = [{ a: 1 }, { a: 2 }];
        const mapper = async (obj) => ({ ...obj, b: obj.a * 2 });
        const results = await asyncMap(items, mapper);
        expect(results).toEqual([{ a: 1, b: 2 }, { a: 2, b: 4 }]);
    });
});

describe('asyncFilter', () => {
    test('filters items based on predicate', async () => {
        const items = [1, 2, 3, 4, 5, 6];
        const predicate = async (x) => x % 2 === 0;
        const results = await asyncFilter(items, predicate);
        expect(results).toEqual([2, 4, 6]);
    });

    test('handles empty array', async () => {
        const results = await asyncFilter([], async () => true);
        expect(results).toEqual([]);
    });

    test('handles all items filtered out', async () => {
        const items = [1, 2, 3];
        const results = await asyncFilter(items, async () => false);
        expect(results).toEqual([]);
    });
});
