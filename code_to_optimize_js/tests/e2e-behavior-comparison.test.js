/**
 * End-to-End Behavior Comparison Test
 *
 * This test verifies that:
 * 1. The instrumentation correctly captures function behavior (args + return value)
 * 2. Serialization/deserialization preserves all value types
 * 3. The comparator correctly identifies equivalent behaviors
 *
 * It simulates what happens during optimization verification:
 * - Run the same tests twice (original vs optimized) with different LOOP_INDEX
 * - Store results to different locations
 * - Compare the serialized values using the comparator
 */

const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');

// Import our modules from npm package
const { serialize, deserialize, getSerializerType, comparator } = require('codeflash');

// Test output directory
const TEST_OUTPUT_DIR = '/tmp/codeflash_e2e_test';

// Sample functions to test with various return types
const testFunctions = {
    // Primitives
    returnNumber: (x) => x * 2,
    returnString: (s) => s.toUpperCase(),
    returnBoolean: (x) => x > 0,
    returnNull: () => null,
    returnUndefined: () => undefined,

    // Special numbers
    returnNaN: () => NaN,
    returnInfinity: () => Infinity,
    returnNegInfinity: () => -Infinity,

    // Complex types
    returnArray: (arr) => arr.map(x => x * 2),
    returnObject: (obj) => ({ ...obj, processed: true }),
    returnMap: (entries) => new Map(entries),
    returnSet: (values) => new Set(values),
    returnDate: (ts) => new Date(ts),
    returnRegExp: (pattern, flags) => new RegExp(pattern, flags),

    // Nested structures
    returnNested: (data) => ({
        array: [1, 2, 3],
        map: new Map([['key', data]]),
        set: new Set([data]),
        date: new Date('2024-01-15'),
    }),

    // TypedArrays
    returnTypedArray: (data) => new Float64Array(data),

    // Error handling
    mayThrow: (shouldThrow) => {
        if (shouldThrow) throw new Error('Test error');
        return 'success';
    },
};

describe('E2E Behavior Comparison', () => {
    beforeAll(() => {
        // Clean up and create test directory
        if (fs.existsSync(TEST_OUTPUT_DIR)) {
            fs.rmSync(TEST_OUTPUT_DIR, { recursive: true });
        }
        fs.mkdirSync(TEST_OUTPUT_DIR, { recursive: true });
        console.log('Using serializer:', getSerializerType());
    });

    afterAll(() => {
        // Cleanup
        if (fs.existsSync(TEST_OUTPUT_DIR)) {
            fs.rmSync(TEST_OUTPUT_DIR, { recursive: true });
        }
    });

    describe('Direct Serialization Round-Trip', () => {
        // Test that serialize -> deserialize -> compare works for all types

        test('primitives round-trip correctly', () => {
            const testCases = [
                42,
                -3.14159,
                'hello world',
                true,
                false,
                null,
                undefined,
                BigInt('9007199254740991'),
            ];

            for (const original of testCases) {
                const serialized = serialize(original);
                const restored = deserialize(serialized);
                expect(comparator(original, restored)).toBe(true);
            }
        });

        test('special numbers round-trip correctly', () => {
            const testCases = [NaN, Infinity, -Infinity, -0];

            for (const original of testCases) {
                const serialized = serialize(original);
                const restored = deserialize(serialized);
                expect(comparator(original, restored)).toBe(true);
            }
        });

        test('complex objects round-trip correctly', () => {
            const testCases = [
                new Map([['a', 1], ['b', 2]]),
                new Set([1, 2, 3]),
                new Date('2024-01-15'),
                /test\d+/gi,
                new Error('test error'),
                new Float64Array([1.1, 2.2, 3.3]),
            ];

            for (const original of testCases) {
                const serialized = serialize(original);
                const restored = deserialize(serialized);
                expect(comparator(original, restored)).toBe(true);
            }
        });

        test('nested structures round-trip correctly', () => {
            const original = {
                array: [1, 'two', { three: 3 }],
                map: new Map([['nested', new Set([1, 2, 3])]]),
                date: new Date('2024-06-15'),
                regex: /pattern/i,
                typed: new Int32Array([10, 20, 30]),
            };

            const serialized = serialize(original);
            const restored = deserialize(serialized);
            expect(comparator(original, restored)).toBe(true);
        });
    });

    describe('Function Behavior Format', () => {
        // Test the [args, kwargs, return_value] format used by instrumentation

        test('behavior tuple format serializes correctly', () => {
            // Simulate what recordResult does: [args, {}, returnValue]
            const args = [42, 'hello'];
            const kwargs = {};  // JS doesn't have kwargs, always empty
            const returnValue = { result: 84, message: 'HELLO' };

            const behaviorTuple = [args, kwargs, returnValue];
            const serialized = serialize(behaviorTuple);
            const restored = deserialize(serialized);

            expect(comparator(behaviorTuple, restored)).toBe(true);
            expect(restored[0]).toEqual(args);
            expect(restored[1]).toEqual(kwargs);
            expect(comparator(restored[2], returnValue)).toBe(true);
        });

        test('behavior with Map return value', () => {
            const args = [['a', 1], ['b', 2]];
            const returnValue = new Map(args);
            const behaviorTuple = [args, {}, returnValue];

            const serialized = serialize(behaviorTuple);
            const restored = deserialize(serialized);

            expect(comparator(behaviorTuple, restored)).toBe(true);
            expect(restored[2] instanceof Map).toBe(true);
            expect(restored[2].get('a')).toBe(1);
        });

        test('behavior with Set return value', () => {
            const args = [[1, 2, 3]];
            const returnValue = new Set([1, 2, 3]);
            const behaviorTuple = [args, {}, returnValue];

            const serialized = serialize(behaviorTuple);
            const restored = deserialize(serialized);

            expect(comparator(behaviorTuple, restored)).toBe(true);
            expect(restored[2] instanceof Set).toBe(true);
            expect(restored[2].has(2)).toBe(true);
        });

        test('behavior with Date return value', () => {
            const args = [1705276800000];  // 2024-01-15
            const returnValue = new Date(1705276800000);
            const behaviorTuple = [args, {}, returnValue];

            const serialized = serialize(behaviorTuple);
            const restored = deserialize(serialized);

            expect(comparator(behaviorTuple, restored)).toBe(true);
            expect(restored[2] instanceof Date).toBe(true);
            expect(restored[2].getTime()).toBe(1705276800000);
        });

        test('behavior with TypedArray return value', () => {
            const args = [[1.1, 2.2, 3.3]];
            const returnValue = new Float64Array([1.1, 2.2, 3.3]);
            const behaviorTuple = [args, {}, returnValue];

            const serialized = serialize(behaviorTuple);
            const restored = deserialize(serialized);

            expect(comparator(behaviorTuple, restored)).toBe(true);
            expect(restored[2] instanceof Float64Array).toBe(true);
        });

        test('behavior with Error (exception case)', () => {
            const error = new TypeError('Invalid argument');
            const serialized = serialize(error);
            const restored = deserialize(serialized);

            expect(comparator(error, restored)).toBe(true);
            expect(restored.name).toBe('TypeError');
            expect(restored.message).toBe('Invalid argument');
        });
    });

    describe('Simulated Original vs Optimized Comparison', () => {
        // Simulate running the same function twice and comparing results

        function runAndCapture(fn, args) {
            try {
                const returnValue = fn(...args);
                return { success: true, value: [args, {}, returnValue] };
            } catch (error) {
                return { success: false, error };
            }
        }

        test('identical behaviors are equal - number function', () => {
            const fn = testFunctions.returnNumber;
            const args = [21];

            // "Original" run
            const original = runAndCapture(fn, args);
            const originalSerialized = serialize(original.value);

            // "Optimized" run (same function, simulating optimization)
            const optimized = runAndCapture(fn, args);
            const optimizedSerialized = serialize(optimized.value);

            // Deserialize and compare (what verification does)
            const originalRestored = deserialize(originalSerialized);
            const optimizedRestored = deserialize(optimizedSerialized);

            expect(comparator(originalRestored, optimizedRestored)).toBe(true);
        });

        test('identical behaviors are equal - Map function', () => {
            const fn = testFunctions.returnMap;
            const args = [[['x', 10], ['y', 20]]];

            const original = runAndCapture(fn, args);
            const originalSerialized = serialize(original.value);

            const optimized = runAndCapture(fn, args);
            const optimizedSerialized = serialize(optimized.value);

            const originalRestored = deserialize(originalSerialized);
            const optimizedRestored = deserialize(optimizedSerialized);

            expect(comparator(originalRestored, optimizedRestored)).toBe(true);
        });

        test('identical behaviors are equal - nested structure', () => {
            const fn = testFunctions.returnNested;
            const args = ['test-data'];

            const original = runAndCapture(fn, args);
            const originalSerialized = serialize(original.value);

            const optimized = runAndCapture(fn, args);
            const optimizedSerialized = serialize(optimized.value);

            const originalRestored = deserialize(originalSerialized);
            const optimizedRestored = deserialize(optimizedSerialized);

            expect(comparator(originalRestored, optimizedRestored)).toBe(true);
        });

        test('different behaviors are NOT equal', () => {
            const fn1 = (x) => x * 2;
            const fn2 = (x) => x * 3;  // Different behavior!
            const args = [10];

            const original = runAndCapture(fn1, args);
            const originalSerialized = serialize(original.value);

            const optimized = runAndCapture(fn2, args);
            const optimizedSerialized = serialize(optimized.value);

            const originalRestored = deserialize(originalSerialized);
            const optimizedRestored = deserialize(optimizedSerialized);

            // Should be FALSE - behaviors differ (20 vs 30)
            expect(comparator(originalRestored, optimizedRestored)).toBe(false);
        });

        test('floating point tolerance works', () => {
            // Simulate slight floating point differences from optimization
            const original = [[[1.0]], {}, 0.30000000000000004];
            const optimized = [[[1.0]], {}, 0.3];

            const originalSerialized = serialize(original);
            const optimizedSerialized = serialize(optimized);

            const originalRestored = deserialize(originalSerialized);
            const optimizedRestored = deserialize(optimizedSerialized);

            // Should be TRUE with default tolerance
            expect(comparator(originalRestored, optimizedRestored)).toBe(true);
        });
    });

    describe('Multiple Invocations Comparison', () => {
        // Simulate multiple test invocations being stored and compared

        test('batch of invocations can be compared', () => {
            const testCases = [
                { fn: testFunctions.returnNumber, args: [1] },
                { fn: testFunctions.returnNumber, args: [100] },
                { fn: testFunctions.returnString, args: ['hello'] },
                { fn: testFunctions.returnArray, args: [[1, 2, 3]] },
                { fn: testFunctions.returnMap, args: [[['a', 1]]] },
                { fn: testFunctions.returnSet, args: [[1, 2, 3]] },
                { fn: testFunctions.returnDate, args: [1705276800000] },
                { fn: testFunctions.returnNested, args: ['data'] },
            ];

            // Simulate original run
            const originalResults = testCases.map(({ fn, args }) => {
                const returnValue = fn(...args);
                return serialize([args, {}, returnValue]);
            });

            // Simulate optimized run (same functions)
            const optimizedResults = testCases.map(({ fn, args }) => {
                const returnValue = fn(...args);
                return serialize([args, {}, returnValue]);
            });

            // Compare all results
            for (let i = 0; i < testCases.length; i++) {
                const originalRestored = deserialize(originalResults[i]);
                const optimizedRestored = deserialize(optimizedResults[i]);

                expect(comparator(originalRestored, optimizedRestored)).toBe(true);
            }
        });
    });

    describe('File-Based Comparison (SQLite Simulation)', () => {
        // Simulate writing to files and reading back for comparison

        test('can write and read back serialized results', () => {
            const originalPath = path.join(TEST_OUTPUT_DIR, 'original.bin');
            const optimizedPath = path.join(TEST_OUTPUT_DIR, 'optimized.bin');

            // Test data
            const behaviorData = {
                args: [42, 'test', { nested: true }],
                kwargs: {},
                returnValue: {
                    result: new Map([['answer', 42]]),
                    metadata: new Set(['processed', 'validated']),
                    timestamp: new Date('2024-01-15'),
                },
            };

            const tuple = [behaviorData.args, behaviorData.kwargs, behaviorData.returnValue];

            // Write "original" result
            const originalBuffer = serialize(tuple);
            fs.writeFileSync(originalPath, originalBuffer);

            // Write "optimized" result (same data, simulating correct optimization)
            const optimizedBuffer = serialize(tuple);
            fs.writeFileSync(optimizedPath, optimizedBuffer);

            // Read back and compare
            const originalRead = fs.readFileSync(originalPath);
            const optimizedRead = fs.readFileSync(optimizedPath);

            const originalRestored = deserialize(originalRead);
            const optimizedRestored = deserialize(optimizedRead);

            expect(comparator(originalRestored, optimizedRestored)).toBe(true);

            // Verify the complex types survived
            expect(originalRestored[2].result instanceof Map).toBe(true);
            expect(originalRestored[2].metadata instanceof Set).toBe(true);
            expect(originalRestored[2].timestamp instanceof Date).toBe(true);
        });

        test('detects differences in file-based comparison', () => {
            const originalPath = path.join(TEST_OUTPUT_DIR, 'original2.bin');
            const optimizedPath = path.join(TEST_OUTPUT_DIR, 'optimized2.bin');

            // Original behavior
            const originalTuple = [[10], {}, 100];
            fs.writeFileSync(originalPath, serialize(originalTuple));

            // "Buggy" optimized behavior
            const optimizedTuple = [[10], {}, 99];  // Wrong result!
            fs.writeFileSync(optimizedPath, serialize(optimizedTuple));

            // Read back and compare
            const originalRestored = deserialize(fs.readFileSync(originalPath));
            const optimizedRestored = deserialize(fs.readFileSync(optimizedPath));

            // Should detect the difference
            expect(comparator(originalRestored, optimizedRestored)).toBe(false);
        });
    });

    describe('Edge Cases', () => {
        test('handles special values in args', () => {
            const tuple = [[NaN, Infinity, undefined, null], {}, 'processed'];

            const serialized = serialize(tuple);
            const restored = deserialize(serialized);

            expect(comparator(tuple, restored)).toBe(true);
            expect(Number.isNaN(restored[0][0])).toBe(true);
            expect(restored[0][1]).toBe(Infinity);
            expect(restored[0][2]).toBe(undefined);
            expect(restored[0][3]).toBe(null);
        });

        test('handles circular references in return value', () => {
            const obj = { value: 42 };
            obj.self = obj;  // Circular reference

            const tuple = [[], {}, obj];
            const serialized = serialize(tuple);
            const restored = deserialize(serialized);

            expect(comparator(tuple, restored)).toBe(true);
            expect(restored[2].self).toBe(restored[2]);
        });

        test('handles empty results', () => {
            const tuple = [[], {}, undefined];

            const serialized = serialize(tuple);
            const restored = deserialize(serialized);

            expect(comparator(tuple, restored)).toBe(true);
        });

        test('handles large arrays', () => {
            const largeArray = Array.from({ length: 1000 }, (_, i) => i);
            const tuple = [[largeArray], {}, largeArray.reduce((a, b) => a + b, 0)];

            const serialized = serialize(tuple);
            const restored = deserialize(serialized);

            expect(comparator(tuple, restored)).toBe(true);
        });
    });
});
