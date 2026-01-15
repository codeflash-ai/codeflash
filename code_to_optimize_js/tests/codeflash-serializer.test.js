/**
 * Extensive Cycle Tests for codeflash-serializer.js
 *
 * Tests the full cycle: serialize -> deserialize -> compare
 * Uses the codeflash-comparator to verify round-trip correctness.
 *
 * Coverage includes:
 * - All primitive types
 * - Special number values (NaN, Infinity, -Infinity)
 * - Collections (Array, Object, Map, Set)
 * - Binary data (TypedArrays, ArrayBuffer, DataView)
 * - Built-in objects (Date, RegExp, Error)
 * - Complex nested structures
 * - Circular references
 * - Edge cases
 */

const {
    serialize,
    deserialize,
    getSerializerType,
    serializeWith,
    hasV8,
    hasMsgpack,
} = require('../codeflash-serializer');

const { comparator, isClose } = require('../codeflash-comparator');

// Helper to test round-trip
function roundTrip(value, options = {}) {
    const buffer = serialize(value);
    const restored = deserialize(buffer);
    return restored;
}

// Helper to test round-trip with comparison
function testRoundTrip(value, comparisonOptions = {}) {
    const restored = roundTrip(value);
    return comparator(value, restored, comparisonOptions);
}

// ============================================================================
// SETUP AND UTILITIES
// ============================================================================

describe('Serializer Setup', () => {
    test('serializer type is detected', () => {
        const type = getSerializerType();
        expect(['v8', 'msgpack']).toContain(type);
        console.log(`Using serializer: ${type}`);
    });

    test('V8 availability', () => {
        console.log(`V8 available: ${hasV8}`);
        // Note: In Jest's VM context, V8 serialization might be detected as "broken"
        // because objects from different VM contexts don't serialize correctly.
        // So we just verify that hasV8 is a boolean, not that it's true.
        expect(typeof hasV8).toBe('boolean');
        // If V8 is available and working, the serializer type should be 'v8'
        if (hasV8) {
            expect(getSerializerType()).toBe('v8');
        }
    });

    test('msgpack availability', () => {
        console.log(`msgpack available: ${hasMsgpack}`);
        expect(hasMsgpack).toBe(true); // We installed it
    });
});

// ============================================================================
// PRIMITIVES - CYCLE TESTS
// ============================================================================

describe('Primitives Cycle Tests', () => {
    describe('null and undefined', () => {
        test('null round-trips correctly', () => {
            expect(testRoundTrip(null)).toBe(true);
        });

        test('undefined round-trips correctly', () => {
            expect(testRoundTrip(undefined)).toBe(true);
        });
    });

    describe('booleans', () => {
        test('true round-trips correctly', () => {
            expect(testRoundTrip(true)).toBe(true);
        });

        test('false round-trips correctly', () => {
            expect(testRoundTrip(false)).toBe(true);
        });
    });

    describe('numbers', () => {
        test('positive integers', () => {
            expect(testRoundTrip(0)).toBe(true);
            expect(testRoundTrip(1)).toBe(true);
            expect(testRoundTrip(42)).toBe(true);
            expect(testRoundTrip(Number.MAX_SAFE_INTEGER)).toBe(true);
        });

        test('negative integers', () => {
            expect(testRoundTrip(-1)).toBe(true);
            expect(testRoundTrip(-42)).toBe(true);
            expect(testRoundTrip(Number.MIN_SAFE_INTEGER)).toBe(true);
        });

        test('floating point numbers', () => {
            expect(testRoundTrip(3.14159)).toBe(true);
            expect(testRoundTrip(-2.71828)).toBe(true);
            expect(testRoundTrip(0.1 + 0.2)).toBe(true);  // 0.30000000000000004
        });

        test('very small numbers', () => {
            expect(testRoundTrip(Number.MIN_VALUE)).toBe(true);
            expect(testRoundTrip(Number.EPSILON)).toBe(true);
            expect(testRoundTrip(1e-300)).toBe(true);
        });

        test('very large numbers', () => {
            expect(testRoundTrip(Number.MAX_VALUE)).toBe(true);
            expect(testRoundTrip(1e300)).toBe(true);
        });

        test('negative zero', () => {
            const restored = roundTrip(-0);
            expect(Object.is(restored, -0) || restored === 0).toBe(true);
        });
    });

    describe('special number values', () => {
        test('NaN round-trips correctly', () => {
            const restored = roundTrip(NaN);
            expect(Number.isNaN(restored)).toBe(true);
        });

        test('Infinity round-trips correctly', () => {
            expect(testRoundTrip(Infinity)).toBe(true);
        });

        test('-Infinity round-trips correctly', () => {
            expect(testRoundTrip(-Infinity)).toBe(true);
        });
    });

    describe('strings', () => {
        test('empty string', () => {
            expect(testRoundTrip('')).toBe(true);
        });

        test('simple strings', () => {
            expect(testRoundTrip('hello')).toBe(true);
            expect(testRoundTrip('hello world')).toBe(true);
        });

        test('unicode strings', () => {
            expect(testRoundTrip('\u00e9')).toBe(true);  // é
            expect(testRoundTrip('\u{1F600}')).toBe(true);  // emoji
            expect(testRoundTrip('日本語')).toBe(true);
            expect(testRoundTrip('مرحبا')).toBe(true);  // Arabic
        });

        test('strings with special characters', () => {
            expect(testRoundTrip('\n\t\r')).toBe(true);
            expect(testRoundTrip('\0')).toBe(true);  // null character
            expect(testRoundTrip('\\')).toBe(true);
            expect(testRoundTrip('"')).toBe(true);
        });

        test('long strings', () => {
            expect(testRoundTrip('a'.repeat(10000))).toBe(true);
            expect(testRoundTrip('ab'.repeat(5000))).toBe(true);
        });

        test('binary-like strings', () => {
            // String with bytes 0-255
            let binaryStr = '';
            for (let i = 0; i < 256; i++) {
                binaryStr += String.fromCharCode(i);
            }
            expect(testRoundTrip(binaryStr)).toBe(true);
        });
    });

    describe('bigint', () => {
        test('small bigints', () => {
            expect(testRoundTrip(0n)).toBe(true);
            expect(testRoundTrip(1n)).toBe(true);
            expect(testRoundTrip(-1n)).toBe(true);
            expect(testRoundTrip(42n)).toBe(true);
        });

        test('large bigints', () => {
            const big = BigInt('12345678901234567890123456789012345678901234567890');
            expect(testRoundTrip(big)).toBe(true);
        });

        test('negative large bigints', () => {
            const big = BigInt('-98765432109876543210987654321098765432109876543210');
            expect(testRoundTrip(big)).toBe(true);
        });

        test('bigint at boundaries', () => {
            expect(testRoundTrip(BigInt(Number.MAX_SAFE_INTEGER))).toBe(true);
            expect(testRoundTrip(BigInt(Number.MAX_SAFE_INTEGER) + 1n)).toBe(true);
        });
    });

    describe('symbols', () => {
        test('symbol with description', () => {
            const original = Symbol('test');
            const restored = roundTrip(original);
            // Symbols can't be truly round-tripped, but description should match
            expect(typeof restored).toBe('symbol');
            expect(restored.description).toBe('test');
        });

        test('symbol without description', () => {
            const original = Symbol();
            const restored = roundTrip(original);
            expect(typeof restored).toBe('symbol');
            expect(restored.description).toBe(undefined);
        });

        test('symbol with empty description', () => {
            const original = Symbol('');
            const restored = roundTrip(original);
            expect(typeof restored).toBe('symbol');
            expect(restored.description).toBe('');
        });
    });
});

// ============================================================================
// ARRAYS - CYCLE TESTS
// ============================================================================

describe('Arrays Cycle Tests', () => {
    describe('basic arrays', () => {
        test('empty array', () => {
            expect(testRoundTrip([])).toBe(true);
        });

        test('array of numbers', () => {
            expect(testRoundTrip([1, 2, 3, 4, 5])).toBe(true);
        });

        test('array of strings', () => {
            expect(testRoundTrip(['a', 'b', 'c'])).toBe(true);
        });

        test('array of mixed primitives', () => {
            expect(testRoundTrip([1, 'two', true, null, undefined])).toBe(true);
        });

        test('array with special numbers', () => {
            const arr = [NaN, Infinity, -Infinity, 0, -0];
            const restored = roundTrip(arr);
            expect(Number.isNaN(restored[0])).toBe(true);
            expect(restored[1]).toBe(Infinity);
            expect(restored[2]).toBe(-Infinity);
            expect(restored[3]).toBe(0);
        });
    });

    describe('nested arrays', () => {
        test('2D array', () => {
            expect(testRoundTrip([[1, 2], [3, 4]])).toBe(true);
        });

        test('3D array', () => {
            expect(testRoundTrip([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])).toBe(true);
        });

        test('deeply nested array', () => {
            const deep = [[[[[[[[[[42]]]]]]]]]];
            expect(testRoundTrip(deep)).toBe(true);
        });

        test('jagged array', () => {
            expect(testRoundTrip([[1], [2, 3], [4, 5, 6]])).toBe(true);
        });
    });

    describe('sparse arrays', () => {
        test('sparse array with holes', () => {
            const sparse = [1, , , 4];  // eslint-disable-line no-sparse-arrays
            const restored = roundTrip(sparse);
            expect(restored.length).toBe(4);
            expect(restored[0]).toBe(1);
            expect(restored[3]).toBe(4);
            expect(1 in restored).toBe(false);  // hole
            expect(2 in restored).toBe(false);  // hole
        });

        test('sparse array at end', () => {
            const sparse = [1, 2, 3];
            sparse[10] = 10;
            const restored = roundTrip(sparse);
            expect(restored.length).toBe(11);
            expect(restored[10]).toBe(10);
        });
    });

    describe('large arrays', () => {
        test('array with 1000 elements', () => {
            const arr = Array.from({ length: 1000 }, (_, i) => i);
            expect(testRoundTrip(arr)).toBe(true);
        });

        test('array with 10000 elements', () => {
            const arr = Array.from({ length: 10000 }, (_, i) => i * 2);
            expect(testRoundTrip(arr)).toBe(true);
        });
    });
});

// ============================================================================
// OBJECTS - CYCLE TESTS
// ============================================================================

describe('Objects Cycle Tests', () => {
    describe('basic objects', () => {
        test('empty object', () => {
            expect(testRoundTrip({})).toBe(true);
        });

        test('simple object', () => {
            expect(testRoundTrip({ a: 1, b: 2 })).toBe(true);
        });

        test('object with mixed values', () => {
            expect(testRoundTrip({
                num: 42,
                str: 'hello',
                bool: true,
                nil: null,
                undef: undefined,
            })).toBe(true);
        });

        test('object with special numbers', () => {
            const obj = { nan: NaN, inf: Infinity, ninf: -Infinity };
            const restored = roundTrip(obj);
            expect(Number.isNaN(restored.nan)).toBe(true);
            expect(restored.inf).toBe(Infinity);
            expect(restored.ninf).toBe(-Infinity);
        });
    });

    describe('nested objects', () => {
        test('nested object', () => {
            expect(testRoundTrip({
                level1: {
                    level2: {
                        value: 42
                    }
                }
            })).toBe(true);
        });

        test('deeply nested object', () => {
            const deep = { a: { b: { c: { d: { e: { f: { g: 'deep' } } } } } } };
            expect(testRoundTrip(deep)).toBe(true);
        });

        test('object with arrays', () => {
            expect(testRoundTrip({
                arr: [1, 2, 3],
                nested: { arr: [4, 5, 6] }
            })).toBe(true);
        });
    });

    describe('objects with special keys', () => {
        test('numeric keys', () => {
            expect(testRoundTrip({ 0: 'zero', 1: 'one', 2: 'two' })).toBe(true);
        });

        test('empty string key', () => {
            expect(testRoundTrip({ '': 'empty key' })).toBe(true);
        });

        test('unicode keys', () => {
            expect(testRoundTrip({ '日本語': 'Japanese', 'émoji': '😀' })).toBe(true);
        });

        test('keys with special characters', () => {
            expect(testRoundTrip({
                'with space': 1,
                'with.dot': 2,
                'with-dash': 3,
                'with_underscore': 4,
            })).toBe(true);
        });
    });

    describe('complex objects', () => {
        test('object with bigint values', () => {
            expect(testRoundTrip({
                small: 42n,
                large: BigInt('123456789012345678901234567890')
            })).toBe(true);
        });

        test('mixed array and object nesting', () => {
            expect(testRoundTrip({
                users: [
                    { name: 'Alice', scores: [90, 85, 88] },
                    { name: 'Bob', scores: [75, 80, 82] },
                ],
                metadata: { count: 2, average: 83.3 }
            })).toBe(true);
        });
    });
});

// ============================================================================
// MAP AND SET - CYCLE TESTS
// ============================================================================

describe('Map and Set Cycle Tests', () => {
    describe('Map', () => {
        test('empty map', () => {
            expect(testRoundTrip(new Map())).toBe(true);
        });

        test('map with string keys', () => {
            const map = new Map([['a', 1], ['b', 2], ['c', 3]]);
            expect(testRoundTrip(map)).toBe(true);
        });

        test('map with number keys', () => {
            const map = new Map([[1, 'one'], [2, 'two'], [3, 'three']]);
            expect(testRoundTrip(map)).toBe(true);
        });

        test('map with mixed key types', () => {
            const map = new Map([
                ['string', 1],
                [42, 2],
                [true, 3],
                [null, 4],
            ]);
            expect(testRoundTrip(map)).toBe(true);
        });

        test('map with object values', () => {
            const map = new Map([
                ['user1', { name: 'Alice', age: 30 }],
                ['user2', { name: 'Bob', age: 25 }],
            ]);
            expect(testRoundTrip(map)).toBe(true);
        });

        test('map with nested maps', () => {
            const inner = new Map([['x', 1], ['y', 2]]);
            const outer = new Map([['inner', inner]]);
            expect(testRoundTrip(outer)).toBe(true);
        });

        test('large map', () => {
            const map = new Map();
            for (let i = 0; i < 1000; i++) {
                map.set(`key${i}`, i * 2);
            }
            expect(testRoundTrip(map)).toBe(true);
        });
    });

    describe('Set', () => {
        test('empty set', () => {
            expect(testRoundTrip(new Set())).toBe(true);
        });

        test('set of numbers', () => {
            const set = new Set([1, 2, 3, 4, 5]);
            expect(testRoundTrip(set)).toBe(true);
        });

        test('set of strings', () => {
            const set = new Set(['a', 'b', 'c']);
            expect(testRoundTrip(set)).toBe(true);
        });

        test('set of mixed primitives', () => {
            const set = new Set([1, 'two', true, null]);
            expect(testRoundTrip(set)).toBe(true);
        });

        test('set with objects', () => {
            const set = new Set([{ a: 1 }, { b: 2 }]);
            expect(testRoundTrip(set)).toBe(true);
        });

        test('set with arrays', () => {
            const set = new Set([[1, 2], [3, 4]]);
            expect(testRoundTrip(set)).toBe(true);
        });

        test('large set', () => {
            const set = new Set();
            for (let i = 0; i < 1000; i++) {
                set.add(i);
            }
            expect(testRoundTrip(set)).toBe(true);
        });
    });

    describe('nested Map and Set', () => {
        test('map containing sets', () => {
            const map = new Map([
                ['evens', new Set([2, 4, 6, 8])],
                ['odds', new Set([1, 3, 5, 7])],
            ]);
            expect(testRoundTrip(map)).toBe(true);
        });

        test('set containing maps', () => {
            const map1 = new Map([['a', 1]]);
            const map2 = new Map([['b', 2]]);
            const set = new Set([map1, map2]);
            expect(testRoundTrip(set)).toBe(true);
        });

        test('object containing map and set', () => {
            expect(testRoundTrip({
                map: new Map([['key', 'value']]),
                set: new Set([1, 2, 3]),
            })).toBe(true);
        });
    });
});

// ============================================================================
// DATE - CYCLE TESTS
// ============================================================================

describe('Date Cycle Tests', () => {
    test('current date', () => {
        const date = new Date();
        expect(testRoundTrip(date)).toBe(true);
    });

    test('specific date', () => {
        const date = new Date('2024-01-15T12:30:45.123Z');
        expect(testRoundTrip(date)).toBe(true);
    });

    test('epoch date', () => {
        const date = new Date(0);
        expect(testRoundTrip(date)).toBe(true);
    });

    test('old date', () => {
        const date = new Date('1970-01-01T00:00:00Z');
        expect(testRoundTrip(date)).toBe(true);
    });

    test('far future date', () => {
        const date = new Date('2100-12-31T23:59:59.999Z');
        expect(testRoundTrip(date)).toBe(true);
    });

    test('date before epoch', () => {
        const date = new Date('1960-01-01T00:00:00Z');
        expect(testRoundTrip(date)).toBe(true);
    });

    test('Invalid Date', () => {
        const date = new Date('invalid');
        const restored = roundTrip(date);
        expect(Number.isNaN(restored.getTime())).toBe(true);
    });

    test('date in object', () => {
        expect(testRoundTrip({
            created: new Date('2024-01-15'),
            updated: new Date('2024-06-15'),
        })).toBe(true);
    });

    test('array of dates', () => {
        const dates = [
            new Date('2024-01-01'),
            new Date('2024-06-01'),
            new Date('2024-12-01'),
        ];
        expect(testRoundTrip(dates)).toBe(true);
    });
});

// ============================================================================
// REGEXP - CYCLE TESTS
// ============================================================================

describe('RegExp Cycle Tests', () => {
    test('simple regex', () => {
        expect(testRoundTrip(/abc/)).toBe(true);
    });

    test('regex with flags', () => {
        expect(testRoundTrip(/abc/gi)).toBe(true);
        expect(testRoundTrip(/abc/m)).toBe(true);
        expect(testRoundTrip(/abc/s)).toBe(true);
        expect(testRoundTrip(/abc/u)).toBe(true);
    });

    test('regex with all flags', () => {
        expect(testRoundTrip(/abc/gimsuy)).toBe(true);
    });

    test('complex regex patterns', () => {
        expect(testRoundTrip(/^[a-z]+\d*$/i)).toBe(true);
        expect(testRoundTrip(/\d{3}-\d{3}-\d{4}/)).toBe(true);
        expect(testRoundTrip(/(?:foo|bar)+/)).toBe(true);
    });

    test('regex with special characters', () => {
        expect(testRoundTrip(/\n\t\r/)).toBe(true);
        expect(testRoundTrip(/\\/)).toBe(true);
        expect(testRoundTrip(/[.*+?^${}()|[\]\\]/)).toBe(true);
    });

    test('regex in object', () => {
        expect(testRoundTrip({
            email: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
            phone: /^\d{3}-\d{3}-\d{4}$/,
        })).toBe(true);
    });

    test('unicode regex', () => {
        expect(testRoundTrip(/\p{Emoji}/u)).toBe(true);
    });
});

// ============================================================================
// ERROR - CYCLE TESTS
// ============================================================================

describe('Error Cycle Tests', () => {
    test('basic Error', () => {
        const error = new Error('test error');
        const restored = roundTrip(error);
        expect(restored instanceof Error).toBe(true);
        expect(restored.message).toBe('test error');
    });

    test('TypeError', () => {
        const error = new TypeError('type error');
        const restored = roundTrip(error);
        expect(restored.name).toBe('TypeError');
        expect(restored.message).toBe('type error');
    });

    test('RangeError', () => {
        const error = new RangeError('range error');
        const restored = roundTrip(error);
        expect(restored.name).toBe('RangeError');
        expect(restored.message).toBe('range error');
    });

    test('SyntaxError', () => {
        const error = new SyntaxError('syntax error');
        const restored = roundTrip(error);
        expect(restored.name).toBe('SyntaxError');
        expect(restored.message).toBe('syntax error');
    });

    test('ReferenceError', () => {
        const error = new ReferenceError('reference error');
        const restored = roundTrip(error);
        expect(restored.name).toBe('ReferenceError');
        expect(restored.message).toBe('reference error');
    });

    test('error with empty message', () => {
        const error = new Error('');
        const restored = roundTrip(error);
        expect(restored.message).toBe('');
    });

    test('errors in array', () => {
        const errors = [
            new Error('error 1'),
            new TypeError('error 2'),
        ];
        const restored = roundTrip(errors);
        expect(restored[0].message).toBe('error 1');
        expect(restored[1].name).toBe('TypeError');
    });
});

// ============================================================================
// TYPED ARRAYS - CYCLE TESTS
// ============================================================================

describe('TypedArrays Cycle Tests', () => {
    describe('integer typed arrays', () => {
        test('Int8Array', () => {
            expect(testRoundTrip(new Int8Array([1, 2, -3, 127, -128]))).toBe(true);
        });

        test('Uint8Array', () => {
            expect(testRoundTrip(new Uint8Array([0, 128, 255]))).toBe(true);
        });

        test('Uint8ClampedArray', () => {
            expect(testRoundTrip(new Uint8ClampedArray([0, 128, 255]))).toBe(true);
        });

        test('Int16Array', () => {
            expect(testRoundTrip(new Int16Array([0, 1000, -1000, 32767, -32768]))).toBe(true);
        });

        test('Uint16Array', () => {
            expect(testRoundTrip(new Uint16Array([0, 32768, 65535]))).toBe(true);
        });

        test('Int32Array', () => {
            expect(testRoundTrip(new Int32Array([0, 2147483647, -2147483648]))).toBe(true);
        });

        test('Uint32Array', () => {
            expect(testRoundTrip(new Uint32Array([0, 2147483648, 4294967295]))).toBe(true);
        });
    });

    describe('float typed arrays', () => {
        test('Float32Array', () => {
            expect(testRoundTrip(new Float32Array([1.1, 2.2, 3.3]))).toBe(true);
        });

        test('Float64Array', () => {
            expect(testRoundTrip(new Float64Array([1.1, 2.2, 3.3]))).toBe(true);
        });

        test('Float32Array with special values', () => {
            const arr = new Float32Array([NaN, Infinity, -Infinity, 0, -0]);
            const restored = roundTrip(arr);
            expect(Number.isNaN(restored[0])).toBe(true);
            expect(restored[1]).toBe(Infinity);
            expect(restored[2]).toBe(-Infinity);
        });

        test('Float64Array with special values', () => {
            const arr = new Float64Array([NaN, Infinity, -Infinity]);
            const restored = roundTrip(arr);
            expect(Number.isNaN(restored[0])).toBe(true);
            expect(restored[1]).toBe(Infinity);
            expect(restored[2]).toBe(-Infinity);
        });
    });

    describe('bigint typed arrays', () => {
        test('BigInt64Array', () => {
            expect(testRoundTrip(new BigInt64Array([0n, 1n, -1n, 9223372036854775807n]))).toBe(true);
        });

        test('BigUint64Array', () => {
            expect(testRoundTrip(new BigUint64Array([0n, 1n, 18446744073709551615n]))).toBe(true);
        });
    });

    describe('large typed arrays', () => {
        test('large Uint8Array', () => {
            const arr = new Uint8Array(10000);
            for (let i = 0; i < arr.length; i++) {
                arr[i] = i % 256;
            }
            expect(testRoundTrip(arr)).toBe(true);
        });

        test('large Float64Array', () => {
            const arr = new Float64Array(1000);
            for (let i = 0; i < arr.length; i++) {
                arr[i] = Math.random();
            }
            expect(testRoundTrip(arr)).toBe(true);
        });
    });

    describe('empty typed arrays', () => {
        test('empty Int8Array', () => {
            expect(testRoundTrip(new Int8Array())).toBe(true);
        });

        test('empty Float64Array', () => {
            expect(testRoundTrip(new Float64Array())).toBe(true);
        });
    });

    describe('typed arrays in objects', () => {
        test('object with multiple typed arrays', () => {
            expect(testRoundTrip({
                bytes: new Uint8Array([1, 2, 3]),
                floats: new Float64Array([1.1, 2.2, 3.3]),
                ints: new Int32Array([-1, 0, 1]),
            })).toBe(true);
        });
    });
});

// ============================================================================
// ARRAYBUFFER AND DATAVIEW - CYCLE TESTS
// ============================================================================

describe('ArrayBuffer and DataView Cycle Tests', () => {
    describe('ArrayBuffer', () => {
        test('empty ArrayBuffer', () => {
            const buf = new ArrayBuffer(0);
            const restored = roundTrip(buf);
            expect(restored.byteLength).toBe(0);
        });

        test('ArrayBuffer with data', () => {
            const buf = new ArrayBuffer(4);
            new Uint8Array(buf).set([1, 2, 3, 4]);
            const restored = roundTrip(buf);
            expect(new Uint8Array(restored)).toEqual(new Uint8Array([1, 2, 3, 4]));
        });

        test('large ArrayBuffer', () => {
            const buf = new ArrayBuffer(1000);
            const view = new Uint8Array(buf);
            for (let i = 0; i < view.length; i++) {
                view[i] = i % 256;
            }
            const restored = roundTrip(buf);
            expect(new Uint8Array(restored)).toEqual(view);
        });
    });

    describe('DataView', () => {
        test('DataView with data', () => {
            const buf = new ArrayBuffer(8);
            const view = new DataView(buf);
            view.setFloat64(0, 3.14159, true);
            const restored = roundTrip(view);
            expect(restored.byteLength).toBe(8);
            expect(isClose(restored.getFloat64(0, true), 3.14159)).toBe(true);
        });

        test('DataView with mixed data', () => {
            const buf = new ArrayBuffer(12);
            const view = new DataView(buf);
            view.setInt32(0, 42, true);
            view.setFloat64(4, 3.14, true);
            const restored = roundTrip(view);
            expect(restored.getInt32(0, true)).toBe(42);
            expect(isClose(restored.getFloat64(4, true), 3.14)).toBe(true);
        });
    });
});

// ============================================================================
// CIRCULAR REFERENCES - CYCLE TESTS
// ============================================================================

describe('Circular References Cycle Tests', () => {
    test('self-referencing object', () => {
        const obj = { value: 42 };
        obj.self = obj;
        const restored = roundTrip(obj);
        expect(restored.value).toBe(42);
        expect(restored.self).toBe(restored);
    });

    test('self-referencing array', () => {
        const arr = [1, 2, 3];
        arr.push(arr);
        const restored = roundTrip(arr);
        expect(restored[0]).toBe(1);
        expect(restored[3]).toBe(restored);
    });

    test('mutually referencing objects', () => {
        const a = { name: 'a' };
        const b = { name: 'b' };
        a.ref = b;
        b.ref = a;
        const restored = roundTrip(a);
        expect(restored.name).toBe('a');
        expect(restored.ref.name).toBe('b');
        expect(restored.ref.ref).toBe(restored);
    });

    test('deep circular reference', () => {
        const obj = {
            level1: {
                level2: {
                    level3: {}
                }
            }
        };
        obj.level1.level2.level3.back = obj;
        const restored = roundTrip(obj);
        expect(restored.level1.level2.level3.back).toBe(restored);
    });

    test('circular reference in Map', () => {
        const map = new Map();
        map.set('self', map);
        const restored = roundTrip(map);
        expect(restored.get('self')).toBe(restored);
    });

    test('circular reference in Set', () => {
        const set = new Set();
        const obj = { set };
        set.add(obj);
        const restored = roundTrip(set);
        const [first] = restored;
        expect(first.set).toBe(restored);
    });

    test('shared reference (diamond pattern)', () => {
        const shared = { value: 'shared' };
        const obj = {
            a: { ref: shared },
            b: { ref: shared },
        };
        const restored = roundTrip(obj);
        expect(restored.a.ref).toBe(restored.b.ref);
    });
});

// ============================================================================
// FUNCTIONS - CYCLE TESTS
// ============================================================================

describe('Functions Cycle Tests', () => {
    test('named function', () => {
        function myFunction() { return 42; }
        const restored = roundTrip(myFunction);
        expect(typeof restored).toBe('function');
        expect(restored.name).toBe('myFunction');
    });

    test('anonymous function', () => {
        const fn = function() { return 42; };
        const restored = roundTrip(fn);
        expect(typeof restored).toBe('function');
    });

    test('arrow function', () => {
        const fn = () => 42;
        const restored = roundTrip(fn);
        expect(typeof restored).toBe('function');
    });

    test('object with function', () => {
        const obj = {
            value: 42,
            method: function myMethod() { return this.value; }
        };
        const restored = roundTrip(obj);
        expect(restored.value).toBe(42);
        expect(typeof restored.method).toBe('function');
        expect(restored.method.name).toBe('myMethod');
    });

    test('array with functions', () => {
        const arr = [1, function fn() {}, 3];
        const restored = roundTrip(arr);
        expect(restored[0]).toBe(1);
        expect(typeof restored[1]).toBe('function');
        expect(restored[2]).toBe(3);
    });
});

// ============================================================================
// EDGE CASES - CYCLE TESTS
// ============================================================================

describe('Edge Cases Cycle Tests', () => {
    describe('empty values', () => {
        test('empty object', () => {
            expect(testRoundTrip({})).toBe(true);
        });

        test('empty array', () => {
            expect(testRoundTrip([])).toBe(true);
        });

        test('empty string', () => {
            expect(testRoundTrip('')).toBe(true);
        });

        test('empty Map', () => {
            expect(testRoundTrip(new Map())).toBe(true);
        });

        test('empty Set', () => {
            expect(testRoundTrip(new Set())).toBe(true);
        });
    });

    describe('deeply nested structures', () => {
        // Note: msgpack uses recursion which can hit stack limits on very deep structures.
        // Our marker-based approach adds additional nesting levels (wrapper objects).
        // These tests use conservative depths that work with both V8 and msgpack.

        test('20 levels deep object', () => {
            let deep = { value: 'bottom' };
            for (let i = 0; i < 20; i++) {
                deep = { nested: deep };
            }
            expect(testRoundTrip(deep)).toBe(true);
        });

        test('20 levels deep array', () => {
            let deep = [42];
            for (let i = 0; i < 20; i++) {
                deep = [deep];
            }
            expect(testRoundTrip(deep)).toBe(true);
        });

        test('mixed nesting 15 levels deep', () => {
            let deep = { value: 42 };
            for (let i = 0; i < 15; i++) {
                if (i % 2 === 0) {
                    deep = { nested: deep };
                } else {
                    deep = [deep];
                }
            }
            expect(testRoundTrip(deep)).toBe(true);
        });
    });

    describe('objects with prototype chain', () => {
        test('class instance', () => {
            class Point {
                constructor(x, y) {
                    this.x = x;
                    this.y = y;
                }
            }
            const point = new Point(3, 4);
            const restored = roundTrip(point);
            // Prototype is lost, but data is preserved
            expect(restored.x).toBe(3);
            expect(restored.y).toBe(4);
        });

        test('object with null prototype', () => {
            const obj = Object.create(null);
            obj.foo = 'bar';
            const restored = roundTrip(obj);
            expect(restored.foo).toBe('bar');
        });
    });

    describe('large structures', () => {
        test('object with 1000 keys', () => {
            const obj = {};
            for (let i = 0; i < 1000; i++) {
                obj[`key${i}`] = i;
            }
            expect(testRoundTrip(obj)).toBe(true);
        });

        test('array with nested objects', () => {
            const arr = Array.from({ length: 100 }, (_, i) => ({
                id: i,
                data: { nested: { value: i * 2 } },
                tags: [`tag${i}`, `tag${i + 1}`],
            }));
            expect(testRoundTrip(arr)).toBe(true);
        });
    });

    describe('mixed complex structures', () => {
        test('complex nested structure', () => {
            const complex = {
                users: new Map([
                    ['alice', { name: 'Alice', scores: new Set([90, 85, 88]) }],
                    ['bob', { name: 'Bob', scores: new Set([75, 80, 82]) }],
                ]),
                metadata: {
                    created: new Date('2024-01-15'),
                    pattern: /user-\d+/,
                    counts: new Int32Array([10, 20, 30]),
                },
                config: {
                    enabled: true,
                    threshold: 0.5,
                    tags: ['production', 'v2'],
                },
            };
            expect(testRoundTrip(complex)).toBe(true);
        });
    });
});

// ============================================================================
// CROSS-SERIALIZER TESTS (if both available)
// ============================================================================

describe('Cross-Serializer Tests', () => {
    // Only run if both serializers are available
    const skipIfNoMsgpack = !hasMsgpack ? test.skip : test;
    const skipIfNoV8 = !hasV8 ? test.skip : test;

    describe('msgpack specific tests', () => {
        skipIfNoMsgpack('primitives via msgpack', () => {
            if (!serializeWith.msgpack) return;

            const values = [null, undefined, true, false, 42, 'hello', 3.14];
            for (const value of values) {
                const buffer = serializeWith.msgpack(value);
                const restored = deserialize(buffer);
                expect(comparator(value, restored)).toBe(true);
            }
        });

        skipIfNoMsgpack('special numbers via msgpack', () => {
            if (!serializeWith.msgpack) return;

            const nanBuffer = serializeWith.msgpack(NaN);
            expect(Number.isNaN(deserialize(nanBuffer))).toBe(true);

            const infBuffer = serializeWith.msgpack(Infinity);
            expect(deserialize(infBuffer)).toBe(Infinity);

            const ninfBuffer = serializeWith.msgpack(-Infinity);
            expect(deserialize(ninfBuffer)).toBe(-Infinity);
        });

        skipIfNoMsgpack('complex objects via msgpack', () => {
            if (!serializeWith.msgpack) return;

            const obj = {
                map: new Map([['a', 1]]),
                set: new Set([1, 2, 3]),
                date: new Date('2024-01-15'),
                regex: /test/gi,
            };
            const buffer = serializeWith.msgpack(obj);
            const restored = deserialize(buffer);
            expect(comparator(obj, restored)).toBe(true);
        });
    });

    describe('V8 specific tests', () => {
        skipIfNoV8('primitives via V8', () => {
            if (!serializeWith.v8) return;

            const values = [null, undefined, true, false, 42, 'hello', 3.14];
            for (const value of values) {
                const buffer = serializeWith.v8(value);
                const restored = deserialize(buffer);
                expect(comparator(value, restored)).toBe(true);
            }
        });

        skipIfNoV8('special numbers via V8', () => {
            if (!serializeWith.v8) return;

            const nanBuffer = serializeWith.v8(NaN);
            expect(Number.isNaN(deserialize(nanBuffer))).toBe(true);

            const infBuffer = serializeWith.v8(Infinity);
            expect(deserialize(infBuffer)).toBe(Infinity);
        });

        skipIfNoV8('circular references via V8', () => {
            if (!serializeWith.v8) return;

            const obj = { value: 42 };
            obj.self = obj;
            const buffer = serializeWith.v8(obj);
            const restored = deserialize(buffer);
            expect(restored.self).toBe(restored);
        });
    });
});

// ============================================================================
// REAL-WORLD SCENARIOS - CYCLE TESTS
// ============================================================================

describe('Real-World Scenarios Cycle Tests', () => {
    test('API response structure', () => {
        const response = {
            status: 200,
            data: {
                users: [
                    { id: 1, name: 'Alice', email: 'alice@example.com', createdAt: new Date('2024-01-01') },
                    { id: 2, name: 'Bob', email: 'bob@example.com', createdAt: new Date('2024-02-01') },
                ],
                pagination: {
                    page: 1,
                    pageSize: 10,
                    total: 100,
                },
            },
            meta: {
                requestId: 'abc123',
                duration: 45.67,
            },
        };
        expect(testRoundTrip(response)).toBe(true);
    });

    test('configuration object', () => {
        const config = {
            database: {
                host: 'localhost',
                port: 5432,
                credentials: {
                    username: 'admin',
                    password: 'secret',
                },
            },
            features: new Set(['feature-a', 'feature-b']),
            limits: new Map([
                ['requests', 1000],
                ['connections', 100],
            ]),
            patterns: {
                email: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
                phone: /^\d{3}-\d{3}-\d{4}$/,
            },
        };
        expect(testRoundTrip(config)).toBe(true);
    });

    test('binary data processing result', () => {
        const result = {
            input: new Uint8Array([0x48, 0x65, 0x6c, 0x6c, 0x6f]),
            output: new Float32Array([1.0, 2.0, 3.0, 4.0]),
            stats: {
                min: 1.0,
                max: 4.0,
                mean: 2.5,
                variance: 1.25,
            },
            histogram: new Int32Array([10, 20, 30, 25, 15]),
        };
        expect(testRoundTrip(result)).toBe(true);
    });

    test('error with context', () => {
        const errorReport = {
            error: new TypeError('Cannot read property of undefined'),
            context: {
                file: 'app.js',
                line: 42,
                column: 10,
            },
            timestamp: new Date(),
            metadata: new Map([
                ['userId', 'user123'],
                ['sessionId', 'session456'],
            ]),
        };
        const restored = roundTrip(errorReport);
        expect(restored.error.name).toBe('TypeError');
        expect(restored.context.file).toBe('app.js');
    });

    test('function test results (codeflash use case)', () => {
        // This simulates what codeflash stores: [args, kwargs, return_value]
        const testResult = [
            ['hello', 'world'],  // args
            {},                   // kwargs (empty in JS)
            'helloworld',        // return value
        ];
        expect(testRoundTrip(testResult)).toBe(true);
    });

    test('function test with complex return value', () => {
        const testResult = [
            [{ data: [1, 2, 3] }, { options: { sort: true } }],  // args
            {},
            {
                result: [3, 2, 1],
                stats: { count: 3, sum: 6 },
                metadata: new Map([['processed', true]]),
            },
        ];
        expect(testRoundTrip(testResult)).toBe(true);
    });
});
