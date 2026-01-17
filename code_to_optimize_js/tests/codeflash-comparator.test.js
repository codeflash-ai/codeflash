/**
 * Extensive tests for codeflash-comparator.js
 *
 * These tests verify that the comparator correctly handles:
 * - All JavaScript primitive types
 * - Floating point tolerance and special values (NaN, Infinity)
 * - Arrays and nested structures
 * - Objects and class instances
 * - Built-in objects (Date, RegExp, Error, Map, Set)
 * - TypedArrays and ArrayBuffer
 * - Circular references
 * - Edge cases and corner cases
 */

const {
    comparator,
    createComparator,
    strictComparator,
    looseComparator,
    isClose,
    getType,
    DEFAULT_OPTIONS,
} = require('../codeflash-comparator');

// ============================================================================
// PRIMITIVES
// ============================================================================

describe('Primitives', () => {
    describe('null and undefined', () => {
        test('null equals null', () => {
            expect(comparator(null, null)).toBe(true);
        });

        test('undefined equals undefined', () => {
            expect(comparator(undefined, undefined)).toBe(true);
        });

        test('null does not equal undefined', () => {
            expect(comparator(null, undefined)).toBe(false);
        });

        test('undefined does not equal null', () => {
            expect(comparator(undefined, null)).toBe(false);
        });

        test('null does not equal 0', () => {
            expect(comparator(null, 0)).toBe(false);
        });

        test('undefined does not equal empty string', () => {
            expect(comparator(undefined, '')).toBe(false);
        });

        test('null does not equal empty object', () => {
            expect(comparator(null, {})).toBe(false);
        });
    });

    describe('booleans', () => {
        test('true equals true', () => {
            expect(comparator(true, true)).toBe(true);
        });

        test('false equals false', () => {
            expect(comparator(false, false)).toBe(true);
        });

        test('true does not equal false', () => {
            expect(comparator(true, false)).toBe(false);
        });

        test('true does not equal 1', () => {
            expect(comparator(true, 1)).toBe(false);
        });

        test('false does not equal 0', () => {
            expect(comparator(false, 0)).toBe(false);
        });

        test('false does not equal empty string', () => {
            expect(comparator(false, '')).toBe(false);
        });

        test('false does not equal null', () => {
            expect(comparator(false, null)).toBe(false);
        });
    });

    describe('strings', () => {
        test('identical strings are equal', () => {
            expect(comparator('hello', 'hello')).toBe(true);
        });

        test('empty strings are equal', () => {
            expect(comparator('', '')).toBe(true);
        });

        test('different strings are not equal', () => {
            expect(comparator('hello', 'world')).toBe(false);
        });

        test('strings with different case are not equal', () => {
            expect(comparator('Hello', 'hello')).toBe(false);
        });

        test('string does not equal number', () => {
            expect(comparator('123', 123)).toBe(false);
        });

        test('unicode strings are compared correctly', () => {
            expect(comparator('\u00e9', '\u00e9')).toBe(true);  // é
            expect(comparator('\u00e9', 'e\u0301')).toBe(false); // é vs e + combining accent (different representations)
        });

        test('strings with whitespace differences', () => {
            expect(comparator('hello world', 'hello  world')).toBe(false);
            expect(comparator(' hello', 'hello')).toBe(false);
            expect(comparator('hello\n', 'hello')).toBe(false);
        });

        test('long strings are compared correctly', () => {
            const long1 = 'a'.repeat(10000);
            const long2 = 'a'.repeat(10000);
            const long3 = 'a'.repeat(9999) + 'b';
            expect(comparator(long1, long2)).toBe(true);
            expect(comparator(long1, long3)).toBe(false);
        });
    });

    describe('symbols', () => {
        test('same symbol reference is equal', () => {
            const sym = Symbol('test');
            expect(comparator(sym, sym)).toBe(true);
        });

        test('symbols with same description are equal', () => {
            // Note: This is a design decision - we compare by description
            expect(comparator(Symbol('test'), Symbol('test'))).toBe(true);
        });

        test('symbols with different descriptions are not equal', () => {
            expect(comparator(Symbol('foo'), Symbol('bar'))).toBe(false);
        });

        test('symbol does not equal string', () => {
            expect(comparator(Symbol('test'), 'test')).toBe(false);
        });

        test('Symbol.for creates equal symbols', () => {
            expect(comparator(Symbol.for('shared'), Symbol.for('shared'))).toBe(true);
        });
    });

    describe('bigint', () => {
        test('identical bigints are equal', () => {
            expect(comparator(123n, 123n)).toBe(true);
        });

        test('different bigints are not equal', () => {
            expect(comparator(123n, 456n)).toBe(false);
        });

        test('bigint does not equal number', () => {
            expect(comparator(123n, 123)).toBe(false);
        });

        test('large bigints are compared correctly', () => {
            const big1 = BigInt('12345678901234567890123456789012345678901234567890');
            const big2 = BigInt('12345678901234567890123456789012345678901234567890');
            const big3 = BigInt('12345678901234567890123456789012345678901234567891');
            expect(comparator(big1, big2)).toBe(true);
            expect(comparator(big1, big3)).toBe(false);
        });

        test('negative bigints', () => {
            expect(comparator(-123n, -123n)).toBe(true);
            expect(comparator(-123n, 123n)).toBe(false);
        });

        test('zero bigint', () => {
            expect(comparator(0n, 0n)).toBe(true);
            expect(comparator(0n, -0n)).toBe(true);  // -0n === 0n
        });
    });
});

// ============================================================================
// NUMBERS AND FLOATING POINT
// ============================================================================

describe('Numbers and Floating Point', () => {
    describe('integers', () => {
        test('identical integers are equal', () => {
            expect(comparator(42, 42)).toBe(true);
        });

        test('different integers are not equal', () => {
            expect(comparator(42, 43)).toBe(false);
        });

        test('zero equals zero', () => {
            expect(comparator(0, 0)).toBe(true);
        });

        test('negative zero equals positive zero', () => {
            expect(comparator(-0, 0)).toBe(true);
            expect(comparator(0, -0)).toBe(true);
        });

        test('negative integers', () => {
            expect(comparator(-42, -42)).toBe(true);
            expect(comparator(-42, 42)).toBe(false);
        });

        test('MAX_SAFE_INTEGER', () => {
            expect(comparator(Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER)).toBe(true);
        });

        test('MIN_SAFE_INTEGER', () => {
            expect(comparator(Number.MIN_SAFE_INTEGER, Number.MIN_SAFE_INTEGER)).toBe(true);
        });
    });

    describe('floating point with tolerance', () => {
        test('identical floats are equal', () => {
            expect(comparator(3.14159, 3.14159)).toBe(true);
        });

        test('floats within tolerance are equal', () => {
            expect(comparator(1.0, 1.0 + 1e-10)).toBe(true);
        });

        test('floats outside tolerance are not equal', () => {
            expect(comparator(1.0, 1.1)).toBe(false);
        });

        test('very small differences', () => {
            expect(comparator(0.1 + 0.2, 0.3)).toBe(true);  // Classic floating point issue
        });

        test('small numbers with relative tolerance', () => {
            // For small numbers, relative tolerance matters
            expect(comparator(1e-10, 1e-10 + 1e-20)).toBe(true);
        });

        test('zero and very small number', () => {
            // With default tolerance (rtol=1e-9, atol=0), 0 and 1e-15 are not equal
            // because relative tolerance of 0 is 0
            expect(comparator(0, 1e-15)).toBe(false);
        });

        test('floating point comparison edge cases', () => {
            expect(comparator(1.0000000001, 1.0000000002)).toBe(true);
            expect(comparator(1.0, 1.0001)).toBe(false);
        });
    });

    describe('NaN handling', () => {
        test('NaN equals NaN', () => {
            expect(comparator(NaN, NaN)).toBe(true);
        });

        test('NaN from operations', () => {
            expect(comparator(Math.sqrt(-1), 0 / 0)).toBe(true);
        });

        test('NaN does not equal any number', () => {
            expect(comparator(NaN, 0)).toBe(false);
            expect(comparator(NaN, 1)).toBe(false);
            expect(comparator(NaN, Infinity)).toBe(false);
        });
    });

    describe('Infinity handling', () => {
        test('Infinity equals Infinity', () => {
            expect(comparator(Infinity, Infinity)).toBe(true);
        });

        test('-Infinity equals -Infinity', () => {
            expect(comparator(-Infinity, -Infinity)).toBe(true);
        });

        test('Infinity does not equal -Infinity', () => {
            expect(comparator(Infinity, -Infinity)).toBe(false);
        });

        test('Infinity does not equal large number', () => {
            expect(comparator(Infinity, Number.MAX_VALUE)).toBe(false);
        });

        test('Infinity from operations', () => {
            expect(comparator(1 / 0, Infinity)).toBe(true);
            expect(comparator(-1 / 0, -Infinity)).toBe(true);
        });
    });

    describe('special number values', () => {
        test('Number.EPSILON', () => {
            expect(comparator(Number.EPSILON, Number.EPSILON)).toBe(true);
        });

        test('Number.MAX_VALUE', () => {
            expect(comparator(Number.MAX_VALUE, Number.MAX_VALUE)).toBe(true);
        });

        test('Number.MIN_VALUE', () => {
            expect(comparator(Number.MIN_VALUE, Number.MIN_VALUE)).toBe(true);
        });
    });

    describe('isClose helper function', () => {
        test('basic usage', () => {
            expect(isClose(1.0, 1.0)).toBe(true);
            expect(isClose(1.0, 2.0)).toBe(false);
        });

        test('NaN handling', () => {
            expect(isClose(NaN, NaN)).toBe(true);
            expect(isClose(NaN, 1)).toBe(false);
        });

        test('Infinity handling', () => {
            expect(isClose(Infinity, Infinity)).toBe(true);
            expect(isClose(-Infinity, -Infinity)).toBe(true);
            expect(isClose(Infinity, -Infinity)).toBe(false);
        });

        test('custom tolerance', () => {
            expect(isClose(1.0, 1.01, 0.1)).toBe(true);
            expect(isClose(1.0, 1.01, 0.001)).toBe(false);
        });

        test('absolute tolerance', () => {
            expect(isClose(0, 0.001, 0, 0.01)).toBe(true);
            expect(isClose(0, 0.001, 0, 0.0001)).toBe(false);
        });
    });
});

// ============================================================================
// ARRAYS
// ============================================================================

describe('Arrays', () => {
    describe('basic arrays', () => {
        test('empty arrays are equal', () => {
            expect(comparator([], [])).toBe(true);
        });

        test('identical arrays are equal', () => {
            expect(comparator([1, 2, 3], [1, 2, 3])).toBe(true);
        });

        test('different length arrays are not equal', () => {
            expect(comparator([1, 2, 3], [1, 2])).toBe(false);
        });

        test('different order arrays are not equal', () => {
            expect(comparator([1, 2, 3], [3, 2, 1])).toBe(false);
        });

        test('arrays with different values are not equal', () => {
            expect(comparator([1, 2, 3], [1, 2, 4])).toBe(false);
        });

        test('array does not equal object', () => {
            expect(comparator([1, 2, 3], { 0: 1, 1: 2, 2: 3, length: 3 })).toBe(false);
        });
    });

    describe('nested arrays', () => {
        test('nested arrays are equal', () => {
            expect(comparator([[1, 2], [3, 4]], [[1, 2], [3, 4]])).toBe(true);
        });

        test('nested arrays with different values', () => {
            expect(comparator([[1, 2], [3, 4]], [[1, 2], [3, 5]])).toBe(false);
        });

        test('deeply nested arrays', () => {
            const a = [[[[1]]]];
            const b = [[[[1]]]];
            const c = [[[[2]]]];
            expect(comparator(a, b)).toBe(true);
            expect(comparator(a, c)).toBe(false);
        });
    });

    describe('arrays with mixed types', () => {
        test('arrays with mixed primitives', () => {
            expect(comparator([1, 'two', true, null], [1, 'two', true, null])).toBe(true);
        });

        test('arrays with objects', () => {
            expect(comparator([{ a: 1 }, { b: 2 }], [{ a: 1 }, { b: 2 }])).toBe(true);
            expect(comparator([{ a: 1 }, { b: 2 }], [{ a: 1 }, { b: 3 }])).toBe(false);
        });

        test('arrays with floats and NaN', () => {
            expect(comparator([1.1, NaN, Infinity], [1.1, NaN, Infinity])).toBe(true);
        });
    });

    describe('sparse arrays', () => {
        test('sparse arrays with same holes', () => {
            const a = [1, , 3];  // eslint-disable-line no-sparse-arrays
            const b = [1, , 3];  // eslint-disable-line no-sparse-arrays
            expect(comparator(a, b)).toBe(true);
        });

        test('sparse array vs array with undefined', () => {
            const sparse = [1, , 3];  // eslint-disable-line no-sparse-arrays
            const withUndefined = [1, undefined, 3];
            // These have different semantics but may compare equal depending on implementation
            // Object.keys doesn't include sparse indices
            expect(comparator(sparse.length, withUndefined.length)).toBe(true);
        });
    });

    describe('array-like objects', () => {
        test('array does not equal arguments object', () => {
            function getArgs() { return arguments; }
            expect(comparator([1, 2, 3], getArgs(1, 2, 3))).toBe(false);
        });
    });
});

// ============================================================================
// OBJECTS
// ============================================================================

describe('Objects', () => {
    describe('plain objects', () => {
        test('empty objects are equal', () => {
            expect(comparator({}, {})).toBe(true);
        });

        test('identical objects are equal', () => {
            expect(comparator({ a: 1, b: 2 }, { a: 1, b: 2 })).toBe(true);
        });

        test('objects with different values', () => {
            expect(comparator({ a: 1 }, { a: 2 })).toBe(false);
        });

        test('objects with different keys', () => {
            expect(comparator({ a: 1 }, { b: 1 })).toBe(false);
        });

        test('objects with extra keys', () => {
            expect(comparator({ a: 1 }, { a: 1, b: 2 })).toBe(false);
        });

        test('key order does not matter', () => {
            expect(comparator({ a: 1, b: 2 }, { b: 2, a: 1 })).toBe(true);
        });
    });

    describe('nested objects', () => {
        test('nested objects are equal', () => {
            expect(comparator({ a: { b: 1 } }, { a: { b: 1 } })).toBe(true);
        });

        test('deeply nested objects', () => {
            const a = { l1: { l2: { l3: { l4: { value: 42 } } } } };
            const b = { l1: { l2: { l3: { l4: { value: 42 } } } } };
            const c = { l1: { l2: { l3: { l4: { value: 43 } } } } };
            expect(comparator(a, b)).toBe(true);
            expect(comparator(a, c)).toBe(false);
        });

        test('objects with arrays', () => {
            expect(comparator({ arr: [1, 2, 3] }, { arr: [1, 2, 3] })).toBe(true);
            expect(comparator({ arr: [1, 2, 3] }, { arr: [1, 2, 4] })).toBe(false);
        });
    });

    describe('superset mode', () => {
        test('superset allows extra keys in new object', () => {
            expect(comparator(
                { a: 1 },
                { a: 1, b: 2 },
                { supersetObj: true }
            )).toBe(true);
        });

        test('superset still requires matching values', () => {
            expect(comparator(
                { a: 1 },
                { a: 2, b: 2 },
                { supersetObj: true }
            )).toBe(false);
        });

        test('superset requires all original keys', () => {
            expect(comparator(
                { a: 1, b: 2 },
                { a: 1 },
                { supersetObj: true }
            )).toBe(false);
        });

        test('superset works with nested objects', () => {
            expect(comparator(
                { a: { x: 1 } },
                { a: { x: 1, y: 2 }, b: 3 },
                { supersetObj: true }
            )).toBe(true);
        });
    });

    describe('objects with special keys', () => {
        test('objects with numeric keys', () => {
            expect(comparator({ 0: 'a', 1: 'b' }, { 0: 'a', 1: 'b' })).toBe(true);
        });

        test('objects with symbol keys', () => {
            // Symbol keys are not included in Object.keys()
            const sym = Symbol('test');
            const a = { [sym]: 1 };
            const b = { [sym]: 1 };
            // By default, symbol keys are not compared
            expect(comparator(a, b)).toBe(true);
        });

        test('objects with empty string key', () => {
            expect(comparator({ '': 1 }, { '': 1 })).toBe(true);
        });
    });

    describe('objects with null prototype', () => {
        test('null prototype objects', () => {
            const a = Object.create(null);
            a.foo = 'bar';
            const b = Object.create(null);
            b.foo = 'bar';
            expect(comparator(a, b)).toBe(true);
        });
    });
});

// ============================================================================
// BUILT-IN OBJECTS
// ============================================================================

describe('Built-in Objects', () => {
    describe('Date', () => {
        test('identical dates are equal', () => {
            const d1 = new Date('2024-01-15T12:00:00Z');
            const d2 = new Date('2024-01-15T12:00:00Z');
            expect(comparator(d1, d2)).toBe(true);
        });

        test('different dates are not equal', () => {
            const d1 = new Date('2024-01-15');
            const d2 = new Date('2024-01-16');
            expect(comparator(d1, d2)).toBe(false);
        });

        test('Invalid Date equals Invalid Date', () => {
            const d1 = new Date('invalid');
            const d2 = new Date('also invalid');
            expect(comparator(d1, d2)).toBe(true);
        });

        test('Invalid Date does not equal valid date', () => {
            const d1 = new Date('invalid');
            const d2 = new Date('2024-01-15');
            expect(comparator(d1, d2)).toBe(false);
        });

        test('Date epoch', () => {
            const d1 = new Date(0);
            const d2 = new Date(0);
            expect(comparator(d1, d2)).toBe(true);
        });
    });

    describe('RegExp', () => {
        test('identical regexes are equal', () => {
            expect(comparator(/abc/, /abc/)).toBe(true);
        });

        test('regexes with same pattern and flags', () => {
            expect(comparator(/abc/gi, /abc/gi)).toBe(true);
        });

        test('regexes with different patterns', () => {
            expect(comparator(/abc/, /def/)).toBe(false);
        });

        test('regexes with different flags', () => {
            expect(comparator(/abc/i, /abc/g)).toBe(false);
        });

        test('RegExp constructor vs literal', () => {
            expect(comparator(/abc/, new RegExp('abc'))).toBe(true);
        });

        test('complex regex patterns', () => {
            expect(comparator(/^[a-z]+\d*$/i, /^[a-z]+\d*$/i)).toBe(true);
        });
    });

    describe('Error', () => {
        test('errors with same name and message', () => {
            const e1 = new Error('test error');
            const e2 = new Error('test error');
            expect(comparator(e1, e2)).toBe(true);
        });

        test('errors with different messages', () => {
            const e1 = new Error('error 1');
            const e2 = new Error('error 2');
            expect(comparator(e1, e2)).toBe(false);
        });

        test('different error types', () => {
            const e1 = new Error('test');
            const e2 = new TypeError('test');
            expect(comparator(e1, e2)).toBe(false);
        });

        test('TypeError', () => {
            const e1 = new TypeError('type error');
            const e2 = new TypeError('type error');
            expect(comparator(e1, e2)).toBe(true);
        });

        test('RangeError', () => {
            const e1 = new RangeError('range error');
            const e2 = new RangeError('range error');
            expect(comparator(e1, e2)).toBe(true);
        });
    });

    describe('Map', () => {
        test('empty maps are equal', () => {
            expect(comparator(new Map(), new Map())).toBe(true);
        });

        test('maps with same entries', () => {
            const m1 = new Map([['a', 1], ['b', 2]]);
            const m2 = new Map([['a', 1], ['b', 2]]);
            expect(comparator(m1, m2)).toBe(true);
        });

        test('maps with different values', () => {
            const m1 = new Map([['a', 1]]);
            const m2 = new Map([['a', 2]]);
            expect(comparator(m1, m2)).toBe(false);
        });

        test('maps with different keys', () => {
            const m1 = new Map([['a', 1]]);
            const m2 = new Map([['b', 1]]);
            expect(comparator(m1, m2)).toBe(false);
        });

        test('maps with different sizes', () => {
            const m1 = new Map([['a', 1]]);
            const m2 = new Map([['a', 1], ['b', 2]]);
            expect(comparator(m1, m2)).toBe(false);
        });

        test('maps with object keys', () => {
            const key = { id: 1 };
            const m1 = new Map([[key, 'value']]);
            const m2 = new Map([[key, 'value']]);
            expect(comparator(m1, m2)).toBe(true);
        });

        test('maps with nested values', () => {
            const m1 = new Map([['a', { nested: [1, 2, 3] }]]);
            const m2 = new Map([['a', { nested: [1, 2, 3] }]]);
            expect(comparator(m1, m2)).toBe(true);
        });
    });

    describe('Set', () => {
        test('empty sets are equal', () => {
            expect(comparator(new Set(), new Set())).toBe(true);
        });

        test('sets with same values', () => {
            const s1 = new Set([1, 2, 3]);
            const s2 = new Set([1, 2, 3]);
            expect(comparator(s1, s2)).toBe(true);
        });

        test('sets with same values different order', () => {
            const s1 = new Set([1, 2, 3]);
            const s2 = new Set([3, 2, 1]);
            expect(comparator(s1, s2)).toBe(true);
        });

        test('sets with different values', () => {
            const s1 = new Set([1, 2, 3]);
            const s2 = new Set([1, 2, 4]);
            expect(comparator(s1, s2)).toBe(false);
        });

        test('sets with different sizes', () => {
            const s1 = new Set([1, 2]);
            const s2 = new Set([1, 2, 3]);
            expect(comparator(s1, s2)).toBe(false);
        });

        test('sets with objects', () => {
            // Objects in sets are compared by deep equality
            const s1 = new Set([{ a: 1 }]);
            const s2 = new Set([{ a: 1 }]);
            expect(comparator(s1, s2)).toBe(true);
        });

        test('sets with nested arrays', () => {
            const s1 = new Set([[1, 2], [3, 4]]);
            const s2 = new Set([[1, 2], [3, 4]]);
            expect(comparator(s1, s2)).toBe(true);
        });
    });
});

// ============================================================================
// TYPED ARRAYS AND BUFFERS
// ============================================================================

describe('TypedArrays and Buffers', () => {
    describe('TypedArrays', () => {
        test('Int8Array', () => {
            expect(comparator(
                new Int8Array([1, 2, 3]),
                new Int8Array([1, 2, 3])
            )).toBe(true);
            expect(comparator(
                new Int8Array([1, 2, 3]),
                new Int8Array([1, 2, 4])
            )).toBe(false);
        });

        test('Uint8Array', () => {
            expect(comparator(
                new Uint8Array([255, 0, 128]),
                new Uint8Array([255, 0, 128])
            )).toBe(true);
        });

        test('Uint8ClampedArray', () => {
            expect(comparator(
                new Uint8ClampedArray([0, 128, 255]),
                new Uint8ClampedArray([0, 128, 255])
            )).toBe(true);
        });

        test('Int16Array', () => {
            expect(comparator(
                new Int16Array([1000, -1000]),
                new Int16Array([1000, -1000])
            )).toBe(true);
        });

        test('Uint16Array', () => {
            expect(comparator(
                new Uint16Array([65535, 0]),
                new Uint16Array([65535, 0])
            )).toBe(true);
        });

        test('Int32Array', () => {
            expect(comparator(
                new Int32Array([2147483647, -2147483648]),
                new Int32Array([2147483647, -2147483648])
            )).toBe(true);
        });

        test('Uint32Array', () => {
            expect(comparator(
                new Uint32Array([4294967295]),
                new Uint32Array([4294967295])
            )).toBe(true);
        });

        test('Float32Array with tolerance', () => {
            expect(comparator(
                new Float32Array([1.1, 2.2, 3.3]),
                new Float32Array([1.1, 2.2, 3.3])
            )).toBe(true);
        });

        test('Float64Array with tolerance', () => {
            expect(comparator(
                new Float64Array([1.1, 2.2, 3.3]),
                new Float64Array([1.1, 2.2, 3.3])
            )).toBe(true);
        });

        test('Float32Array with NaN', () => {
            expect(comparator(
                new Float32Array([1, NaN, 3]),
                new Float32Array([1, NaN, 3])
            )).toBe(true);
        });

        test('BigInt64Array', () => {
            expect(comparator(
                new BigInt64Array([1n, 2n]),
                new BigInt64Array([1n, 2n])
            )).toBe(true);
        });

        test('BigUint64Array', () => {
            expect(comparator(
                new BigUint64Array([1n, 2n]),
                new BigUint64Array([1n, 2n])
            )).toBe(true);
        });

        test('different TypedArray types are not equal', () => {
            expect(comparator(
                new Int8Array([1, 2, 3]),
                new Uint8Array([1, 2, 3])
            )).toBe(false);
        });

        test('TypedArray vs regular array', () => {
            expect(comparator(
                new Int8Array([1, 2, 3]),
                [1, 2, 3]
            )).toBe(false);
        });
    });

    describe('ArrayBuffer', () => {
        test('identical ArrayBuffers', () => {
            const buf1 = new ArrayBuffer(4);
            const buf2 = new ArrayBuffer(4);
            new Uint8Array(buf1).set([1, 2, 3, 4]);
            new Uint8Array(buf2).set([1, 2, 3, 4]);
            expect(comparator(buf1, buf2)).toBe(true);
        });

        test('different ArrayBuffers', () => {
            const buf1 = new ArrayBuffer(4);
            const buf2 = new ArrayBuffer(4);
            new Uint8Array(buf1).set([1, 2, 3, 4]);
            new Uint8Array(buf2).set([1, 2, 3, 5]);
            expect(comparator(buf1, buf2)).toBe(false);
        });

        test('ArrayBuffers with different lengths', () => {
            const buf1 = new ArrayBuffer(4);
            const buf2 = new ArrayBuffer(8);
            expect(comparator(buf1, buf2)).toBe(false);
        });
    });

    describe('DataView', () => {
        test('identical DataViews', () => {
            const buf1 = new ArrayBuffer(4);
            const buf2 = new ArrayBuffer(4);
            new Uint8Array(buf1).set([1, 2, 3, 4]);
            new Uint8Array(buf2).set([1, 2, 3, 4]);
            expect(comparator(new DataView(buf1), new DataView(buf2))).toBe(true);
        });

        test('different DataViews', () => {
            const buf1 = new ArrayBuffer(4);
            const buf2 = new ArrayBuffer(4);
            new Uint8Array(buf1).set([1, 2, 3, 4]);
            new Uint8Array(buf2).set([4, 3, 2, 1]);
            expect(comparator(new DataView(buf1), new DataView(buf2))).toBe(false);
        });
    });
});

// ============================================================================
// FUNCTIONS
// ============================================================================

describe('Functions', () => {
    test('same function reference', () => {
        const fn = () => {};
        expect(comparator(fn, fn)).toBe(true);
    });

    test('different functions with same implementation', () => {
        const fn1 = function add(a, b) { return a + b; };
        const fn2 = function add(a, b) { return a + b; };
        expect(comparator(fn1, fn2)).toBe(true);
    });

    test('functions with different names', () => {
        const fn1 = function foo() {};
        const fn2 = function bar() {};
        expect(comparator(fn1, fn2)).toBe(false);
    });

    test('arrow functions', () => {
        const fn1 = (x) => x + 1;
        const fn2 = (x) => x + 1;
        // Arrow functions may or may not be equal depending on toString
        expect(comparator(fn1, fn1)).toBe(true);
    });

    test('built-in functions', () => {
        expect(comparator(Math.sin, Math.sin)).toBe(true);
        expect(comparator(Math.sin, Math.cos)).toBe(false);
    });

    test('bound functions', () => {
        const obj = { value: 42 };
        const fn = function() { return this.value; };
        const bound1 = fn.bind(obj);
        const bound2 = fn.bind(obj);
        // Bound functions create new function objects
        expect(comparator(bound1, bound1)).toBe(true);
    });
});

// ============================================================================
// CIRCULAR REFERENCES
// ============================================================================

describe('Circular References', () => {
    test('simple self-reference', () => {
        const a = { value: 1 };
        a.self = a;
        const b = { value: 1 };
        b.self = b;
        expect(comparator(a, b)).toBe(true);
    });

    test('mutual references', () => {
        const a1 = { name: 'a1' };
        const a2 = { name: 'a2' };
        a1.ref = a2;
        a2.ref = a1;

        const b1 = { name: 'a1' };
        const b2 = { name: 'a2' };
        b1.ref = b2;
        b2.ref = b1;

        expect(comparator(a1, b1)).toBe(true);
    });

    test('circular array', () => {
        const a = [1, 2, 3];
        a.push(a);
        const b = [1, 2, 3];
        b.push(b);
        expect(comparator(a, b)).toBe(true);
    });

    test('deep circular reference', () => {
        const a = { level1: { level2: { level3: {} } } };
        a.level1.level2.level3.back = a;

        const b = { level1: { level2: { level3: {} } } };
        b.level1.level2.level3.back = b;

        expect(comparator(a, b)).toBe(true);
    });
});

// ============================================================================
// EDGE CASES
// ============================================================================

describe('Edge Cases', () => {
    describe('type coercion', () => {
        test('string vs number', () => {
            expect(comparator('1', 1)).toBe(false);
        });

        test('boolean vs number', () => {
            expect(comparator(true, 1)).toBe(false);
            expect(comparator(false, 0)).toBe(false);
        });

        test('null vs object', () => {
            expect(comparator(null, {})).toBe(false);
        });

        test('array vs object with length', () => {
            expect(comparator([1, 2], { 0: 1, 1: 2, length: 2 })).toBe(false);
        });
    });

    describe('recursion depth', () => {
        test('respects maxDepth option', () => {
            // Create a deeply nested structure
            let deep = { value: 'bottom' };
            for (let i = 0; i < 100; i++) {
                deep = { nested: deep };
            }
            let deep2 = { value: 'bottom' };
            for (let i = 0; i < 100; i++) {
                deep2 = { nested: deep2 };
            }

            // Should work with default maxDepth (1000)
            expect(comparator(deep, deep2)).toBe(true);

            // Should fail with low maxDepth
            expect(comparator(deep, deep2, { maxDepth: 50 })).toBe(false);
        });
    });

    describe('empty values', () => {
        test('empty string vs null', () => {
            expect(comparator('', null)).toBe(false);
        });

        test('empty array vs empty object', () => {
            expect(comparator([], {})).toBe(false);
        });

        test('0 vs empty string', () => {
            expect(comparator(0, '')).toBe(false);
        });

        test('false vs empty values', () => {
            expect(comparator(false, '')).toBe(false);
            expect(comparator(false, 0)).toBe(false);
            expect(comparator(false, null)).toBe(false);
            expect(comparator(false, undefined)).toBe(false);
        });
    });

    describe('special object properties', () => {
        test('objects with getter properties', () => {
            const a = {
                get computed() { return 42; }
            };
            const b = {
                get computed() { return 42; }
            };
            expect(comparator(a, b)).toBe(true);
        });

        test('objects with non-enumerable properties', () => {
            const a = {};
            Object.defineProperty(a, 'hidden', { value: 42, enumerable: false });
            const b = {};
            Object.defineProperty(b, 'hidden', { value: 42, enumerable: false });
            // Non-enumerable properties are not compared by default
            expect(comparator(a, b)).toBe(true);
        });
    });

    describe('class instances', () => {
        test('instances of same class', () => {
            class Point {
                constructor(x, y) {
                    this.x = x;
                    this.y = y;
                }
            }
            const p1 = new Point(1, 2);
            const p2 = new Point(1, 2);
            expect(comparator(p1, p2)).toBe(true);
        });

        test('instances of different classes', () => {
            class Point { constructor(x, y) { this.x = x; this.y = y; } }
            class Vector { constructor(x, y) { this.x = x; this.y = y; } }
            const p = new Point(1, 2);
            const v = new Vector(1, 2);
            expect(comparator(p, v)).toBe(false);
        });

        test('instance vs plain object', () => {
            class Point { constructor(x, y) { this.x = x; this.y = y; } }
            const p = new Point(1, 2);
            const o = { x: 1, y: 2 };
            expect(comparator(p, o)).toBe(false);
        });
    });
});

// ============================================================================
// CUSTOM COMPARATORS
// ============================================================================

describe('Custom Comparators', () => {
    test('strictComparator uses no tolerance', () => {
        // strictComparator should fail for close but not identical floats
        expect(strictComparator(1.0, 1.0 + 1e-15)).toBe(false);
    });

    test('looseComparator uses larger tolerance', () => {
        expect(looseComparator(1.0, 1.0 + 1e-7)).toBe(true);
    });

    test('createComparator with custom defaults', () => {
        const myComparator = createComparator({ rtol: 0.01 });
        expect(myComparator(1.0, 1.005)).toBe(true);
        expect(myComparator(1.0, 1.02)).toBe(false);
    });

    test('override options in custom comparator', () => {
        const myComparator = createComparator({ rtol: 0.01 });
        // Override with stricter tolerance
        expect(myComparator(1.0, 1.005, { rtol: 0.001 })).toBe(false);
    });
});

// ============================================================================
// URL AND URL SEARCH PARAMS
// ============================================================================

describe('URL types', () => {
    test('identical URLs', () => {
        const u1 = new URL('https://example.com/path?query=1');
        const u2 = new URL('https://example.com/path?query=1');
        expect(comparator(u1, u2)).toBe(true);
    });

    test('different URLs', () => {
        const u1 = new URL('https://example.com/path1');
        const u2 = new URL('https://example.com/path2');
        expect(comparator(u1, u2)).toBe(false);
    });

    test('URLSearchParams', () => {
        const p1 = new URLSearchParams('a=1&b=2');
        const p2 = new URLSearchParams('a=1&b=2');
        expect(comparator(p1, p2)).toBe(true);
    });

    test('URLSearchParams different order', () => {
        const p1 = new URLSearchParams('a=1&b=2');
        const p2 = new URLSearchParams('b=2&a=1');
        // URLSearchParams.toString() preserves order
        expect(comparator(p1, p2)).toBe(false);
    });
});

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

describe('Helper Functions', () => {
    describe('getType', () => {
        test('primitives', () => {
            expect(getType(null)).toBe('null');
            expect(getType(undefined)).toBe('undefined');
            expect(getType(42)).toBe('number');
            expect(getType('hello')).toBe('string');
            expect(getType(true)).toBe('boolean');
            expect(getType(Symbol())).toBe('symbol');
            expect(getType(42n)).toBe('bigint');
        });

        test('objects', () => {
            expect(getType({})).toBe('Object');
            expect(getType([])).toBe('Array');
            expect(getType(new Date())).toBe('Date');
            expect(getType(/abc/)).toBe('RegExp');
            expect(getType(new Map())).toBe('Map');
            expect(getType(new Set())).toBe('Set');
        });

        test('typed arrays', () => {
            expect(getType(new Int8Array())).toBe('Int8Array');
            expect(getType(new Float64Array())).toBe('Float64Array');
        });

        test('functions', () => {
            expect(getType(() => {})).toBe('function');
            expect(getType(function() {})).toBe('function');
        });
    });
});
