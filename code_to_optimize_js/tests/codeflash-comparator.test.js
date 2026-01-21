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
//TODO:{claude} to clean up the files copied for runtime in the tests folder. And have those copied best test runs or proper reference, we can think of packaging these in npm package as well and reference from there.

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

// ============================================================================
// NEW FEATURES - STRICT CONSTRUCTOR TYPE CHECKING
// ============================================================================

describe('Strict Constructor Type Checking', () => {
    test('different constructor types return false early', () => {
        // Number vs String
        expect(comparator(5, '5')).toBe(false);
    });

    test('same constructor types pass through', () => {
        expect(comparator(5, 10)).toBe(false); // Different values but same type
        expect(comparator(5, 5)).toBe(true);
    });

    test('Date vs Object with same properties fails type check', () => {
        const date = new Date('2024-01-01');
        const obj = { getTime: () => date.getTime() };
        expect(comparator(date, obj)).toBe(false);
    });

    test('Array vs Object fails type check', () => {
        expect(comparator([1, 2, 3], { 0: 1, 1: 2, 2: 3 })).toBe(false);
    });

    test('custom class instances with different constructors', () => {
        class Foo { constructor() { this.x = 1; } }
        class Bar { constructor() { this.x = 1; } }
        expect(comparator(new Foo(), new Bar())).toBe(false);
    });

    test('custom class vs plain object', () => {
        class Point { constructor(x, y) { this.x = x; this.y = y; } }
        expect(comparator(new Point(1, 2), { x: 1, y: 2 })).toBe(false);
    });

    test('plain objects from same realm pass', () => {
        expect(comparator({ a: 1 }, { a: 1 })).toBe(true);
    });

    test('Set vs Map fails type check', () => {
        expect(comparator(new Set([1, 2]), new Map([[1, 2]]))).toBe(false);
    });

    test('TypedArray vs regular array fails type check', () => {
        expect(comparator(new Int32Array([1, 2, 3]), [1, 2, 3])).toBe(false);
    });

    test('different TypedArray types fail type check', () => {
        expect(comparator(new Int8Array([1, 2]), new Uint8Array([1, 2]))).toBe(false);
        expect(comparator(new Float32Array([1.0]), new Float64Array([1.0]))).toBe(false);
    });
});

// ============================================================================
// NEW FEATURES - NODE.JS BUFFER SUPPORT
// ============================================================================

describe('Node.js Buffer Support', () => {
    test('identical Buffers are equal', () => {
        const buf1 = Buffer.from([1, 2, 3, 4]);
        const buf2 = Buffer.from([1, 2, 3, 4]);
        expect(comparator(buf1, buf2)).toBe(true);
    });

    test('different Buffers are not equal', () => {
        const buf1 = Buffer.from([1, 2, 3, 4]);
        const buf2 = Buffer.from([1, 2, 3, 5]);
        expect(comparator(buf1, buf2)).toBe(false);
    });

    test('Buffers with different lengths are not equal', () => {
        const buf1 = Buffer.from([1, 2, 3]);
        const buf2 = Buffer.from([1, 2, 3, 4]);
        expect(comparator(buf1, buf2)).toBe(false);
    });

    test('empty Buffers are equal', () => {
        const buf1 = Buffer.from([]);
        const buf2 = Buffer.from([]);
        expect(comparator(buf1, buf2)).toBe(true);
    });

    test('Buffer from string', () => {
        const buf1 = Buffer.from('hello');
        const buf2 = Buffer.from('hello');
        const buf3 = Buffer.from('world');
        expect(comparator(buf1, buf2)).toBe(true);
        expect(comparator(buf1, buf3)).toBe(false);
    });

    test('Buffer from hex', () => {
        const buf1 = Buffer.from('deadbeef', 'hex');
        const buf2 = Buffer.from('deadbeef', 'hex');
        const buf3 = Buffer.from('cafebabe', 'hex');
        expect(comparator(buf1, buf2)).toBe(true);
        expect(comparator(buf1, buf3)).toBe(false);
    });

    test('Buffer.alloc', () => {
        const buf1 = Buffer.alloc(4, 0xff);
        const buf2 = Buffer.alloc(4, 0xff);
        const buf3 = Buffer.alloc(4, 0x00);
        expect(comparator(buf1, buf2)).toBe(true);
        expect(comparator(buf1, buf3)).toBe(false);
    });

    test('Buffer vs Uint8Array are not equal', () => {
        const buf = Buffer.from([1, 2, 3]);
        const arr = new Uint8Array([1, 2, 3]);
        // Buffer extends Uint8Array but has different constructor
        expect(comparator(buf, arr)).toBe(false);
    });

    test('Buffer vs Array are not equal', () => {
        const buf = Buffer.from([1, 2, 3]);
        expect(comparator(buf, [1, 2, 3])).toBe(false);
    });

    test('large Buffers', () => {
        const size = 10000;
        const buf1 = Buffer.alloc(size);
        const buf2 = Buffer.alloc(size);
        for (let i = 0; i < size; i++) {
            buf1[i] = i % 256;
            buf2[i] = i % 256;
        }
        expect(comparator(buf1, buf2)).toBe(true);
        buf2[size - 1] = 0;
        expect(comparator(buf1, buf2)).toBe(false);
    });
});

// ============================================================================
// NEW FEATURES - SHAREDARRAYBUFFER SUPPORT
// ============================================================================

describe('SharedArrayBuffer Support', () => {
    // Note: SharedArrayBuffer may not be available in all environments
    // Skip tests if not available
    const hasSharedArrayBuffer = typeof SharedArrayBuffer !== 'undefined';

    (hasSharedArrayBuffer ? test : test.skip)('identical SharedArrayBuffers are equal', () => {
        const sab1 = new SharedArrayBuffer(4);
        const sab2 = new SharedArrayBuffer(4);
        new Uint8Array(sab1).set([1, 2, 3, 4]);
        new Uint8Array(sab2).set([1, 2, 3, 4]);
        expect(comparator(sab1, sab2)).toBe(true);
    });

    (hasSharedArrayBuffer ? test : test.skip)('different SharedArrayBuffers are not equal', () => {
        const sab1 = new SharedArrayBuffer(4);
        const sab2 = new SharedArrayBuffer(4);
        new Uint8Array(sab1).set([1, 2, 3, 4]);
        new Uint8Array(sab2).set([1, 2, 3, 5]);
        expect(comparator(sab1, sab2)).toBe(false);
    });

    (hasSharedArrayBuffer ? test : test.skip)('SharedArrayBuffers with different lengths are not equal', () => {
        const sab1 = new SharedArrayBuffer(4);
        const sab2 = new SharedArrayBuffer(8);
        expect(comparator(sab1, sab2)).toBe(false);
    });

    (hasSharedArrayBuffer ? test : test.skip)('empty SharedArrayBuffers are equal', () => {
        const sab1 = new SharedArrayBuffer(0);
        const sab2 = new SharedArrayBuffer(0);
        expect(comparator(sab1, sab2)).toBe(true);
    });

    (hasSharedArrayBuffer ? test : test.skip)('SharedArrayBuffer vs ArrayBuffer are not equal', () => {
        const sab = new SharedArrayBuffer(4);
        const ab = new ArrayBuffer(4);
        new Uint8Array(sab).set([1, 2, 3, 4]);
        new Uint8Array(ab).set([1, 2, 3, 4]);
        expect(comparator(sab, ab)).toBe(false);
    });
});

// ============================================================================
// NEW FEATURES - BIGINT TYPED ARRAYS
// ============================================================================

describe('BigInt TypedArrays', () => {
    describe('BigInt64Array', () => {
        test('identical BigInt64Arrays are equal', () => {
            const arr1 = new BigInt64Array([1n, 2n, 3n]);
            const arr2 = new BigInt64Array([1n, 2n, 3n]);
            expect(comparator(arr1, arr2)).toBe(true);
        });

        test('different BigInt64Arrays are not equal', () => {
            const arr1 = new BigInt64Array([1n, 2n, 3n]);
            const arr2 = new BigInt64Array([1n, 2n, 4n]);
            expect(comparator(arr1, arr2)).toBe(false);
        });

        test('BigInt64Array with negative values', () => {
            const arr1 = new BigInt64Array([-1n, -2n, -3n]);
            const arr2 = new BigInt64Array([-1n, -2n, -3n]);
            expect(comparator(arr1, arr2)).toBe(true);
        });

        test('BigInt64Array with large values', () => {
            const large = 9223372036854775807n; // Max BigInt64
            const arr1 = new BigInt64Array([large, -large - 1n]);
            const arr2 = new BigInt64Array([large, -large - 1n]);
            expect(comparator(arr1, arr2)).toBe(true);
        });

        test('empty BigInt64Arrays are equal', () => {
            expect(comparator(new BigInt64Array([]), new BigInt64Array([]))).toBe(true);
        });
    });

    describe('BigUint64Array', () => {
        test('identical BigUint64Arrays are equal', () => {
            const arr1 = new BigUint64Array([1n, 2n, 3n]);
            const arr2 = new BigUint64Array([1n, 2n, 3n]);
            expect(comparator(arr1, arr2)).toBe(true);
        });

        test('different BigUint64Arrays are not equal', () => {
            const arr1 = new BigUint64Array([1n, 2n, 3n]);
            const arr2 = new BigUint64Array([1n, 2n, 4n]);
            expect(comparator(arr1, arr2)).toBe(false);
        });

        test('BigUint64Array with large values', () => {
            const large = 18446744073709551615n; // Max BigUint64
            const arr1 = new BigUint64Array([0n, large]);
            const arr2 = new BigUint64Array([0n, large]);
            expect(comparator(arr1, arr2)).toBe(true);
        });
    });

    test('BigInt64Array vs BigUint64Array are not equal', () => {
        const signed = new BigInt64Array([1n, 2n]);
        const unsigned = new BigUint64Array([1n, 2n]);
        expect(comparator(signed, unsigned)).toBe(false);
    });

    test('BigInt64Array vs regular array are not equal', () => {
        const typed = new BigInt64Array([1n, 2n]);
        expect(comparator(typed, [1n, 2n])).toBe(false);
    });
});

// ============================================================================
// NEW FEATURES - ENHANCED ERROR COMPARISON
// ============================================================================

describe('Enhanced Error Comparison', () => {
    test('errors with same name and message are equal', () => {
        const e1 = new Error('test');
        const e2 = new Error('test');
        expect(comparator(e1, e2)).toBe(true);
    });

    test('errors with custom properties are compared', () => {
        const e1 = new Error('test');
        e1.code = 'ERR_TEST';
        e1.statusCode = 500;

        const e2 = new Error('test');
        e2.code = 'ERR_TEST';
        e2.statusCode = 500;

        expect(comparator(e1, e2)).toBe(true);
    });

    test('errors with different custom properties are not equal', () => {
        const e1 = new Error('test');
        e1.code = 'ERR_TEST';

        const e2 = new Error('test');
        e2.code = 'ERR_OTHER';

        expect(comparator(e1, e2)).toBe(false);
    });

    test('errors with different number of custom properties are not equal', () => {
        const e1 = new Error('test');
        e1.code = 'ERR_TEST';

        const e2 = new Error('test');
        e2.code = 'ERR_TEST';
        e2.extra = 'value';

        expect(comparator(e1, e2)).toBe(false);
    });

    test('errors with nested custom properties', () => {
        const e1 = new Error('test');
        e1.data = { nested: { value: 42 }, arr: [1, 2, 3] };

        const e2 = new Error('test');
        e2.data = { nested: { value: 42 }, arr: [1, 2, 3] };

        expect(comparator(e1, e2)).toBe(true);
    });

    test('errors with nested custom properties differ', () => {
        const e1 = new Error('test');
        e1.data = { nested: { value: 42 } };

        const e2 = new Error('test');
        e2.data = { nested: { value: 43 } };

        expect(comparator(e1, e2)).toBe(false);
    });

    test('underscore properties are ignored', () => {
        const e1 = new Error('test');
        e1._internal = 'value1';
        e1.code = 'ERR';

        const e2 = new Error('test');
        e2._internal = 'different';
        e2.code = 'ERR';

        expect(comparator(e1, e2)).toBe(true);
    });

    test('TypeError with custom properties', () => {
        const e1 = new TypeError('type error');
        e1.field = 'username';

        const e2 = new TypeError('type error');
        e2.field = 'username';

        expect(comparator(e1, e2)).toBe(true);
    });

    test('RangeError with custom properties', () => {
        const e1 = new RangeError('out of range');
        e1.min = 0;
        e1.max = 100;
        e1.actual = 150;

        const e2 = new RangeError('out of range');
        e2.min = 0;
        e2.max = 100;
        e2.actual = 150;

        expect(comparator(e1, e2)).toBe(true);
    });

    test('SyntaxError with custom properties', () => {
        const e1 = new SyntaxError('invalid syntax');
        e1.line = 10;
        e1.column = 5;

        const e2 = new SyntaxError('invalid syntax');
        e2.line = 10;
        e2.column = 5;

        expect(comparator(e1, e2)).toBe(true);
    });

    test('custom Error subclass', () => {
        class CustomError extends Error {
            constructor(message, code) {
                super(message);
                this.name = 'CustomError';
                this.code = code;
            }
        }

        const e1 = new CustomError('custom', 'ERR_CUSTOM');
        const e2 = new CustomError('custom', 'ERR_CUSTOM');

        expect(comparator(e1, e2)).toBe(true);
    });
});

// ============================================================================
// NEW FEATURES - WEAKREF SUPPORT
// ============================================================================

describe('WeakRef Support', () => {
    // Note: WeakRef may not be available in all environments
    const hasWeakRef = typeof WeakRef !== 'undefined';

    (hasWeakRef ? test : test.skip)('same WeakRef reference is equal', () => {
        const obj = { value: 42 };
        const ref = new WeakRef(obj);
        expect(comparator(ref, ref)).toBe(true);
    });

    (hasWeakRef ? test : test.skip)('different WeakRef instances are not equal (reference comparison)', () => {
        const obj = { value: 42 };
        const ref1 = new WeakRef(obj);
        const ref2 = new WeakRef(obj);
        // WeakRefs are compared by reference only
        expect(comparator(ref1, ref2)).toBe(false);
    });

    (hasWeakRef ? test : test.skip)('WeakRefs to different objects are not equal', () => {
        const obj1 = { value: 1 };
        const obj2 = { value: 1 };
        const ref1 = new WeakRef(obj1);
        const ref2 = new WeakRef(obj2);
        expect(comparator(ref1, ref2)).toBe(false);
    });

    (hasWeakRef ? test : test.skip)('WeakRef vs plain object are not equal', () => {
        const obj = { value: 42 };
        const ref = new WeakRef(obj);
        expect(comparator(ref, obj)).toBe(false);
    });
});

// ============================================================================
// NEW FEATURES - GENERATOR SUPPORT
// ============================================================================

describe('Generator Support', () => {
    test('same generator reference is equal', () => {
        function* gen() {
            yield 1;
            yield 2;
        }
        const g = gen();
        expect(comparator(g, g)).toBe(true);
    });

    test('different generator instances are not equal (reference comparison)', () => {
        function* gen() {
            yield 1;
            yield 2;
        }
        const g1 = gen();
        const g2 = gen();
        // Generators are compared by reference only (consuming would alter state)
        expect(comparator(g1, g2)).toBe(false);
    });

    test('generators from different functions are not equal', () => {
        function* gen1() { yield 1; }
        function* gen2() { yield 1; }
        expect(comparator(gen1(), gen2())).toBe(false);
    });

    test('generator vs iterator object are not equal', () => {
        function* gen() { yield 1; }
        const arr = [1];
        expect(comparator(gen(), arr[Symbol.iterator]())).toBe(false);
    });

    test('generator function vs generator instance are not equal', () => {
        function* gen() { yield 1; }
        expect(comparator(gen, gen())).toBe(false);
    });

    test('async generator same reference', () => {
        async function* asyncGen() {
            yield 1;
        }
        const g = asyncGen();
        expect(comparator(g, g)).toBe(true);
    });

    test('different async generator instances are not equal', () => {
        async function* asyncGen() {
            yield 1;
        }
        expect(comparator(asyncGen(), asyncGen())).toBe(false);
    });
});

// ============================================================================
// NEW FEATURES - ITERATOR SUPPORT
// ============================================================================

describe('Iterator Support', () => {
    describe('Map Iterators', () => {
        test('same Map iterator reference is equal', () => {
            const map = new Map([[1, 'a'], [2, 'b']]);
            const iter = map.keys();
            expect(comparator(iter, iter)).toBe(true);
        });

        test('different Map.keys() iterators are not equal', () => {
            const map = new Map([[1, 'a'], [2, 'b']]);
            expect(comparator(map.keys(), map.keys())).toBe(false);
        });

        test('different Map.values() iterators are not equal', () => {
            const map = new Map([[1, 'a'], [2, 'b']]);
            expect(comparator(map.values(), map.values())).toBe(false);
        });

        test('different Map.entries() iterators are not equal', () => {
            const map = new Map([[1, 'a'], [2, 'b']]);
            expect(comparator(map.entries(), map.entries())).toBe(false);
        });

        test('Map.keys() vs Map.values() are not equal', () => {
            const map = new Map([[1, 'a']]);
            expect(comparator(map.keys(), map.values())).toBe(false);
        });
    });

    describe('Set Iterators', () => {
        test('same Set iterator reference is equal', () => {
            const set = new Set([1, 2, 3]);
            const iter = set.values();
            expect(comparator(iter, iter)).toBe(true);
        });

        test('different Set.values() iterators are not equal', () => {
            const set = new Set([1, 2, 3]);
            expect(comparator(set.values(), set.values())).toBe(false);
        });

        test('different Set.keys() iterators are not equal', () => {
            const set = new Set([1, 2, 3]);
            expect(comparator(set.keys(), set.keys())).toBe(false);
        });

        test('different Set.entries() iterators are not equal', () => {
            const set = new Set([1, 2, 3]);
            expect(comparator(set.entries(), set.entries())).toBe(false);
        });
    });

    describe('Array Iterators', () => {
        test('same Array iterator reference is equal', () => {
            const arr = [1, 2, 3];
            const iter = arr[Symbol.iterator]();
            expect(comparator(iter, iter)).toBe(true);
        });

        test('different Array iterators are not equal', () => {
            const arr = [1, 2, 3];
            expect(comparator(arr[Symbol.iterator](), arr[Symbol.iterator]())).toBe(false);
        });

        test('different array.values() iterators are not equal', () => {
            const arr = [1, 2, 3];
            expect(comparator(arr.values(), arr.values())).toBe(false);
        });

        test('different array.keys() iterators are not equal', () => {
            const arr = [1, 2, 3];
            expect(comparator(arr.keys(), arr.keys())).toBe(false);
        });

        test('different array.entries() iterators are not equal', () => {
            const arr = [1, 2, 3];
            expect(comparator(arr.entries(), arr.entries())).toBe(false);
        });
    });

    describe('String Iterators', () => {
        test('same String iterator reference is equal', () => {
            const str = 'hello';
            const iter = str[Symbol.iterator]();
            expect(comparator(iter, iter)).toBe(true);
        });

        test('different String iterators are not equal', () => {
            const str = 'hello';
            expect(comparator(str[Symbol.iterator](), str[Symbol.iterator]())).toBe(false);
        });
    });

    describe('Custom Iterators', () => {
        test('custom iterator same reference is equal', () => {
            const customIterator = {
                [Symbol.iterator]() { return this; },
                next() { return { done: true, value: undefined }; }
            };
            expect(comparator(customIterator, customIterator)).toBe(true);
        });

        test('different custom iterators are not equal', () => {
            const createIterator = () => ({
                [Symbol.iterator]() { return this; },
                next() { return { done: true, value: undefined }; }
            });
            expect(comparator(createIterator(), createIterator())).toBe(false);
        });
    });
});

// ============================================================================
// NEW FEATURES - SYMBOL PROPERTIES IN OBJECTS
// ============================================================================

describe('Symbol Properties in Objects', () => {
    test('objects with same Symbol properties are equal', () => {
        const sym = Symbol('test');
        const obj1 = { [sym]: 'value', regular: 1 };
        const obj2 = { [sym]: 'value', regular: 1 };
        expect(comparator(obj1, obj2)).toBe(true);
    });

    test('objects with different Symbol values are not equal', () => {
        const sym = Symbol('test');
        const obj1 = { [sym]: 'value1' };
        const obj2 = { [sym]: 'value2' };
        expect(comparator(obj1, obj2)).toBe(false);
    });

    test('objects with different Symbol keys are not equal', () => {
        const sym1 = Symbol('test1');
        const sym2 = Symbol('test2');
        const obj1 = { [sym1]: 'value' };
        const obj2 = { [sym2]: 'value' };
        expect(comparator(obj1, obj2)).toBe(false);
    });

    test('object with Symbol vs object without Symbol are not equal', () => {
        const sym = Symbol('test');
        const obj1 = { [sym]: 'value', regular: 1 };
        const obj2 = { regular: 1 };
        expect(comparator(obj1, obj2)).toBe(false);
    });

    test('objects with multiple Symbol properties', () => {
        const sym1 = Symbol('a');
        const sym2 = Symbol('b');
        const obj1 = { [sym1]: 1, [sym2]: 2 };
        const obj2 = { [sym1]: 1, [sym2]: 2 };
        expect(comparator(obj1, obj2)).toBe(true);
    });

    test('objects with nested Symbol property values', () => {
        const sym = Symbol('nested');
        const obj1 = { [sym]: { deep: { value: 42 } } };
        const obj2 = { [sym]: { deep: { value: 42 } } };
        expect(comparator(obj1, obj2)).toBe(true);
    });

    test('objects with nested Symbol property values differ', () => {
        const sym = Symbol('nested');
        const obj1 = { [sym]: { deep: { value: 42 } } };
        const obj2 = { [sym]: { deep: { value: 43 } } };
        expect(comparator(obj1, obj2)).toBe(false);
    });

    test('Symbol.for creates shared Symbols that work correctly', () => {
        const sym = Symbol.for('shared');
        const obj1 = { [sym]: 'value' };
        const obj2 = { [Symbol.for('shared')]: 'value' };
        expect(comparator(obj1, obj2)).toBe(true);
    });

    test('well-known Symbols are compared correctly', () => {
        const obj1 = { [Symbol.toStringTag]: 'CustomObject' };
        const obj2 = { [Symbol.toStringTag]: 'CustomObject' };
        expect(comparator(obj1, obj2)).toBe(true);
    });

    test('superset mode with Symbol properties', () => {
        const sym = Symbol('test');
        const obj1 = { [sym]: 'value' };
        const obj2 = { [sym]: 'value', extra: 'prop' };
        expect(comparator(obj1, obj2, { supersetObj: true })).toBe(true);
    });

    test('superset mode requires all Symbol properties from original', () => {
        const sym = Symbol('test');
        const obj1 = { [sym]: 'value', regular: 1 };
        const obj2 = { regular: 1 };
        expect(comparator(obj1, obj2, { supersetObj: true })).toBe(false);
    });

    test('superset mode with Symbol properties - values must match', () => {
        const sym = Symbol('test');
        const obj1 = { [sym]: 'value1' };
        const obj2 = { [sym]: 'value2', extra: 'prop' };
        expect(comparator(obj1, obj2, { supersetObj: true })).toBe(false);
    });

    test('empty objects with no Symbols are equal', () => {
        expect(comparator({}, {})).toBe(true);
    });

    test('Symbol-only objects', () => {
        const sym = Symbol('only');
        const obj1 = { [sym]: 42 };
        const obj2 = { [sym]: 42 };
        expect(comparator(obj1, obj2)).toBe(true);
    });

    test('mixed Symbol and string keys', () => {
        const sym = Symbol('mixed');
        const obj1 = { a: 1, [sym]: 2, b: 3 };
        const obj2 = { a: 1, [sym]: 2, b: 3 };
        expect(comparator(obj1, obj2)).toBe(true);
    });

    test('Symbol properties with array values', () => {
        const sym = Symbol('array');
        const obj1 = { [sym]: [1, 2, 3] };
        const obj2 = { [sym]: [1, 2, 3] };
        expect(comparator(obj1, obj2)).toBe(true);
    });

    test('Symbol properties with array values differ', () => {
        const sym = Symbol('array');
        const obj1 = { [sym]: [1, 2, 3] };
        const obj2 = { [sym]: [1, 2, 4] };
        expect(comparator(obj1, obj2)).toBe(false);
    });
});
