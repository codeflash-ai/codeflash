/**
 * Codeflash Comparator - Deep equality comparison for JavaScript values
 *
 * This module provides a robust comparator function for comparing JavaScript
 * values to determine behavioral equivalence between original and optimized code.
 *
 * Features:
 * - Handles all JavaScript primitive types
 * - Floating point comparison with relative tolerance (like Python's math.isclose)
 * - Deep comparison of objects, arrays, Maps, Sets
 * - Handles special values: NaN, Infinity, -Infinity, undefined, null
 * - Handles TypedArrays (including BigInt64Array, BigUint64Array), Date, RegExp, Error objects
 * - Node.js Buffer support
 * - SharedArrayBuffer support
 * - Error objects with custom properties comparison
 * - WeakRef, Generator, and Iterator support (reference comparison)
 * - Symbol property comparison in objects
 * - Circular reference detection
 * - Superset mode: allows new object to have additional keys
 * - Strict constructor type checking for early mismatch detection
 *
 * Usage:
 *   const { comparator } = require('./codeflash-comparator');
 *   comparator(original, optimized);              // Exact comparison
 *   comparator(original, optimized, { supersetObj: true });  // Allow extra keys
 */

'use strict';

/**
 * Default options for the comparator.
 */
const DEFAULT_OPTIONS = {
    // Relative tolerance for floating point comparison (like Python's rtol)
    rtol: 1e-9,
    // Absolute tolerance for floating point comparison (like Python's atol)
    atol: 0,
    // If true, the new object is allowed to have more keys than the original
    supersetObj: false,
    // Maximum recursion depth to prevent stack overflow
    maxDepth: 1000,
};

/**
 * Check if two floating point numbers are close within tolerance.
 * Equivalent to Python's math.isclose(a, b, rel_tol, abs_tol).
 *
 * @param {number} a - First number
 * @param {number} b - Second number
 * @param {number} rtol - Relative tolerance (default: 1e-9)
 * @param {number} atol - Absolute tolerance (default: 0)
 * @returns {boolean} - True if numbers are close
 */
function isClose(a, b, rtol = 1e-9, atol = 0) {
    // Handle identical values (including both being 0)
    if (a === b) return true;

    // Handle NaN
    if (Number.isNaN(a) && Number.isNaN(b)) return true;
    if (Number.isNaN(a) || Number.isNaN(b)) return false;

    // Handle Infinity
    if (!Number.isFinite(a) || !Number.isFinite(b)) {
        return a === b; // Both must be same infinity
    }

    // Use the same formula as Python's math.isclose
    // abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
    const diff = Math.abs(a - b);
    const maxAbs = Math.max(Math.abs(a), Math.abs(b));
    return diff <= Math.max(rtol * maxAbs, atol);
}

/**
 * Get the precise type of a value for comparison.
 *
 * @param {any} value - The value to get the type of
 * @returns {string} - The type name
 */
function getType(value) {
    if (value === null) return 'null';
    if (value === undefined) return 'undefined';

    const type = typeof value;
    if (type !== 'object') return type;

    // Get the constructor name for objects
    const constructorName = value.constructor?.name;
    if (constructorName) return constructorName;

    // Fallback to Object.prototype.toString
    return Object.prototype.toString.call(value).slice(8, -1);
}

/**
 * Check if a value is a TypedArray.
 *
 * @param {any} value - The value to check
 * @returns {boolean} - True if TypedArray
 */
function isTypedArray(value) {
    return ArrayBuffer.isView(value) && !(value instanceof DataView);
}

/**
 * Compare two values for deep equality.
 *
 * @param {any} orig - Original value
 * @param {any} newVal - New value to compare
 * @param {Object} options - Comparison options
 * @param {number} options.rtol - Relative tolerance for floats
 * @param {number} options.atol - Absolute tolerance for floats
 * @param {boolean} options.supersetObj - Allow new object to have extra keys
 * @param {number} options.maxDepth - Maximum recursion depth
 * @returns {boolean} - True if values are equivalent
 */
function comparator(orig, newVal, options = {}) {
    const opts = { ...DEFAULT_OPTIONS, ...options };

    // Track visited objects to handle circular references
    const visited = new WeakMap();

    function compare(a, b, depth) {
        // Check recursion depth
        if (depth > opts.maxDepth) {
            console.warn('[comparator] Maximum recursion depth exceeded');
            return false;
        }

        // === Identical references ===
        if (a === b) return true;

        // === Handle null and undefined ===
        if (a === null || a === undefined || b === null || b === undefined) {
            return a === b;
        }

        // === Strict constructor type checking ===
        // Check constructor mismatch early (similar to Python's type checking)
        const constructorA = a?.constructor;
        const constructorB = b?.constructor;
        if (constructorA !== constructorB) {
            // Allow comparison if both are plain objects from different realms
            const nameA = constructorA?.name;
            const nameB = constructorB?.name;
            if (nameA !== nameB) {
                return false;
            }
        }

        // === Type checking ===
        const typeA = typeof a;
        const typeB = typeof b;

        if (typeA !== typeB) {
            // Special case: comparing number with BigInt
            // In JavaScript, 1n !== 1, but we might want to consider them equal
            // For strict behavioral comparison, we'll say they're different
            return false;
        }

        // === Primitives ===

        // Numbers (including NaN and Infinity)
        if (typeA === 'number') {
            return isClose(a, b, opts.rtol, opts.atol);
        }

        // Strings, booleans
        if (typeA === 'string' || typeA === 'boolean') {
            return a === b;
        }

        // BigInt
        if (typeA === 'bigint') {
            return a === b;
        }

        // Symbols - compare by description since Symbol() always creates unique
        if (typeA === 'symbol') {
            return a.description === b.description;
        }

        // Functions - compare by reference (same function)
        if (typeA === 'function') {
            // Functions are equal if they're the same reference
            // or if they have the same name and source code
            if (a === b) return true;
            // For bound functions or native functions, we can only compare by reference
            try {
                return a.name === b.name && a.toString() === b.toString();
            } catch (e) {
                return false;
            }
        }

        // === Objects (typeA === 'object') ===

        // Check for circular references
        if (visited.has(a)) {
            // If we've seen 'a' before, check if 'b' was the corresponding value
            return visited.get(a) === b;
        }

        // Mark as visited before recursing
        visited.set(a, b);

        try {
            // === Arrays ===
            if (Array.isArray(a)) {
                if (!Array.isArray(b)) return false;
                if (a.length !== b.length) return false;
                return a.every((elem, i) => compare(elem, b[i], depth + 1));
            }

            // === TypedArrays (Int8Array, Uint8Array, Float32Array, etc.) ===
            if (isTypedArray(a)) {
                if (!isTypedArray(b)) return false;
                if (a.constructor !== b.constructor) return false;
                if (a.length !== b.length) return false;

                // For float arrays, use tolerance comparison
                if (a instanceof Float32Array || a instanceof Float64Array) {
                    for (let i = 0; i < a.length; i++) {
                        if (!isClose(a[i], b[i], opts.rtol, opts.atol)) return false;
                    }
                    return true;
                }

                // For BigInt arrays, use exact comparison
                if (a instanceof BigInt64Array || a instanceof BigUint64Array) {
                    for (let i = 0; i < a.length; i++) {
                        if (a[i] !== b[i]) return false;
                    }
                    return true;
                }

                // For integer arrays, use exact comparison
                for (let i = 0; i < a.length; i++) {
                    if (a[i] !== b[i]) return false;
                }
                return true;
            }

            // === ArrayBuffer ===
            if (a instanceof ArrayBuffer) {
                if (!(b instanceof ArrayBuffer)) return false;
                if (a.byteLength !== b.byteLength) return false;
                const viewA = new Uint8Array(a);
                const viewB = new Uint8Array(b);
                for (let i = 0; i < viewA.length; i++) {
                    if (viewA[i] !== viewB[i]) return false;
                }
                return true;
            }

            // === DataView ===
            if (a instanceof DataView) {
                if (!(b instanceof DataView)) return false;
                if (a.byteLength !== b.byteLength) return false;
                for (let i = 0; i < a.byteLength; i++) {
                    if (a.getUint8(i) !== b.getUint8(i)) return false;
                }
                return true;
            }

            // === Node.js Buffer ===
            if (typeof Buffer !== 'undefined' && Buffer.isBuffer(a)) {
                if (!Buffer.isBuffer(b)) return false;
                if (a.length !== b.length) return false;
                return a.equals(b);
            }

            // === SharedArrayBuffer ===
            if (typeof SharedArrayBuffer !== 'undefined' && a instanceof SharedArrayBuffer) {
                if (!(b instanceof SharedArrayBuffer)) return false;
                if (a.byteLength !== b.byteLength) return false;
                const viewA = new Uint8Array(a);
                const viewB = new Uint8Array(b);
                for (let i = 0; i < viewA.length; i++) {
                    if (viewA[i] !== viewB[i]) return false;
                }
                return true;
            }

            // === Date ===
            if (a instanceof Date) {
                if (!(b instanceof Date)) return false;
                // Handle Invalid Date (NaN time)
                const timeA = a.getTime();
                const timeB = b.getTime();
                if (Number.isNaN(timeA) && Number.isNaN(timeB)) return true;
                return timeA === timeB;
            }

            // === RegExp ===
            if (a instanceof RegExp) {
                if (!(b instanceof RegExp)) return false;
                return a.source === b.source && a.flags === b.flags;
            }

            // === Error ===
            if (a instanceof Error) {
                if (!(b instanceof Error)) return false;
                // Compare error name and message
                if (a.name !== b.name) return false;
                if (a.message !== b.message) return false;
                // Compare all non-underscore enumerable properties (like Python's __dict__)
                const propsA = Object.keys(a).filter(k => !k.startsWith('_'));
                const propsB = Object.keys(b).filter(k => !k.startsWith('_'));
                if (propsA.length !== propsB.length) return false;
                for (const key of propsA) {
                    if (!propsB.includes(key)) return false;
                    if (!compare(a[key], b[key], depth + 1)) return false;
                }
                return true;
            }

            // === Map ===
            if (a instanceof Map) {
                if (!(b instanceof Map)) return false;
                if (a.size !== b.size) return false;
                for (const [key, val] of a) {
                    if (!b.has(key)) return false;
                    if (!compare(val, b.get(key), depth + 1)) return false;
                }
                return true;
            }

            // === Set ===
            if (a instanceof Set) {
                if (!(b instanceof Set)) return false;
                if (a.size !== b.size) return false;
                // For Sets, we need to find matching elements
                // This is O(n^2) but necessary for deep comparison
                const bArray = Array.from(b);
                for (const valA of a) {
                    let found = false;
                    for (let i = 0; i < bArray.length; i++) {
                        if (compare(valA, bArray[i], depth + 1)) {
                            found = true;
                            bArray.splice(i, 1); // Remove matched element
                            break;
                        }
                    }
                    if (!found) return false;
                }
                return true;
            }

            // === WeakMap / WeakSet ===
            // Cannot iterate over these, so we can only compare by reference
            if (a instanceof WeakMap || a instanceof WeakSet) {
                return a === b;
            }

            // === WeakRef ===
            // WeakRefs can only be compared by reference
            if (typeof WeakRef !== 'undefined' && a instanceof WeakRef) {
                return a === b;
            }

            // === Promise ===
            // Promises can only be compared by reference
            if (a instanceof Promise) {
                return a === b;
            }

            // === Generator ===
            // Get the generator prototype for comparison
            const generatorPrototype = Object.getPrototypeOf(function* () {});
            if (Object.getPrototypeOf(a) === generatorPrototype) {
                // Generators can only be compared by reference (consuming would alter state)
                return a === b;
            }

            // === Map/Set/Array Iterators and Async Generator ===
            // These are created by .keys(), .values(), .entries() methods or async generator functions
            const typeName = Object.prototype.toString.call(a);
            if (
                typeName === '[object Map Iterator]' ||
                typeName === '[object Set Iterator]' ||
                typeName === '[object Array Iterator]' ||
                typeName === '[object AsyncGenerator]'
            ) {
                // Iterators and async generators can only be compared by reference (consuming would alter state)
                return a === b;
            }

            // === Generic Iterator objects ===
            // Objects that implement the iterator protocol
            if (typeof a[Symbol.iterator] === 'function' && typeof a.next === 'function') {
                // Iterators compared by reference (can't consume to compare)
                return a === b;
            }

            // === URL ===
            if (typeof URL !== 'undefined' && a instanceof URL) {
                if (!(b instanceof URL)) return false;
                return a.href === b.href;
            }

            // === URLSearchParams ===
            if (typeof URLSearchParams !== 'undefined' && a instanceof URLSearchParams) {
                if (!(b instanceof URLSearchParams)) return false;
                return a.toString() === b.toString();
            }

            // === Plain Objects ===
            // This includes class instances

            const keysA = Object.keys(a);
            const keysB = Object.keys(b);
            const symbolsA = Object.getOwnPropertySymbols(a);
            const symbolsB = Object.getOwnPropertySymbols(b);

            if (opts.supersetObj) {
                // In superset mode, all keys from original must exist in new
                // but new can have additional keys
                // Check string keys
                for (const key of keysA) {
                    if (!(key in b)) return false;
                    if (!compare(a[key], b[key], depth + 1)) return false;
                }
                // Check symbol keys
                for (const sym of symbolsA) {
                    if (!symbolsB.includes(sym)) return false;
                    if (!compare(a[sym], b[sym], depth + 1)) return false;
                }
                return true;
            } else {
                // Exact key matching
                if (keysA.length !== keysB.length) return false;
                if (symbolsA.length !== symbolsB.length) return false;

                // Check string keys
                for (const key of keysA) {
                    if (!(key in b)) return false;
                    if (!compare(a[key], b[key], depth + 1)) return false;
                }
                // Check symbol keys
                for (const sym of symbolsA) {
                    if (!symbolsB.includes(sym)) return false;
                    if (!compare(a[sym], b[sym], depth + 1)) return false;
                }
                return true;
            }
        } finally {
            // Clean up visited tracking
            // Note: We don't delete from visited because the same object
            // might appear multiple times in the structure
        }
    }

    try {
        return compare(orig, newVal, 0);
    } catch (e) {
        console.error('[comparator] Error during comparison:', e);
        return false;
    }
}

/**
 * Create a comparator with custom default options.
 *
 * @param {Object} defaultOptions - Default options for all comparisons
 * @returns {Function} - Comparator function with bound defaults
 */
function createComparator(defaultOptions = {}) {
    const opts = { ...DEFAULT_OPTIONS, ...defaultOptions };
    return (orig, newVal, overrideOptions = {}) => {
        return comparator(orig, newVal, { ...opts, ...overrideOptions });
    };
}

/**
 * Strict comparator that requires exact equality (no tolerance).
 */
const strictComparator = createComparator({ rtol: 0, atol: 0 });

/**
 * Loose comparator with larger tolerance for floating point.
 */
const looseComparator = createComparator({ rtol: 1e-6, atol: 1e-9 });

// Export public API
module.exports = {
    comparator,
    createComparator,
    strictComparator,
    looseComparator,
    isClose,
    getType,
    DEFAULT_OPTIONS,
};
