/**
 * Codeflash Universal Serializer
 *
 * A robust serialization system for JavaScript values that:
 * 1. Prefers V8 serialization (Node.js native) - fastest, handles all JS types
 * 2. Falls back to msgpack with custom extensions (for Bun/browser environments)
 *
 * Supports:
 * - All primitive types (null, undefined, boolean, number, string, bigint, symbol)
 * - Special numbers (NaN, Infinity, -Infinity)
 * - Objects, Arrays (including sparse arrays)
 * - Map, Set, WeakMap references, WeakSet references
 * - Date, RegExp, Error (and subclasses)
 * - TypedArrays (Int8Array, Uint8Array, Float32Array, etc.)
 * - ArrayBuffer, SharedArrayBuffer, DataView
 * - Circular references
 * - Functions (by reference/name only)
 *
 * Usage:
 *   const { serialize, deserialize, getSerializerType } = require('./codeflash-serializer');
 *
 *   const buffer = serialize(value);
 *   const restored = deserialize(buffer);
 */

'use strict';

// ============================================================================
// SERIALIZER DETECTION
// ============================================================================

let useV8 = false;
let v8Module = null;

// Try to load V8 module (available in Node.js)
try {
    v8Module = require('v8');
    // Verify serialize/deserialize are available
    if (typeof v8Module.serialize === 'function' && typeof v8Module.deserialize === 'function') {
        // Perform a self-test to verify V8 serialization works correctly
        // This catches cases like Jest's VM context where V8 serialization
        // produces data that deserializes incorrectly (Maps become plain objects)
        const testMap = new Map([['__test__', 1]]);
        const testBuffer = v8Module.serialize(testMap);
        const testRestored = v8Module.deserialize(testBuffer);

        if (testRestored instanceof Map && testRestored.get('__test__') === 1) {
            useV8 = true;
        } else {
            // V8 serialization is broken in this environment (e.g., Jest)
            useV8 = false;
        }
    }
} catch (e) {
    // V8 not available (Bun, browser, etc.)
}

// Load msgpack as fallback
let msgpack = null;
try {
    msgpack = require('@msgpack/msgpack');
} catch (e) {
    // msgpack not installed
}

/**
 * Get the serializer type being used.
 * @returns {string} - 'v8' or 'msgpack'
 */
function getSerializerType() {
    return useV8 ? 'v8' : 'msgpack';
}

// ============================================================================
// V8 SERIALIZATION (PRIMARY)
// ============================================================================

/**
 * Serialize a value using V8's native serialization.
 * This handles all JavaScript types including:
 * - Primitives, Objects, Arrays
 * - Map, Set, Date, RegExp, Error
 * - TypedArrays, ArrayBuffer
 * - Circular references
 *
 * @param {any} value - Value to serialize
 * @returns {Buffer} - Serialized buffer
 */
function serializeV8(value) {
    try {
        return v8Module.serialize(value);
    } catch (e) {
        // V8 can't serialize some things (functions, symbols in some contexts)
        // Fall back to wrapped serialization
        return v8Module.serialize(wrapForV8(value));
    }
}

/**
 * Deserialize a V8-serialized buffer.
 *
 * @param {Buffer} buffer - Serialized buffer
 * @returns {any} - Deserialized value
 */
function deserializeV8(buffer) {
    const value = v8Module.deserialize(buffer);
    return unwrapFromV8(value);
}

/**
 * Wrap values that V8 can't serialize natively.
 * V8 can't serialize: functions, symbols (in some cases)
 */
function wrapForV8(value, seen = new WeakMap()) {
    if (value === null || value === undefined) return value;

    const type = typeof value;

    // Primitives that V8 handles
    if (type === 'number' || type === 'string' || type === 'boolean' || type === 'bigint') {
        return value;
    }

    // Symbols - wrap with marker
    if (type === 'symbol') {
        return { __codeflash_type__: 'Symbol', description: value.description };
    }

    // Functions - wrap with marker
    if (type === 'function') {
        return {
            __codeflash_type__: 'Function',
            name: value.name || 'anonymous',
            // Can't serialize function body reliably
        };
    }

    // Objects
    if (type === 'object') {
        // Check for circular reference
        if (seen.has(value)) {
            return seen.get(value);
        }

        // V8 handles most objects natively
        // Just need to recurse into arrays and plain objects to wrap nested functions/symbols

        if (Array.isArray(value)) {
            const wrapped = [];
            seen.set(value, wrapped);
            for (let i = 0; i < value.length; i++) {
                if (i in value) {
                    wrapped[i] = wrapForV8(value[i], seen);
                }
            }
            return wrapped;
        }

        // V8 handles these natively
        if (value instanceof Date || value instanceof RegExp || value instanceof Error ||
            value instanceof Map || value instanceof Set ||
            ArrayBuffer.isView(value) || value instanceof ArrayBuffer) {
            return value;
        }

        // Plain objects - recurse
        const wrapped = {};
        seen.set(value, wrapped);
        for (const key of Object.keys(value)) {
            wrapped[key] = wrapForV8(value[key], seen);
        }
        return wrapped;
    }

    return value;
}

/**
 * Unwrap values that were wrapped for V8 serialization.
 */
function unwrapFromV8(value, seen = new WeakMap()) {
    if (value === null || value === undefined) return value;

    const type = typeof value;

    if (type !== 'object') return value;

    // Check for circular reference
    if (seen.has(value)) {
        return seen.get(value);
    }

    // Check for wrapped types
    if (value.__codeflash_type__) {
        switch (value.__codeflash_type__) {
            case 'Symbol':
                return Symbol(value.description);
            case 'Function':
                // Can't restore function body, return a placeholder
                const fn = function() { throw new Error(`Deserialized function placeholder: ${value.name}`); };
                Object.defineProperty(fn, 'name', { value: value.name });
                return fn;
            default:
                // Unknown wrapped type, return as-is
                return value;
        }
    }

    // Arrays
    if (Array.isArray(value)) {
        const unwrapped = [];
        seen.set(value, unwrapped);
        for (let i = 0; i < value.length; i++) {
            if (i in value) {
                unwrapped[i] = unwrapFromV8(value[i], seen);
            }
        }
        return unwrapped;
    }

    // V8 restores these natively
    if (value instanceof Date || value instanceof RegExp || value instanceof Error ||
        value instanceof Map || value instanceof Set ||
        ArrayBuffer.isView(value) || value instanceof ArrayBuffer) {
        return value;
    }

    // Plain objects - recurse
    const unwrapped = {};
    seen.set(value, unwrapped);
    for (const key of Object.keys(value)) {
        unwrapped[key] = unwrapFromV8(value[key], seen);
    }
    return unwrapped;
}

// ============================================================================
// MSGPACK SERIALIZATION (FALLBACK)
// ============================================================================

/**
 * Extension type IDs for msgpack.
 * Using negative IDs to avoid conflicts with user-defined extensions.
 */
const EXT_TYPES = {
    UNDEFINED: 0x01,
    NAN: 0x02,
    INFINITY_POS: 0x03,
    INFINITY_NEG: 0x04,
    BIGINT: 0x05,
    SYMBOL: 0x06,
    DATE: 0x07,
    REGEXP: 0x08,
    ERROR: 0x09,
    MAP: 0x0A,
    SET: 0x0B,
    INT8ARRAY: 0x10,
    UINT8ARRAY: 0x11,
    UINT8CLAMPEDARRAY: 0x12,
    INT16ARRAY: 0x13,
    UINT16ARRAY: 0x14,
    INT32ARRAY: 0x15,
    UINT32ARRAY: 0x16,
    FLOAT32ARRAY: 0x17,
    FLOAT64ARRAY: 0x18,
    BIGINT64ARRAY: 0x19,
    BIGUINT64ARRAY: 0x1A,
    ARRAYBUFFER: 0x1B,
    DATAVIEW: 0x1C,
    FUNCTION: 0x1D,
    CIRCULAR_REF: 0x1E,
    SPARSE_ARRAY: 0x1F,
};

/**
 * Create msgpack extension codec for JavaScript types.
 */
function createMsgpackCodec() {
    const extensionCodec = new msgpack.ExtensionCodec();

    // Undefined
    extensionCodec.register({
        type: EXT_TYPES.UNDEFINED,
        encode: (value) => {
            if (value === undefined) return new Uint8Array(0);
            return null;
        },
        decode: () => undefined,
    });

    // NaN
    extensionCodec.register({
        type: EXT_TYPES.NAN,
        encode: (value) => {
            if (typeof value === 'number' && Number.isNaN(value)) return new Uint8Array(0);
            return null;
        },
        decode: () => NaN,
    });

    // Positive Infinity
    extensionCodec.register({
        type: EXT_TYPES.INFINITY_POS,
        encode: (value) => {
            if (value === Infinity) return new Uint8Array(0);
            return null;
        },
        decode: () => Infinity,
    });

    // Negative Infinity
    extensionCodec.register({
        type: EXT_TYPES.INFINITY_NEG,
        encode: (value) => {
            if (value === -Infinity) return new Uint8Array(0);
            return null;
        },
        decode: () => -Infinity,
    });

    // BigInt
    extensionCodec.register({
        type: EXT_TYPES.BIGINT,
        encode: (value) => {
            if (typeof value === 'bigint') {
                const str = value.toString();
                return new TextEncoder().encode(str);
            }
            return null;
        },
        decode: (data) => {
            const str = new TextDecoder().decode(data);
            return BigInt(str);
        },
    });

    // Symbol
    extensionCodec.register({
        type: EXT_TYPES.SYMBOL,
        encode: (value) => {
            if (typeof value === 'symbol') {
                // Distinguish between undefined description and empty string
                // Use a special marker for undefined description
                const desc = value.description;
                if (desc === undefined) {
                    return new TextEncoder().encode('\x00__UNDEF__');
                }
                return new TextEncoder().encode(desc);
            }
            return null;
        },
        decode: (data) => {
            const description = new TextDecoder().decode(data);
            // Check for undefined marker
            if (description === '\x00__UNDEF__') {
                return Symbol();
            }
            return Symbol(description);
        },
    });

    // Note: Date is handled via marker objects in prepareForMsgpack/restoreFromMsgpack
    // because msgpack's built-in timestamp extension doesn't properly handle NaN (Invalid Date)

    // RegExp - use Object.prototype.toString for cross-context detection
    extensionCodec.register({
        type: EXT_TYPES.REGEXP,
        encode: (value) => {
            if (Object.prototype.toString.call(value) === '[object RegExp]') {
                const obj = { source: value.source, flags: value.flags };
                return msgpack.encode(obj);
            }
            return null;
        },
        decode: (data) => {
            const obj = msgpack.decode(data);
            return new RegExp(obj.source, obj.flags);
        },
    });

    // Error - use Object.prototype.toString for cross-context detection
    extensionCodec.register({
        type: EXT_TYPES.ERROR,
        encode: (value) => {
            // Check for Error-like objects (cross-VM-context compatible)
            if (Object.prototype.toString.call(value) === '[object Error]' ||
                (value && value.name && value.message !== undefined && value.stack !== undefined)) {
                const obj = {
                    name: value.name,
                    message: value.message,
                    stack: value.stack,
                    // Include custom properties
                    ...Object.fromEntries(
                        Object.entries(value).filter(([k]) => !['name', 'message', 'stack'].includes(k))
                    ),
                };
                return msgpack.encode(obj);
            }
            return null;
        },
        decode: (data) => {
            const obj = msgpack.decode(data);
            let ErrorClass = Error;
            // Try to use the appropriate error class
            const errorClasses = {
                TypeError, RangeError, SyntaxError, ReferenceError,
                URIError, EvalError, Error
            };
            if (obj.name in errorClasses) {
                ErrorClass = errorClasses[obj.name];
            }
            const error = new ErrorClass(obj.message);
            error.stack = obj.stack;
            // Restore custom properties
            for (const [key, val] of Object.entries(obj)) {
                if (!['name', 'message', 'stack'].includes(key)) {
                    error[key] = val;
                }
            }
            return error;
        },
    });

    // Function (limited - can't serialize body)
    extensionCodec.register({
        type: EXT_TYPES.FUNCTION,
        encode: (value) => {
            if (typeof value === 'function') {
                return new TextEncoder().encode(value.name || 'anonymous');
            }
            return null;
        },
        decode: (data) => {
            const name = new TextDecoder().decode(data);
            const fn = function() { throw new Error(`Deserialized function placeholder: ${name}`); };
            Object.defineProperty(fn, 'name', { value: name });
            return fn;
        },
    });

    return extensionCodec;
}

// Singleton codec instance
let msgpackCodec = null;

function getMsgpackCodec() {
    if (!msgpackCodec && msgpack) {
        msgpackCodec = createMsgpackCodec();
    }
    return msgpackCodec;
}

/**
 * Prepare a value for msgpack serialization.
 * Handles types that need special treatment beyond extensions.
 */
function prepareForMsgpack(value, seen = new Map(), refId = { current: 0 }) {
    if (value === null) return null;
    // undefined needs special handling because msgpack converts it to null
    if (value === undefined) return { __codeflash_undefined__: true };

    const type = typeof value;

    // Special number values that msgpack doesn't handle correctly
    if (type === 'number') {
        if (Number.isNaN(value)) return { __codeflash_nan__: true };
        if (value === Infinity) return { __codeflash_infinity__: true };
        if (value === -Infinity) return { __codeflash_neg_infinity__: true };
        return value;
    }

    // Primitives that msgpack handles or our extensions handle
    if (type === 'string' || type === 'boolean' ||
        type === 'bigint' || type === 'symbol' || type === 'function') {
        return value;
    }

    if (type !== 'object') return value;

    // Check for circular reference
    if (seen.has(value)) {
        return { __codeflash_circular__: seen.get(value) };
    }

    // Assign reference ID for potential circular refs
    const id = refId.current++;
    seen.set(value, id);

    // Use toString for cross-VM-context type detection
    const tag = Object.prototype.toString.call(value);

    // Date - handle specially because msgpack's built-in timestamp doesn't handle NaN
    if (tag === '[object Date]') {
        const time = value.getTime();
        // Store as marker object with the timestamp
        // We use a string representation to preserve NaN
        return {
            __codeflash_date__: Number.isNaN(time) ? '__NAN__' : time,
            __id__: id,
        };
    }

    // RegExp, Error - handled by extensions
    if (tag === '[object RegExp]' || tag === '[object Error]') {
        return value;
    }

    // Map (use toString for cross-VM-context)
    if (tag === '[object Map]') {
        const entries = [];
        for (const [k, v] of value) {
            entries.push([prepareForMsgpack(k, seen, refId), prepareForMsgpack(v, seen, refId)]);
        }
        return { __codeflash_map__: entries, __id__: id };
    }

    // Set (use toString for cross-VM-context)
    if (tag === '[object Set]') {
        const values = [];
        for (const v of value) {
            values.push(prepareForMsgpack(v, seen, refId));
        }
        return { __codeflash_set__: values, __id__: id };
    }

    // TypedArrays (use ArrayBuffer.isView which works cross-context)
    if (ArrayBuffer.isView(value) && tag !== '[object DataView]') {
        return {
            __codeflash_typedarray__: value.constructor.name,
            data: Array.from(value),
            __id__: id,
        };
    }

    // DataView (use toString for cross-VM-context)
    if (tag === '[object DataView]') {
        return {
            __codeflash_dataview__: true,
            data: Array.from(new Uint8Array(value.buffer, value.byteOffset, value.byteLength)),
            __id__: id,
        };
    }

    // ArrayBuffer (use toString for cross-VM-context)
    if (tag === '[object ArrayBuffer]') {
        return {
            __codeflash_arraybuffer__: true,
            data: Array.from(new Uint8Array(value)),
            __id__: id,
        };
    }

    // Arrays - always wrap in marker to preserve __id__ for circular references
    // (msgpack doesn't preserve non-numeric properties on arrays)
    if (Array.isArray(value)) {
        const isSparse = value.length > 0 && Object.keys(value).length !== value.length;
        if (isSparse) {
            // Sparse array - store as object with indices
            const sparse = { __codeflash_sparse_array__: true, length: value.length, elements: {}, __id__: id };
            for (const key of Object.keys(value)) {
                sparse.elements[key] = prepareForMsgpack(value[key], seen, refId);
            }
            return sparse;
        }
        // Dense array - wrap in marker object to preserve __id__
        const elements = [];
        for (let i = 0; i < value.length; i++) {
            elements[i] = prepareForMsgpack(value[i], seen, refId);
        }
        return { __codeflash_array__: elements, __id__: id };
    }

    // Plain objects
    const obj = { __id__: id };
    for (const key of Object.keys(value)) {
        obj[key] = prepareForMsgpack(value[key], seen, refId);
    }
    return obj;
}

/**
 * Restore a value after msgpack deserialization.
 */
function restoreFromMsgpack(value, refs = new Map()) {
    if (value === null || value === undefined) return value;

    const type = typeof value;
    if (type !== 'object') return value;

    // Built-in types that msgpack handles via extensions - return as-is
    // These should NOT be treated as plain objects (use toString for cross-VM-context)
    // Note: Date is handled via marker objects, so not included here
    const tag = Object.prototype.toString.call(value);
    if (tag === '[object RegExp]' || tag === '[object Error]') {
        return value;
    }

    // Special value markers
    if (value.__codeflash_undefined__) return undefined;
    if (value.__codeflash_nan__) return NaN;
    if (value.__codeflash_infinity__) return Infinity;
    if (value.__codeflash_neg_infinity__) return -Infinity;

    // Date marker
    if (value.__codeflash_date__ !== undefined) {
        const time = value.__codeflash_date__ === '__NAN__' ? NaN : value.__codeflash_date__;
        const date = new Date(time);
        const id = value.__id__;
        if (id !== undefined) refs.set(id, date);
        return date;
    }

    // Check for circular reference marker
    if (value.__codeflash_circular__ !== undefined) {
        return refs.get(value.__codeflash_circular__);
    }

    // Store reference if this object has an ID
    const id = value.__id__;

    // Map
    if (value.__codeflash_map__) {
        const map = new Map();
        if (id !== undefined) refs.set(id, map);
        for (const [k, v] of value.__codeflash_map__) {
            map.set(restoreFromMsgpack(k, refs), restoreFromMsgpack(v, refs));
        }
        return map;
    }

    // Set
    if (value.__codeflash_set__) {
        const set = new Set();
        if (id !== undefined) refs.set(id, set);
        for (const v of value.__codeflash_set__) {
            set.add(restoreFromMsgpack(v, refs));
        }
        return set;
    }

    // TypedArrays
    if (value.__codeflash_typedarray__) {
        const TypedArrayClass = globalThis[value.__codeflash_typedarray__];
        if (TypedArrayClass) {
            const arr = new TypedArrayClass(value.data);
            if (id !== undefined) refs.set(id, arr);
            return arr;
        }
    }

    // DataView
    if (value.__codeflash_dataview__) {
        const buffer = new ArrayBuffer(value.data.length);
        new Uint8Array(buffer).set(value.data);
        const view = new DataView(buffer);
        if (id !== undefined) refs.set(id, view);
        return view;
    }

    // ArrayBuffer
    if (value.__codeflash_arraybuffer__) {
        const buffer = new ArrayBuffer(value.data.length);
        new Uint8Array(buffer).set(value.data);
        if (id !== undefined) refs.set(id, buffer);
        return buffer;
    }

    // Dense array marker
    if (value.__codeflash_array__) {
        const arr = [];
        if (id !== undefined) refs.set(id, arr);
        const elements = value.__codeflash_array__;
        for (let i = 0; i < elements.length; i++) {
            arr[i] = restoreFromMsgpack(elements[i], refs);
        }
        return arr;
    }

    // Sparse array
    if (value.__codeflash_sparse_array__) {
        const arr = new Array(value.length);
        if (id !== undefined) refs.set(id, arr);
        for (const [key, val] of Object.entries(value.elements)) {
            arr[parseInt(key, 10)] = restoreFromMsgpack(val, refs);
        }
        return arr;
    }

    // Arrays (legacy - shouldn't happen with new format, but keep for safety)
    if (Array.isArray(value)) {
        const arr = [];
        if (id !== undefined) refs.set(id, arr);
        for (let i = 0; i < value.length; i++) {
            if (i in value) {
                arr[i] = restoreFromMsgpack(value[i], refs);
            }
        }
        return arr;
    }

    // Plain objects - remove __id__ from result
    const obj = {};
    if (id !== undefined) refs.set(id, obj);
    for (const [key, val] of Object.entries(value)) {
        if (key !== '__id__') {
            obj[key] = restoreFromMsgpack(val, refs);
        }
    }
    return obj;
}

/**
 * Serialize a value using msgpack with extensions.
 *
 * @param {any} value - Value to serialize
 * @returns {Buffer} - Serialized buffer
 */
function serializeMsgpack(value) {
    if (!msgpack) {
        throw new Error('msgpack not available and V8 serialization not available');
    }

    const codec = getMsgpackCodec();
    const prepared = prepareForMsgpack(value);
    const encoded = msgpack.encode(prepared, { extensionCodec: codec });
    return Buffer.from(encoded);
}

/**
 * Deserialize a msgpack-serialized buffer.
 *
 * @param {Buffer|Uint8Array} buffer - Serialized buffer
 * @returns {any} - Deserialized value
 */
function deserializeMsgpack(buffer) {
    if (!msgpack) {
        throw new Error('msgpack not available');
    }

    const codec = getMsgpackCodec();
    const decoded = msgpack.decode(buffer, { extensionCodec: codec });
    return restoreFromMsgpack(decoded);
}

// ============================================================================
// PUBLIC API
// ============================================================================

/**
 * Serialize a value using the best available method.
 * Prefers V8 serialization, falls back to msgpack.
 *
 * @param {any} value - Value to serialize
 * @returns {Buffer} - Serialized buffer with format marker
 */
function serialize(value) {
    // Add a format marker byte at the start
    // 0x01 = V8, 0x02 = msgpack
    if (useV8) {
        const serialized = serializeV8(value);
        const result = Buffer.allocUnsafe(serialized.length + 1);
        result[0] = 0x01;
        serialized.copy(result, 1);
        return result;
    } else {
        const serialized = serializeMsgpack(value);
        const result = Buffer.allocUnsafe(serialized.length + 1);
        result[0] = 0x02;
        serialized.copy(result, 1);
        return result;
    }
}

/**
 * Deserialize a buffer that was serialized with serialize().
 * Automatically detects the format from the marker byte.
 *
 * @param {Buffer|Uint8Array} buffer - Serialized buffer
 * @returns {any} - Deserialized value
 */
function deserialize(buffer) {
    if (!buffer || buffer.length === 0) {
        throw new Error('Empty buffer cannot be deserialized');
    }

    const format = buffer[0];
    const data = buffer.slice(1);

    if (format === 0x01) {
        // V8 format
        if (!useV8) {
            throw new Error('Buffer was serialized with V8 but V8 is not available');
        }
        return deserializeV8(data);
    } else if (format === 0x02) {
        // msgpack format
        return deserializeMsgpack(data);
    } else {
        throw new Error(`Unknown serialization format: ${format}`);
    }
}

/**
 * Force serialization using a specific method.
 * Useful for testing or cross-environment compatibility.
 */
const serializeWith = {
    v8: useV8 ? (value) => {
        const serialized = serializeV8(value);
        const result = Buffer.allocUnsafe(serialized.length + 1);
        result[0] = 0x01;
        serialized.copy(result, 1);
        return result;
    } : null,

    msgpack: msgpack ? (value) => {
        const serialized = serializeMsgpack(value);
        const result = Buffer.allocUnsafe(serialized.length + 1);
        result[0] = 0x02;
        serialized.copy(result, 1);
        return result;
    } : null,
};

// ============================================================================
// EXPORTS
// ============================================================================

module.exports = {
    // Main API
    serialize,
    deserialize,
    getSerializerType,

    // Force specific serializer
    serializeWith,

    // Low-level (for testing)
    serializeV8: useV8 ? serializeV8 : null,
    deserializeV8: useV8 ? deserializeV8 : null,
    serializeMsgpack: msgpack ? serializeMsgpack : null,
    deserializeMsgpack: msgpack ? deserializeMsgpack : null,

    // Feature detection
    hasV8: useV8,
    hasMsgpack: !!msgpack,

    // Extension types (for reference)
    EXT_TYPES,
};
