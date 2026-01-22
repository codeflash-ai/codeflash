/**
 * @codeflash/jest-runtime - ES Module wrapper
 *
 * This file provides ES Module exports for the package.
 * The actual implementation is in CommonJS (index.js).
 */

import cjs from './index.js';

// Re-export all named exports
export const {
    // Main Instrumentation API
    capture,
    capturePerf,
    captureMultiple,

    // Test Lifecycle
    writeResults,
    clearResults,
    getResults,
    setTestName,
    initDatabase,
    resetInvocationCounters,

    // Serialization
    serialize,
    deserialize,
    getSerializerType,
    safeSerialize,
    safeDeserialize,

    // Comparison
    comparator,
    createComparator,
    strictComparator,
    looseComparator,
    isClose,

    // Result Comparison
    readTestResults,
    compareResults,
    compareBuffers,

    // Utilities
    getInvocationIndex,
    sanitizeTestId,

    // Constants
    LOOP_INDEX,
    OUTPUT_FILE,
    TEST_ITERATION,

    // Feature Detection
    hasV8,
    hasMsgpack,
} = cjs;

// Default export for convenience
export default cjs;
