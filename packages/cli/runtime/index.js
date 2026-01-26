/**
 * @codeflash/cli
 *
 * Codeflash CLI runtime helpers for test instrumentation and behavior verification.
 *
 * Main exports:
 * - capture: Capture function return values for behavior verification
 * - capturePerf: Capture performance metrics (timing only)
 * - serialize/deserialize: Value serialization for storage
 * - comparator: Deep equality comparison
 *
 * Usage (CommonJS):
 *   const { capture, capturePerf } = require('@codeflash/cli');
 *
 * Usage (ES Modules):
 *   import { capture, capturePerf } from '@codeflash/cli';
 */

'use strict';

// Main capture functions (instrumentation)
const capture = require('./capture');

// Serialization utilities
const serializer = require('./serializer');

// Comparison utilities
const comparator = require('./comparator');

// Result comparison (used by CLI)
const compareResults = require('./compare-results');

// Re-export all public APIs
module.exports = {
    // === Main Instrumentation API ===
    capture: capture.capture,
    capturePerf: capture.capturePerf,
    captureMultiple: capture.captureMultiple,

    // === Test Lifecycle ===
    writeResults: capture.writeResults,
    clearResults: capture.clearResults,
    getResults: capture.getResults,
    setTestName: capture.setTestName,
    initDatabase: capture.initDatabase,
    resetInvocationCounters: capture.resetInvocationCounters,

    // === Serialization ===
    serialize: serializer.serialize,
    deserialize: serializer.deserialize,
    getSerializerType: serializer.getSerializerType,
    safeSerialize: capture.safeSerialize,
    safeDeserialize: capture.safeDeserialize,

    // === Comparison ===
    comparator: comparator.comparator,
    createComparator: comparator.createComparator,
    strictComparator: comparator.strictComparator,
    looseComparator: comparator.looseComparator,
    isClose: comparator.isClose,

    // === Result Comparison (CLI helpers) ===
    readTestResults: compareResults.readTestResults,
    compareResults: compareResults.compareResults,
    compareBuffers: compareResults.compareBuffers,

    // === Utilities ===
    getInvocationIndex: capture.getInvocationIndex,
    sanitizeTestId: capture.sanitizeTestId,

    // === Constants ===
    LOOP_INDEX: capture.LOOP_INDEX,
    OUTPUT_FILE: capture.OUTPUT_FILE,
    TEST_ITERATION: capture.TEST_ITERATION,

    // === Feature Detection ===
    hasV8: serializer.hasV8,
    hasMsgpack: serializer.hasMsgpack,
};
