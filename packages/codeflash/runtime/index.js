/**
 * codeflash
 *
 * Codeflash CLI runtime helpers for test instrumentation and behavior verification.
 *
 * Main exports:
 * - capture: Capture function return values for behavior verification
 * - capturePerf: Capture performance metrics (timing only)
 * - serialize/deserialize: Value serialization for storage
 * - comparator: Deep equality comparison
 * - tracer: Function tracing for replay test generation
 * - replay: Replay test utilities
 *
 * Usage (CommonJS):
 *   const { capture, capturePerf } = require('codeflash');
 *
 * Usage (ES Modules):
 *   import { capture, capturePerf } from 'codeflash';
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

// Function tracing (for replay test generation)
let tracer = null;
try {
    tracer = require('./tracer');
} catch (e) {
    // Tracer may not be available if better-sqlite3 is not installed
}

// Replay test utilities
let replay = null;
try {
    replay = require('./replay');
} catch (e) {
    // Replay may not be available
}

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

    // === Batch Looping Control (used by loop-runner) ===
    incrementBatch: capture.incrementBatch,
    getCurrentBatch: capture.getCurrentBatch,
    checkSharedTimeLimit: capture.checkSharedTimeLimit,
    PERF_BATCH_SIZE: capture.PERF_BATCH_SIZE,
    PERF_LOOP_COUNT: capture.PERF_LOOP_COUNT,

    // === Feature Detection ===
    hasV8: serializer.hasV8,
    hasMsgpack: serializer.hasMsgpack,

    // === Function Tracing (for replay test generation) ===
    tracer: tracer ? {
        init: tracer.init,
        wrap: tracer.wrap,
        createWrapper: tracer.createWrapper,
        disable: tracer.disable,
        enable: tracer.enable,
        getStats: tracer.getStats,
    } : null,

    // === Replay Test Utilities ===
    replay: replay ? {
        getNextArg: replay.getNextArg,
        getTracesWithMetadata: replay.getTracesWithMetadata,
        getTracedFunctions: replay.getTracedFunctions,
        getTraceMetadata: replay.getTraceMetadata,
        generateReplayTest: replay.generateReplayTest,
        createReplayTestFromTrace: replay.createReplayTestFromTrace,
    } : null,
};
