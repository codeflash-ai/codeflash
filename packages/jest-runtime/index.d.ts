/**
 * @codeflash/jest-runtime - TypeScript Definitions
 */

/**
 * Capture function return value for behavior verification.
 * Writes result to SQLite database.
 *
 * @param funcName - Name of the function being tested
 * @param lineId - Unique identifier for the call site (e.g., "42" or "42_1")
 * @param targetFn - The function to execute and capture
 * @param args - Arguments to pass to the function
 * @returns The return value of targetFn
 */
export function capture<T>(
    funcName: string,
    lineId: string,
    targetFn: (...args: any[]) => T,
    ...args: any[]
): T;

/**
 * Capture function performance (timing only).
 * Only outputs to stdout, does not write to database.
 *
 * @param funcName - Name of the function being tested
 * @param lineId - Unique identifier for the call site
 * @param targetFn - The function to execute and time
 * @param args - Arguments to pass to the function
 * @returns The return value of targetFn
 */
export function capturePerf<T>(
    funcName: string,
    lineId: string,
    targetFn: (...args: any[]) => T,
    ...args: any[]
): T;

/**
 * Capture multiple invocations in a single call.
 */
export function captureMultiple<T>(
    funcName: string,
    lineId: string,
    invocations: Array<{ fn: (...args: any[]) => T; args: any[] }>
): T[];

/**
 * Write buffered results to disk.
 */
export function writeResults(): void;

/**
 * Clear all buffered results.
 */
export function clearResults(): void;

/**
 * Get all buffered results.
 */
export function getResults(): any[];

/**
 * Set the current test name for result attribution.
 */
export function setTestName(name: string): void;

/**
 * Initialize the SQLite database.
 */
export function initDatabase(): void;

/**
 * Reset invocation counters (call between tests).
 */
export function resetInvocationCounters(): void;

/**
 * Serialize a value to a Buffer.
 * Uses V8 serialization if available, falls back to msgpack.
 */
export function serialize(value: any): Buffer;

/**
 * Deserialize a Buffer back to a value.
 */
export function deserialize(buffer: Buffer): any;

/**
 * Get the serializer type being used ('v8' or 'msgpack').
 */
export function getSerializerType(): 'v8' | 'msgpack';

/**
 * Safely serialize a value, catching errors.
 */
export function safeSerialize(value: any): Buffer | null;

/**
 * Safely deserialize a buffer, catching errors.
 */
export function safeDeserialize(buffer: Buffer): any;

/**
 * Compare two values for deep equality.
 */
export function comparator(
    original: any,
    newValue: any,
    options?: ComparatorOptions
): boolean;

export interface ComparatorOptions {
    /** Relative tolerance for floating point comparison (default: 1e-9) */
    rtol?: number;
    /** Absolute tolerance for floating point comparison (default: 0) */
    atol?: number;
    /** Allow new object to have additional keys (default: false) */
    supersetObj?: boolean;
    /** Maximum recursion depth (default: 1000) */
    maxDepth?: number;
}

/**
 * Create a comparator with custom default options.
 */
export function createComparator(
    defaultOptions?: ComparatorOptions
): (original: any, newValue: any, overrideOptions?: ComparatorOptions) => boolean;

/**
 * Strict comparator (no floating point tolerance).
 */
export const strictComparator: (original: any, newValue: any) => boolean;

/**
 * Loose comparator (larger floating point tolerance).
 */
export const looseComparator: (original: any, newValue: any) => boolean;

/**
 * Check if two floating point numbers are close within tolerance.
 */
export function isClose(a: number, b: number, rtol?: number, atol?: number): boolean;

/**
 * Read test results from a SQLite database.
 */
export function readTestResults(dbPath: string): Map<string, any>;

/**
 * Compare two sets of test results.
 */
export function compareResults(
    originalResults: Map<string, any>,
    candidateResults: Map<string, any>
): CompareResultsOutput;

export interface CompareResultsOutput {
    equivalent: boolean;
    diffs: Array<{
        invocation_id: string;
        scope: string;
        original: string;
        candidate: string | null;
        test_info?: {
            test_module_path: string;
            test_function_name: string;
            function_getting_tested: string;
        };
    }>;
    total_invocations: number;
    original_count: number;
    candidate_count: number;
}

/**
 * Compare serialized result buffers directly.
 */
export function compareBuffers(originalBuffer: Buffer, candidateBuffer: Buffer): boolean;

/**
 * Get invocation index for a test ID.
 */
export function getInvocationIndex(testId: string): number;

/**
 * Sanitize test ID for safe output.
 */
export function sanitizeTestId(testId: string): string;

/** Current loop index from environment */
export const LOOP_INDEX: number;

/** Output file path from environment */
export const OUTPUT_FILE: string;

/** Test iteration from environment */
export const TEST_ITERATION: string;

/** Whether V8 serialization is available */
export const hasV8: boolean;

/** Whether msgpack is available */
export const hasMsgpack: boolean;
