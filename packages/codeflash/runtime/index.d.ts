/**
 * Codeflash TypeScript Declarations
 */

/**
 * Capture a function call for behavior verification.
 * Records inputs, outputs, timing to SQLite database.
 *
 * @param funcName - Name of the function being tested
 * @param lineId - Line number identifier in test file
 * @param fn - The function to call
 * @param args - Arguments to pass to the function
 * @returns The function's return value
 */
export function capture<T extends (...args: any[]) => any>(
    funcName: string,
    lineId: string,
    fn: T,
    ...args: Parameters<T>
): ReturnType<T>;

/**
 * Capture a function call for performance benchmarking.
 * Only measures timing, prints to stdout.
 *
 * @param funcName - Name of the function being tested
 * @param lineId - Line number identifier in test file
 * @param fn - The function to call
 * @param args - Arguments to pass to the function
 * @returns The function's return value
 */
export function capturePerf<T extends (...args: any[]) => any>(
    funcName: string,
    lineId: string,
    fn: T,
    ...args: Parameters<T>
): ReturnType<T>;

/**
 * Capture multiple invocations for benchmarking.
 *
 * @param funcName - Name of the function being tested
 * @param lineId - Line number identifier
 * @param fn - The function to call
 * @param argsList - List of argument arrays to test
 * @returns Array of return values
 */
export function captureMultiple<T extends (...args: any[]) => any>(
    funcName: string,
    lineId: string,
    fn: T,
    argsList: Parameters<T>[]
): ReturnType<T>[];

/**
 * Write remaining results to file.
 */
export function writeResults(): void;

/**
 * Clear all recorded results.
 */
export function clearResults(): void;

/**
 * Get the current results buffer.
 */
export function getResults(): any[];

/**
 * Set the current test name.
 */
export function setTestName(name: string): void;

/**
 * Serialize a value for storage.
 */
export function safeSerialize(value: any): Buffer;

/**
 * Deserialize a buffer back to a value.
 */
export function safeDeserialize(buffer: Buffer | Uint8Array): any;

/**
 * Initialize the SQLite database.
 */
export function initDatabase(): void;

/**
 * Reset invocation counters.
 */
export function resetInvocationCounters(): void;

/**
 * Get invocation index for a testId.
 */
export function getInvocationIndex(testId: string): number;

/**
 * Sanitize a string for use in test IDs.
 */
export function sanitizeTestId(str: string): string;

/**
 * Get the serializer type being used.
 */
export function getSerializerType(): 'v8' | 'msgpack';

/**
 * Current loop index from environment.
 */
export const LOOP_INDEX: number;

/**
 * Output file path from environment.
 */
export const OUTPUT_FILE: string;

/**
 * Test iteration from environment.
 */
export const TEST_ITERATION: string;

// Default export for CommonJS compatibility
declare const codeflash: {
    capture: typeof capture;
    capturePerf: typeof capturePerf;
    captureMultiple: typeof captureMultiple;
    writeResults: typeof writeResults;
    clearResults: typeof clearResults;
    getResults: typeof getResults;
    setTestName: typeof setTestName;
    safeSerialize: typeof safeSerialize;
    safeDeserialize: typeof safeDeserialize;
    initDatabase: typeof initDatabase;
    resetInvocationCounters: typeof resetInvocationCounters;
    getInvocationIndex: typeof getInvocationIndex;
    sanitizeTestId: typeof sanitizeTestId;
    getSerializerType: typeof getSerializerType;
    LOOP_INDEX: typeof LOOP_INDEX;
    OUTPUT_FILE: typeof OUTPUT_FILE;
    TEST_ITERATION: typeof TEST_ITERATION;
};

export default codeflash;
