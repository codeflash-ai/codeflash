/**
 * Async utility functions - ES Module version.
 * Contains intentionally inefficient implementations for optimization testing.
 */

/**
 * Simulate a delay (for testing purposes).
 * @param {number} ms - Milliseconds to delay
 * @returns {Promise<void>}
 */
export function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Process items sequentially when they could be parallel.
 * Intentionally inefficient - processes items one at a time.
 * @param {any[]} items - Items to process
 * @param {function} processor - Async function to process each item
 * @returns {Promise<any[]>} Processed results
 */
export async function processItemsSequential(items, processor) {
    const results = [];
    for (let i = 0; i < items.length; i++) {
        const result = await processor(items[i]);
        results.push(result);
    }
    return results;
}

/**
 * Map over items asynchronously with a concurrency limit.
 * Intentionally simple/inefficient implementation - ignores concurrency.
 * @param {any[]} items - Items to process
 * @param {function} mapper - Async mapper function
 * @param {number} concurrency - Max concurrent operations (currently ignored)
 * @returns {Promise<any[]>} Mapped results
 */
export async function asyncMap(items, mapper, concurrency = 1) {
    // Inefficient: ignores concurrency, processes sequentially
    const results = [];
    for (const item of items) {
        results.push(await mapper(item));
    }
    return results;
}

/**
 * Filter items asynchronously.
 * Inefficient implementation that processes items one by one.
 * @param {any[]} items - Items to filter
 * @param {function} predicate - Async predicate function
 * @returns {Promise<any[]>} Filtered items
 */
export async function asyncFilter(items, predicate) {
    const results = [];
    for (const item of items) {
        const shouldInclude = await predicate(item);
        if (shouldInclude) {
            results.push(item);
        }
    }
    return results;
}
