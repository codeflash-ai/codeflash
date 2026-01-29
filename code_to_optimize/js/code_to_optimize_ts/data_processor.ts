/**
 * DataProcessor class - demonstrates class method optimization in TypeScript.
 * Contains intentionally inefficient implementations for optimization testing.
 */

/**
 * A class for processing data arrays with various operations.
 */
export class DataProcessor<T> {
    private data: T[];

    /**
     * Create a DataProcessor instance.
     * @param data - Initial data array
     */
    constructor(data: T[] = []) {
        this.data = [...data];
    }

    /**
     * Find duplicates in the data array.
     * Intentionally inefficient O(n²) implementation.
     * @returns Array of duplicate values
     */
    findDuplicates(): T[] {
        const duplicates: T[] = [];
        for (let i = 0; i < this.data.length; i++) {
            for (let j = i + 1; j < this.data.length; j++) {
                if (this.data[i] === this.data[j]) {
                    if (!duplicates.includes(this.data[i])) {
                        duplicates.push(this.data[i]);
                    }
                }
            }
        }
        return duplicates;
    }

    /**
     * Sort the data using bubble sort.
     * Intentionally inefficient O(n²) implementation.
     * @returns Sorted copy of the data
     */
    sortData(): T[] {
        const result = [...this.data];
        const n = result.length;
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n - 1; j++) {
                if (result[j] > result[j + 1]) {
                    const temp = result[j];
                    result[j] = result[j + 1];
                    result[j + 1] = temp;
                }
            }
        }
        return result;
    }

    /**
     * Get unique values from the data.
     * Intentionally inefficient O(n²) implementation.
     * @returns Array of unique values
     */
    getUnique(): T[] {
        const unique: T[] = [];
        for (let i = 0; i < this.data.length; i++) {
            let found = false;
            for (let j = 0; j < unique.length; j++) {
                if (unique[j] === this.data[i]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                unique.push(this.data[i]);
            }
        }
        return unique;
    }

    /**
     * Get the data array.
     * @returns The data array
     */
    getData(): T[] {
        return [...this.data];
    }
}
