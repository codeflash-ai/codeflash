/**
 * Formatting helper functions - ES Module version.
 */

/**
 * Format a number to specified decimal places.
 * @param num - Number to format
 * @param decimals - Number of decimal places
 * @returns Formatted number
 */
export function formatNumber(num, decimals) {
    return Number(num.toFixed(decimals));
}

/**
 * Validate that input is a valid number.
 * @param value - Value to validate
 * @param name - Parameter name for error message
 * @throws Error if value is not a valid number
 */
export function validateInput(value, name) {
    if (typeof value !== 'number' || isNaN(value)) {
        throw new Error(`Invalid ${name}: must be a number`);
    }
}

/**
 * Format currency with symbol.
 * @param amount - Amount to format
 * @param symbol - Currency symbol
 * @returns Formatted currency string
 */
export function formatCurrency(amount, symbol = '$') {
    return `${symbol}${formatNumber(amount, 2)}`;
}