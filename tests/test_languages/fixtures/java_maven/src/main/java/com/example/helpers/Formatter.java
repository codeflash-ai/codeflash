package com.example.helpers;

/**
 * Formatting utility functions.
 */
public class Formatter {

    /**
     * Format a number with specified decimal places.
     *
     * @param value Number to format
     * @param decimals Number of decimal places
     * @return Formatted number as string
     */
    public static String formatNumber(double value, int decimals) {
        return String.format("%." + decimals + "f", value);
    }

    /**
     * Validate that input is a positive number.
     *
     * @param value Value to validate
     * @param name Name of the parameter (for error message)
     * @throws IllegalArgumentException if value is not positive
     */
    public static void validateInput(double value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be non-negative, got: " + value);
        }
    }

    /**
     * Convert number to percentage string.
     *
     * @param value Decimal value (0.5 = 50%)
     * @return Percentage string
     */
    public static String toPercentage(double value) {
        return formatNumber(value * 100, 2) + "%";
    }

    /**
     * Pad a string to specified length.
     *
     * @param str String to pad
     * @param length Target length
     * @param padChar Character to pad with
     * @return Padded string
     */
    public static String padLeft(String str, int length, char padChar) {
        // Inefficient: creates many intermediate strings
        StringBuilder result = new StringBuilder(str);
        while (result.length() < length) {
            result.insert(0, padChar);
        }
        return result.toString();
    }

    /**
     * Repeat a string n times.
     *
     * @param str String to repeat
     * @param times Number of repetitions
     * @return Repeated string
     */
    public static String repeat(String str, int times) {
        // Inefficient: string concatenation in loop
        String result = "";
        for (int i = 0; i < times; i++) {
            result = result + str;
        }
        return result;
    }
}
