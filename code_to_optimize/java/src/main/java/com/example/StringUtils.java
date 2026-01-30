package com.example;

import java.util.ArrayList;
import java.util.List;

/**
 * String utility functions.
 */
public class StringUtils {

    /**
     * Reverse a string character by character.
     *
     * @param s String to reverse
     * @return Reversed string
     */
    public static String reverseString(String s) {
        if (s == null || s.isEmpty()) {
            return s;
        }

        String result = "";
        for (int i = s.length() - 1; i >= 0; i--) {
            result = result + s.charAt(i);
        }
        return result;
    }

    /**
     * Check if a string is a palindrome.
     *
     * @param s String to check
     * @return true if s is a palindrome
     */
    public static boolean isPalindrome(String s) {
        if (s == null || s.isEmpty()) {
            return true;
        }

        String reversed = reverseString(s);
        return s.equals(reversed);
    }

    /**
     * Count the number of words in a string.
     *
     * @param s String to count words in
     * @return Number of words
     */
    public static int countWords(String s) {
        if (s == null || s.trim().isEmpty()) {
            return 0;
        }

        String[] words = s.trim().split("\\s+");
        return words.length;
    }

    /**
     * Capitalize the first letter of each word.
     *
     * @param s String to capitalize
     * @return String with each word capitalized
     */
    public static String capitalizeWords(String s) {
        if (s == null || s.isEmpty()) {
            return s;
        }

        String[] words = s.split(" ");
        String result = "";

        for (int i = 0; i < words.length; i++) {
            if (words[i].length() > 0) {
                String capitalized = words[i].substring(0, 1).toUpperCase()
                        + words[i].substring(1).toLowerCase();
                result = result + capitalized;
            }
            if (i < words.length - 1) {
                result = result + " ";
            }
        }

        return result;
    }

    /**
     * Count occurrences of a substring in a string.
     *
     * @param s String to search in
     * @param sub Substring to count
     * @return Number of occurrences
     */
    public static int countOccurrences(String s, String sub) {
        if (s == null || sub == null || sub.isEmpty()) {
            return 0;
        }

        int count = 0;
        int index = 0;

        while ((index = s.indexOf(sub, index)) != -1) {
            count++;
            index = index + 1;
        }

        return count;
    }

    /**
     * Remove all whitespace from a string.
     *
     * @param s String to process
     * @return String without whitespace
     */
    public static String removeWhitespace(String s) {
        if (s == null || s.isEmpty()) {
            return s;
        }

        String result = "";
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (!Character.isWhitespace(c)) {
                result = result + c;
            }
        }
        return result;
    }

    /**
     * Find all indices where a character appears in a string.
     *
     * @param s String to search
     * @param c Character to find
     * @return List of indices where character appears
     */
    public static List<Integer> findAllIndices(String s, char c) {
        List<Integer> indices = new ArrayList<>();

        if (s == null || s.isEmpty()) {
            return indices;
        }

        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == c) {
                indices.add(i);
            }
        }

        return indices;
    }

    /**
     * Check if a string contains only digits.
     *
     * @param s String to check
     * @return true if string contains only digits
     */
    public static boolean isNumeric(String s) {
        if (s == null || s.isEmpty()) {
            return false;
        }

        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c < '0' || c > '9') {
                return false;
            }
        }
        return true;
    }

    /**
     * Repeat a string n times.
     *
     * @param s String to repeat
     * @param n Number of times to repeat
     * @return Repeated string
     */
    public static String repeat(String s, int n) {
        if (s == null || n <= 0) {
            return "";
        }

        String result = "";
        for (int i = 0; i < n; i++) {
            result = result + s;
        }
        return result;
    }

    /**
     * Truncate a string to a maximum length with ellipsis.
     *
     * @param s String to truncate
     * @param maxLength Maximum length (including ellipsis)
     * @return Truncated string
     */
    public static String truncate(String s, int maxLength) {
        if (s == null || maxLength <= 0) {
            return "";
        }

        if (s.length() <= maxLength) {
            return s;
        }

        if (maxLength <= 3) {
            return s.substring(0, maxLength);
        }

        return s.substring(0, maxLength - 3) + "...";
    }

    /**
     * Convert a string to title case.
     *
     * @param s String to convert
     * @return Title case string
     */
    public static String toTitleCase(String s) {
        if (s == null || s.isEmpty()) {
            return s;
        }

        return s.substring(0, 1).toUpperCase() + s.substring(1).toLowerCase();
    }
}
