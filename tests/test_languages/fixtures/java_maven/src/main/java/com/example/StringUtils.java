package com.example;

import java.util.ArrayList;
import java.util.List;

/**
 * String utility class with methods to optimize.
 */
public class StringUtils {

    /**
     * Reverse a string character by character.
     *
     * @param str String to reverse
     * @return Reversed string
     */
    public static String reverse(String str) {
        if (str == null || str.isEmpty()) {
            return str;
        }
        // Inefficient: string concatenation in loop
        String result = "";
        for (int i = str.length() - 1; i >= 0; i--) {
            result = result + str.charAt(i);
        }
        return result;
    }

    /**
     * Check if a string is a palindrome.
     *
     * @param str String to check
     * @return true if palindrome, false otherwise
     */
    public static boolean isPalindrome(String str) {
        if (str == null) {
            return false;
        }
        // Inefficient: creates reversed string instead of comparing in place
        String reversed = reverse(str.toLowerCase().replaceAll("\\s+", ""));
        String cleaned = str.toLowerCase().replaceAll("\\s+", "");
        return cleaned.equals(reversed);
    }

    /**
     * Count occurrences of a substring.
     *
     * @param str String to search in
     * @param sub Substring to find
     * @return Number of occurrences
     */
    public static int countOccurrences(String str, String sub) {
        if (str == null || sub == null || sub.isEmpty()) {
            return 0;
        }
        // Inefficient: creates many intermediate strings
        int count = 0;
        int index = 0;
        while ((index = str.indexOf(sub, index)) != -1) {
            count++;
            index++;
        }
        return count;
    }

    /**
     * Find all anagrams of a word in a text.
     *
     * @param text Text to search in
     * @param word Word to find anagrams of
     * @return List of starting indices of anagrams
     */
    public static List<Integer> findAnagrams(String text, String word) {
        List<Integer> result = new ArrayList<>();
        if (text == null || word == null || text.length() < word.length()) {
            return result;
        }

        // Inefficient: recalculates sorted word for each position
        int wordLen = word.length();
        for (int i = 0; i <= text.length() - wordLen; i++) {
            String window = text.substring(i, i + wordLen);
            if (isAnagram(window, word)) {
                result.add(i);
            }
        }
        return result;
    }

    /**
     * Check if two strings are anagrams.
     *
     * @param s1 First string
     * @param s2 Second string
     * @return true if anagrams, false otherwise
     */
    public static boolean isAnagram(String s1, String s2) {
        if (s1 == null || s2 == null || s1.length() != s2.length()) {
            return false;
        }
        // Inefficient: sorts both strings
        char[] arr1 = s1.toLowerCase().toCharArray();
        char[] arr2 = s2.toLowerCase().toCharArray();
        java.util.Arrays.sort(arr1);
        java.util.Arrays.sort(arr2);
        return java.util.Arrays.equals(arr1, arr2);
    }

    /**
     * Find longest common prefix of an array of strings.
     *
     * @param strings Array of strings
     * @return Longest common prefix
     */
    public static String longestCommonPrefix(String[] strings) {
        if (strings == null || strings.length == 0) {
            return "";
        }
        // Inefficient: vertical scanning approach
        String prefix = strings[0];
        for (int i = 1; i < strings.length; i++) {
            while (strings[i].indexOf(prefix) != 0) {
                prefix = prefix.substring(0, prefix.length() - 1);
                if (prefix.isEmpty()) {
                    return "";
                }
            }
        }
        return prefix;
    }
}
