package com.example;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Data processing class with complex methods to optimize.
 */
public class DataProcessor {

    /**
     * Find duplicate elements in a list.
     *
     * @param list List to check for duplicates
     * @param <T> Type of elements
     * @return List of duplicate elements
     */
    public static <T> List<T> findDuplicates(List<T> list) {
        List<T> duplicates = new ArrayList<>();
        if (list == null) {
            return duplicates;
        }
        // Inefficient: O(n^2) nested loop
        for (int i = 0; i < list.size(); i++) {
            for (int j = i + 1; j < list.size(); j++) {
                if (list.get(i).equals(list.get(j)) && !duplicates.contains(list.get(i))) {
                    duplicates.add(list.get(i));
                }
            }
        }
        return duplicates;
    }

    /**
     * Group elements by a key function.
     *
     * @param list List to group
     * @param keyExtractor Function to extract key from element
     * @param <T> Type of elements
     * @param <K> Type of key
     * @return Map of key to list of elements
     */
    public static <T, K> Map<K, List<T>> groupBy(List<T> list, java.util.function.Function<T, K> keyExtractor) {
        Map<K, List<T>> result = new HashMap<>();
        if (list == null) {
            return result;
        }
        // Could use streams, but explicit loop for optimization opportunity
        for (T item : list) {
            K key = keyExtractor.apply(item);
            if (!result.containsKey(key)) {
                result.put(key, new ArrayList<>());
            }
            result.get(key).add(item);
        }
        return result;
    }

    /**
     * Find intersection of two lists.
     *
     * @param list1 First list
     * @param list2 Second list
     * @param <T> Type of elements
     * @return List of common elements
     */
    public static <T> List<T> intersection(List<T> list1, List<T> list2) {
        List<T> result = new ArrayList<>();
        if (list1 == null || list2 == null) {
            return result;
        }
        // Inefficient: O(n*m) nested loop
        for (T item : list1) {
            if (list2.contains(item) && !result.contains(item)) {
                result.add(item);
            }
        }
        return result;
    }

    /**
     * Flatten a nested list structure.
     *
     * @param nestedList List of lists
     * @param <T> Type of elements
     * @return Flattened list
     */
    public static <T> List<T> flatten(List<List<T>> nestedList) {
        List<T> result = new ArrayList<>();
        if (nestedList == null) {
            return result;
        }
        // Simple but could be optimized with capacity hints
        for (List<T> innerList : nestedList) {
            if (innerList != null) {
                result.addAll(innerList);
            }
        }
        return result;
    }

    /**
     * Count frequency of each element.
     *
     * @param list List to count
     * @param <T> Type of elements
     * @return Map of element to frequency
     */
    public static <T> Map<T, Integer> countFrequency(List<T> list) {
        Map<T, Integer> frequency = new HashMap<>();
        if (list == null) {
            return frequency;
        }
        for (T item : list) {
            // Inefficient: could use merge or compute
            if (frequency.containsKey(item)) {
                frequency.put(item, frequency.get(item) + 1);
            } else {
                frequency.put(item, 1);
            }
        }
        return frequency;
    }

    /**
     * Find the nth most frequent element.
     *
     * @param list List to search
     * @param n Position (1-based)
     * @param <T> Type of elements
     * @return nth most frequent element, or null if not found
     */
    public static <T> T nthMostFrequent(List<T> list, int n) {
        if (list == null || list.isEmpty() || n < 1) {
            return null;
        }
        Map<T, Integer> frequency = countFrequency(list);

        // Inefficient: sort all entries to find nth
        List<Map.Entry<T, Integer>> entries = new ArrayList<>(frequency.entrySet());
        entries.sort((e1, e2) -> e2.getValue().compareTo(e1.getValue()));

        if (n > entries.size()) {
            return null;
        }
        return entries.get(n - 1).getKey();
    }

    /**
     * Partition list into chunks of specified size.
     *
     * @param list List to partition
     * @param chunkSize Size of each chunk
     * @param <T> Type of elements
     * @return List of chunks
     */
    public static <T> List<List<T>> partition(List<T> list, int chunkSize) {
        List<List<T>> result = new ArrayList<>();
        if (list == null || chunkSize <= 0) {
            return result;
        }
        // Inefficient: creates sublists with copying
        for (int i = 0; i < list.size(); i += chunkSize) {
            int end = Math.min(i + chunkSize, list.size());
            result.add(new ArrayList<>(list.subList(i, end)));
        }
        return result;
    }
}
