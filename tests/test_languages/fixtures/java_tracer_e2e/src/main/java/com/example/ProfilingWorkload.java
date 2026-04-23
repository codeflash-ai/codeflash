package com.example;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Profiling workload for benchmarking the codeflash tracing agent.
 * Exercises different argument types to stress serialization paths.
 */
public class ProfilingWorkload {

    // 1. Primitives only — cheapest to serialize
    public static int addInts(int a, int b) {
        return a + b;
    }

    // 2. String arguments — moderate serialization cost
    public static String concatStrings(String a, String b) {
        return a + b;
    }

    // 3. Array argument — requires element-by-element serialization
    public static int sumArray(int[] values) {
        int sum = 0;
        for (int v : values) {
            sum += v;
        }
        return sum;
    }

    // 4. Collection argument — triggers recursive Kryo processing
    public static int sumList(List<Integer> values) {
        int sum = 0;
        for (int v : values) {
            sum += v;
        }
        return sum;
    }

    // 5. Nested map — deep object graph, expensive serialization
    public static int countMapEntries(Map<String, List<Integer>> data) {
        int count = 0;
        for (List<Integer> list : data.values()) {
            count += list.size();
        }
        return count;
    }

    public static void main(String[] args) {
        int iterations = 1000;

        // 1. Primitives
        for (int i = 0; i < iterations; i++) {
            addInts(i, i + 1);
        }

        // 2. Strings
        for (int i = 0; i < iterations; i++) {
            concatStrings("hello-" + i, "-world");
        }

        // 3. Arrays
        int[] arr = new int[100];
        for (int i = 0; i < arr.length; i++) arr[i] = i;
        for (int i = 0; i < iterations; i++) {
            sumArray(arr);
        }

        // 4. Lists
        List<Integer> list = new ArrayList<>(100);
        for (int i = 0; i < 100; i++) list.add(i);
        for (int i = 0; i < iterations; i++) {
            sumList(list);
        }

        // 5. Nested maps
        Map<String, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < 10; i++) {
            List<Integer> vals = new ArrayList<>();
            for (int j = 0; j < 10; j++) vals.add(j);
            map.put("key-" + i, vals);
        }
        for (int i = 0; i < iterations; i++) {
            countMapEntries(map);
        }

        System.out.println("ProfilingWorkload complete.");
    }
}
