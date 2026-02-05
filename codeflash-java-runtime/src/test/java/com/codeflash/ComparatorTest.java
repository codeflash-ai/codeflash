package com.codeflash;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Comparator.
 */
@DisplayName("Comparator Tests")
class ComparatorTest {

    @Nested
    @DisplayName("Primitive Comparison")
    class PrimitiveTests {

        @Test
        @DisplayName("integers: exact match")
        void testIntegers() {
            assertTrue(Comparator.compare(42, 42));
            assertFalse(Comparator.compare(42, 43));
        }

        @Test
        @DisplayName("longs: exact match")
        void testLongs() {
            assertTrue(Comparator.compare(Long.MAX_VALUE, Long.MAX_VALUE));
            assertFalse(Comparator.compare(1L, 2L));
        }

        @Test
        @DisplayName("doubles: epsilon tolerance")
        void testDoubleEpsilon() {
            // Within epsilon - should be equal
            assertTrue(Comparator.compare(1.0, 1.0 + 1e-10));
            assertTrue(Comparator.compare(3.14159, 3.14159 + 1e-12));

            // Outside epsilon - should not be equal
            assertFalse(Comparator.compare(1.0, 1.1));
            assertFalse(Comparator.compare(1.0, 1.0 + 1e-8));
        }

        @Test
        @DisplayName("floats: epsilon tolerance")
        void testFloatEpsilon() {
            assertTrue(Comparator.compare(1.0f, 1.0f + 1e-10f));
            assertFalse(Comparator.compare(1.0f, 1.1f));
        }

        @Test
        @DisplayName("NaN: should equal NaN")
        void testNaN() {
            assertTrue(Comparator.compare(Double.NaN, Double.NaN));
            assertTrue(Comparator.compare(Float.NaN, Float.NaN));
        }

        @Test
        @DisplayName("Infinity: same sign should be equal")
        void testInfinity() {
            assertTrue(Comparator.compare(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY));
            assertTrue(Comparator.compare(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY));
            assertFalse(Comparator.compare(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY));
        }

        @Test
        @DisplayName("booleans: exact match")
        void testBooleans() {
            assertTrue(Comparator.compare(true, true));
            assertTrue(Comparator.compare(false, false));
            assertFalse(Comparator.compare(true, false));
        }

        @Test
        @DisplayName("strings: exact match")
        void testStrings() {
            assertTrue(Comparator.compare("hello", "hello"));
            assertTrue(Comparator.compare("", ""));
            assertFalse(Comparator.compare("hello", "world"));
        }

        @Test
        @DisplayName("characters: exact match")
        void testCharacters() {
            assertTrue(Comparator.compare('a', 'a'));
            assertFalse(Comparator.compare('a', 'b'));
        }
    }

    @Nested
    @DisplayName("Null Handling")
    class NullTests {

        @Test
        @DisplayName("both null: should be equal")
        void testBothNull() {
            assertTrue(Comparator.compare(null, null));
        }

        @Test
        @DisplayName("one null: should not be equal")
        void testOneNull() {
            assertFalse(Comparator.compare(null, "value"));
            assertFalse(Comparator.compare("value", null));
        }
    }

    @Nested
    @DisplayName("Collection Comparison")
    class CollectionTests {

        @Test
        @DisplayName("lists: order matters")
        void testLists() {
            List<Integer> list1 = Arrays.asList(1, 2, 3);
            List<Integer> list2 = Arrays.asList(1, 2, 3);
            List<Integer> list3 = Arrays.asList(3, 2, 1);

            assertTrue(Comparator.compare(list1, list2));
            assertFalse(Comparator.compare(list1, list3));
        }

        @Test
        @DisplayName("lists: different sizes")
        void testListsDifferentSizes() {
            List<Integer> list1 = Arrays.asList(1, 2, 3);
            List<Integer> list2 = Arrays.asList(1, 2);

            assertFalse(Comparator.compare(list1, list2));
        }

        @Test
        @DisplayName("sets: order doesn't matter")
        void testSets() {
            Set<Integer> set1 = new HashSet<>(Arrays.asList(1, 2, 3));
            Set<Integer> set2 = new HashSet<>(Arrays.asList(3, 2, 1));

            assertTrue(Comparator.compare(set1, set2));
        }

        @Test
        @DisplayName("sets: different contents")
        void testSetsDifferentContents() {
            Set<Integer> set1 = new HashSet<>(Arrays.asList(1, 2, 3));
            Set<Integer> set2 = new HashSet<>(Arrays.asList(1, 2, 4));

            assertFalse(Comparator.compare(set1, set2));
        }

        @Test
        @DisplayName("empty collections: should be equal")
        void testEmptyCollections() {
            assertTrue(Comparator.compare(new ArrayList<>(), new ArrayList<>()));
            assertTrue(Comparator.compare(new HashSet<>(), new HashSet<>()));
        }

        @Test
        @DisplayName("nested collections")
        void testNestedCollections() {
            List<List<Integer>> nested1 = Arrays.asList(
                Arrays.asList(1, 2),
                Arrays.asList(3, 4)
            );
            List<List<Integer>> nested2 = Arrays.asList(
                Arrays.asList(1, 2),
                Arrays.asList(3, 4)
            );

            assertTrue(Comparator.compare(nested1, nested2));
        }
    }

    @Nested
    @DisplayName("Map Comparison")
    class MapTests {

        @Test
        @DisplayName("maps: same contents")
        void testMaps() {
            Map<String, Integer> map1 = new HashMap<>();
            map1.put("one", 1);
            map1.put("two", 2);

            Map<String, Integer> map2 = new HashMap<>();
            map2.put("two", 2);
            map2.put("one", 1);

            assertTrue(Comparator.compare(map1, map2));
        }

        @Test
        @DisplayName("maps: different values")
        void testMapsDifferentValues() {
            Map<String, Integer> map1 = Map.of("key", 1);
            Map<String, Integer> map2 = Map.of("key", 2);

            assertFalse(Comparator.compare(map1, map2));
        }

        @Test
        @DisplayName("maps: different keys")
        void testMapsDifferentKeys() {
            Map<String, Integer> map1 = Map.of("key1", 1);
            Map<String, Integer> map2 = Map.of("key2", 1);

            assertFalse(Comparator.compare(map1, map2));
        }

        @Test
        @DisplayName("maps: different sizes")
        void testMapsDifferentSizes() {
            Map<String, Integer> map1 = Map.of("one", 1, "two", 2);
            Map<String, Integer> map2 = Map.of("one", 1);

            assertFalse(Comparator.compare(map1, map2));
        }

        @Test
        @DisplayName("nested maps")
        void testNestedMaps() {
            Map<String, Object> map1 = new HashMap<>();
            map1.put("inner", Map.of("key", "value"));

            Map<String, Object> map2 = new HashMap<>();
            map2.put("inner", Map.of("key", "value"));

            assertTrue(Comparator.compare(map1, map2));
        }
    }

    @Nested
    @DisplayName("Array Comparison")
    class ArrayTests {

        @Test
        @DisplayName("int arrays: element-wise comparison")
        void testIntArrays() {
            int[] arr1 = {1, 2, 3};
            int[] arr2 = {1, 2, 3};
            int[] arr3 = {1, 2, 4};

            assertTrue(Comparator.compare(arr1, arr2));
            assertFalse(Comparator.compare(arr1, arr3));
        }

        @Test
        @DisplayName("object arrays: element-wise comparison")
        void testObjectArrays() {
            String[] arr1 = {"a", "b", "c"};
            String[] arr2 = {"a", "b", "c"};

            assertTrue(Comparator.compare(arr1, arr2));
        }

        @Test
        @DisplayName("arrays: different lengths")
        void testArraysDifferentLengths() {
            int[] arr1 = {1, 2, 3};
            int[] arr2 = {1, 2};

            assertFalse(Comparator.compare(arr1, arr2));
        }
    }

    @Nested
    @DisplayName("Exception Comparison")
    class ExceptionTests {

        @Test
        @DisplayName("same exception type and message: equal")
        void testSameException() {
            Exception e1 = new IllegalArgumentException("test");
            Exception e2 = new IllegalArgumentException("test");

            assertTrue(Comparator.compare(e1, e2));
        }

        @Test
        @DisplayName("different exception types: not equal")
        void testDifferentExceptionTypes() {
            Exception e1 = new IllegalArgumentException("test");
            Exception e2 = new IllegalStateException("test");

            assertFalse(Comparator.compare(e1, e2));
        }

        @Test
        @DisplayName("different messages: not equal")
        void testDifferentMessages() {
            Exception e1 = new RuntimeException("message 1");
            Exception e2 = new RuntimeException("message 2");

            assertFalse(Comparator.compare(e1, e2));
        }

        @Test
        @DisplayName("both null messages: equal")
        void testBothNullMessages() {
            Exception e1 = new RuntimeException((String) null);
            Exception e2 = new RuntimeException((String) null);

            assertTrue(Comparator.compare(e1, e2));
        }
    }

    @Nested
    @DisplayName("Placeholder Rejection")
    class PlaceholderTests {

        @Test
        @DisplayName("original contains placeholder: throws exception")
        void testOriginalPlaceholder() {
            KryoPlaceholder placeholder = new KryoPlaceholder(
                "java.net.Socket", "<socket>", "error", "path"
            );

            assertThrows(KryoPlaceholderAccessException.class, () -> {
                Comparator.compare(placeholder, "anything");
            });
        }

        @Test
        @DisplayName("new contains placeholder: throws exception")
        void testNewPlaceholder() {
            KryoPlaceholder placeholder = new KryoPlaceholder(
                "java.net.Socket", "<socket>", "error", "path"
            );

            assertThrows(KryoPlaceholderAccessException.class, () -> {
                Comparator.compare("anything", placeholder);
            });
        }

        @Test
        @DisplayName("placeholder in nested structure: throws exception")
        void testNestedPlaceholder() {
            KryoPlaceholder placeholder = new KryoPlaceholder(
                "java.net.Socket", "<socket>", "error", "data.socket"
            );

            Map<String, Object> map1 = new HashMap<>();
            map1.put("socket", placeholder);

            Map<String, Object> map2 = new HashMap<>();
            map2.put("socket", "different");

            assertThrows(KryoPlaceholderAccessException.class, () -> {
                Comparator.compare(map1, map2);
            });
        }

        @Test
        @DisplayName("compareWithDetails captures error message")
        void testCompareWithDetails() {
            KryoPlaceholder placeholder = new KryoPlaceholder(
                "java.net.Socket", "<socket>", "error", "path"
            );

            Comparator.ComparisonResult result =
                Comparator.compareWithDetails(placeholder, "anything");

            assertFalse(result.isEqual());
            assertTrue(result.hasError());
            assertNotNull(result.getErrorMessage());
        }
    }

    @Nested
    @DisplayName("Custom Objects")
    class CustomObjectTests {

        @Test
        @DisplayName("objects with same field values: equal")
        void testSameFields() {
            TestObj obj1 = new TestObj("name", 42);
            TestObj obj2 = new TestObj("name", 42);

            assertTrue(Comparator.compare(obj1, obj2));
        }

        @Test
        @DisplayName("objects with different field values: not equal")
        void testDifferentFields() {
            TestObj obj1 = new TestObj("name", 42);
            TestObj obj2 = new TestObj("name", 43);

            assertFalse(Comparator.compare(obj1, obj2));
        }

        @Test
        @DisplayName("nested objects")
        void testNestedObjects() {
            TestNested nested1 = new TestNested(new TestObj("inner", 1));
            TestNested nested2 = new TestNested(new TestObj("inner", 1));

            assertTrue(Comparator.compare(nested1, nested2));
        }
    }

    @Nested
    @DisplayName("Type Compatibility")
    class TypeCompatibilityTests {

        @Test
        @DisplayName("different list implementations: compatible")
        void testDifferentListTypes() {
            List<Integer> arrayList = new ArrayList<>(Arrays.asList(1, 2, 3));
            List<Integer> linkedList = new LinkedList<>(Arrays.asList(1, 2, 3));

            assertTrue(Comparator.compare(arrayList, linkedList));
        }

        @Test
        @DisplayName("different map implementations: compatible")
        void testDifferentMapTypes() {
            Map<String, Integer> hashMap = new HashMap<>();
            hashMap.put("key", 1);

            Map<String, Integer> linkedHashMap = new LinkedHashMap<>();
            linkedHashMap.put("key", 1);

            assertTrue(Comparator.compare(hashMap, linkedHashMap));
        }

        @Test
        @DisplayName("incompatible types: not equal")
        void testIncompatibleTypes() {
            assertFalse(Comparator.compare("string", 42));
            assertFalse(Comparator.compare(new ArrayList<>(), new HashMap<>()));
        }
    }

    @Nested
    @DisplayName("Optional Comparison")
    class OptionalTests {

        @Test
        @DisplayName("both empty: equal")
        void testBothEmpty() {
            assertTrue(Comparator.compare(Optional.empty(), Optional.empty()));
        }

        @Test
        @DisplayName("both present with same value: equal")
        void testBothPresentSame() {
            assertTrue(Comparator.compare(Optional.of("value"), Optional.of("value")));
        }

        @Test
        @DisplayName("one empty, one present: not equal")
        void testOneEmpty() {
            assertFalse(Comparator.compare(Optional.empty(), Optional.of("value")));
            assertFalse(Comparator.compare(Optional.of("value"), Optional.empty()));
        }

        @Test
        @DisplayName("both present with different values: not equal")
        void testDifferentValues() {
            assertFalse(Comparator.compare(Optional.of("a"), Optional.of("b")));
        }
    }

    @Nested
    @DisplayName("Enum Comparison")
    class EnumTests {

        @Test
        @DisplayName("same enum values: equal")
        void testSameEnum() {
            assertTrue(Comparator.compare(TestEnum.A, TestEnum.A));
        }

        @Test
        @DisplayName("different enum values: not equal")
        void testDifferentEnum() {
            assertFalse(Comparator.compare(TestEnum.A, TestEnum.B));
        }
    }

    // Test helper classes

    static class TestObj {
        String name;
        int value;

        TestObj(String name, int value) {
            this.name = name;
            this.value = value;
        }
    }

    static class TestNested {
        TestObj inner;

        TestNested(TestObj inner) {
            this.inner = inner;
        }
    }

    enum TestEnum {
        A, B, C
    }
}
