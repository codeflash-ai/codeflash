package com.codeflash;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for ObjectComparator.
 */
@DisplayName("ObjectComparator Tests")
class ObjectComparatorTest {

    @Nested
    @DisplayName("Primitive Comparison")
    class PrimitiveTests {

        @Test
        @DisplayName("integers: exact match")
        void testIntegers() {
            assertTrue(ObjectComparator.compare(42, 42));
            assertFalse(ObjectComparator.compare(42, 43));
        }

        @Test
        @DisplayName("longs: exact match")
        void testLongs() {
            assertTrue(ObjectComparator.compare(Long.MAX_VALUE, Long.MAX_VALUE));
            assertFalse(ObjectComparator.compare(1L, 2L));
        }

        @Test
        @DisplayName("doubles: epsilon tolerance")
        void testDoubleEpsilon() {
            // Within epsilon - should be equal
            assertTrue(ObjectComparator.compare(1.0, 1.0 + 1e-10));
            assertTrue(ObjectComparator.compare(3.14159, 3.14159 + 1e-12));

            // Outside epsilon - should not be equal
            assertFalse(ObjectComparator.compare(1.0, 1.1));
            assertFalse(ObjectComparator.compare(1.0, 1.0 + 1e-8));
        }

        @Test
        @DisplayName("floats: epsilon tolerance")
        void testFloatEpsilon() {
            assertTrue(ObjectComparator.compare(1.0f, 1.0f + 1e-10f));
            assertFalse(ObjectComparator.compare(1.0f, 1.1f));
        }

        @Test
        @DisplayName("NaN: should equal NaN")
        void testNaN() {
            assertTrue(ObjectComparator.compare(Double.NaN, Double.NaN));
            assertTrue(ObjectComparator.compare(Float.NaN, Float.NaN));
        }

        @Test
        @DisplayName("Infinity: same sign should be equal")
        void testInfinity() {
            assertTrue(ObjectComparator.compare(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY));
            assertTrue(ObjectComparator.compare(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY));
            assertFalse(ObjectComparator.compare(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY));
        }

        @Test
        @DisplayName("booleans: exact match")
        void testBooleans() {
            assertTrue(ObjectComparator.compare(true, true));
            assertTrue(ObjectComparator.compare(false, false));
            assertFalse(ObjectComparator.compare(true, false));
        }

        @Test
        @DisplayName("strings: exact match")
        void testStrings() {
            assertTrue(ObjectComparator.compare("hello", "hello"));
            assertTrue(ObjectComparator.compare("", ""));
            assertFalse(ObjectComparator.compare("hello", "world"));
        }

        @Test
        @DisplayName("characters: exact match")
        void testCharacters() {
            assertTrue(ObjectComparator.compare('a', 'a'));
            assertFalse(ObjectComparator.compare('a', 'b'));
        }
    }

    @Nested
    @DisplayName("Null Handling")
    class NullTests {

        @Test
        @DisplayName("both null: should be equal")
        void testBothNull() {
            assertTrue(ObjectComparator.compare(null, null));
        }

        @Test
        @DisplayName("one null: should not be equal")
        void testOneNull() {
            assertFalse(ObjectComparator.compare(null, "value"));
            assertFalse(ObjectComparator.compare("value", null));
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

            assertTrue(ObjectComparator.compare(list1, list2));
            assertFalse(ObjectComparator.compare(list1, list3));
        }

        @Test
        @DisplayName("lists: different sizes")
        void testListsDifferentSizes() {
            List<Integer> list1 = Arrays.asList(1, 2, 3);
            List<Integer> list2 = Arrays.asList(1, 2);

            assertFalse(ObjectComparator.compare(list1, list2));
        }

        @Test
        @DisplayName("sets: order doesn't matter")
        void testSets() {
            Set<Integer> set1 = new HashSet<>(Arrays.asList(1, 2, 3));
            Set<Integer> set2 = new HashSet<>(Arrays.asList(3, 2, 1));

            assertTrue(ObjectComparator.compare(set1, set2));
        }

        @Test
        @DisplayName("sets: different contents")
        void testSetsDifferentContents() {
            Set<Integer> set1 = new HashSet<>(Arrays.asList(1, 2, 3));
            Set<Integer> set2 = new HashSet<>(Arrays.asList(1, 2, 4));

            assertFalse(ObjectComparator.compare(set1, set2));
        }

        @Test
        @DisplayName("empty collections: should be equal")
        void testEmptyCollections() {
            assertTrue(ObjectComparator.compare(new ArrayList<>(), new ArrayList<>()));
            assertTrue(ObjectComparator.compare(new HashSet<>(), new HashSet<>()));
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

            assertTrue(ObjectComparator.compare(nested1, nested2));
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

            assertTrue(ObjectComparator.compare(map1, map2));
        }

        @Test
        @DisplayName("maps: different values")
        void testMapsDifferentValues() {
            Map<String, Integer> map1 = Map.of("key", 1);
            Map<String, Integer> map2 = Map.of("key", 2);

            assertFalse(ObjectComparator.compare(map1, map2));
        }

        @Test
        @DisplayName("maps: different keys")
        void testMapsDifferentKeys() {
            Map<String, Integer> map1 = Map.of("key1", 1);
            Map<String, Integer> map2 = Map.of("key2", 1);

            assertFalse(ObjectComparator.compare(map1, map2));
        }

        @Test
        @DisplayName("maps: different sizes")
        void testMapsDifferentSizes() {
            Map<String, Integer> map1 = Map.of("one", 1, "two", 2);
            Map<String, Integer> map2 = Map.of("one", 1);

            assertFalse(ObjectComparator.compare(map1, map2));
        }

        @Test
        @DisplayName("nested maps")
        void testNestedMaps() {
            Map<String, Object> map1 = new HashMap<>();
            map1.put("inner", Map.of("key", "value"));

            Map<String, Object> map2 = new HashMap<>();
            map2.put("inner", Map.of("key", "value"));

            assertTrue(ObjectComparator.compare(map1, map2));
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

            assertTrue(ObjectComparator.compare(arr1, arr2));
            assertFalse(ObjectComparator.compare(arr1, arr3));
        }

        @Test
        @DisplayName("object arrays: element-wise comparison")
        void testObjectArrays() {
            String[] arr1 = {"a", "b", "c"};
            String[] arr2 = {"a", "b", "c"};

            assertTrue(ObjectComparator.compare(arr1, arr2));
        }

        @Test
        @DisplayName("arrays: different lengths")
        void testArraysDifferentLengths() {
            int[] arr1 = {1, 2, 3};
            int[] arr2 = {1, 2};

            assertFalse(ObjectComparator.compare(arr1, arr2));
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

            assertTrue(ObjectComparator.compare(e1, e2));
        }

        @Test
        @DisplayName("different exception types: not equal")
        void testDifferentExceptionTypes() {
            Exception e1 = new IllegalArgumentException("test");
            Exception e2 = new IllegalStateException("test");

            assertFalse(ObjectComparator.compare(e1, e2));
        }

        @Test
        @DisplayName("different messages: not equal")
        void testDifferentMessages() {
            Exception e1 = new RuntimeException("message 1");
            Exception e2 = new RuntimeException("message 2");

            assertFalse(ObjectComparator.compare(e1, e2));
        }

        @Test
        @DisplayName("both null messages: equal")
        void testBothNullMessages() {
            Exception e1 = new RuntimeException((String) null);
            Exception e2 = new RuntimeException((String) null);

            assertTrue(ObjectComparator.compare(e1, e2));
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
                ObjectComparator.compare(placeholder, "anything");
            });
        }

        @Test
        @DisplayName("new contains placeholder: throws exception")
        void testNewPlaceholder() {
            KryoPlaceholder placeholder = new KryoPlaceholder(
                "java.net.Socket", "<socket>", "error", "path"
            );

            assertThrows(KryoPlaceholderAccessException.class, () -> {
                ObjectComparator.compare("anything", placeholder);
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
                ObjectComparator.compare(map1, map2);
            });
        }

        @Test
        @DisplayName("compareWithDetails captures error message")
        void testCompareWithDetails() {
            KryoPlaceholder placeholder = new KryoPlaceholder(
                "java.net.Socket", "<socket>", "error", "path"
            );

            ObjectComparator.ComparisonResult result =
                ObjectComparator.compareWithDetails(placeholder, "anything");

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

            assertTrue(ObjectComparator.compare(obj1, obj2));
        }

        @Test
        @DisplayName("objects with different field values: not equal")
        void testDifferentFields() {
            TestObj obj1 = new TestObj("name", 42);
            TestObj obj2 = new TestObj("name", 43);

            assertFalse(ObjectComparator.compare(obj1, obj2));
        }

        @Test
        @DisplayName("nested objects")
        void testNestedObjects() {
            TestNested nested1 = new TestNested(new TestObj("inner", 1));
            TestNested nested2 = new TestNested(new TestObj("inner", 1));

            assertTrue(ObjectComparator.compare(nested1, nested2));
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

            assertTrue(ObjectComparator.compare(arrayList, linkedList));
        }

        @Test
        @DisplayName("different map implementations: compatible")
        void testDifferentMapTypes() {
            Map<String, Integer> hashMap = new HashMap<>();
            hashMap.put("key", 1);

            Map<String, Integer> linkedHashMap = new LinkedHashMap<>();
            linkedHashMap.put("key", 1);

            assertTrue(ObjectComparator.compare(hashMap, linkedHashMap));
        }

        @Test
        @DisplayName("incompatible types: not equal")
        void testIncompatibleTypes() {
            assertFalse(ObjectComparator.compare("string", 42));
            assertFalse(ObjectComparator.compare(new ArrayList<>(), new HashMap<>()));
        }
    }

    @Nested
    @DisplayName("Optional Comparison")
    class OptionalTests {

        @Test
        @DisplayName("both empty: equal")
        void testBothEmpty() {
            assertTrue(ObjectComparator.compare(Optional.empty(), Optional.empty()));
        }

        @Test
        @DisplayName("both present with same value: equal")
        void testBothPresentSame() {
            assertTrue(ObjectComparator.compare(Optional.of("value"), Optional.of("value")));
        }

        @Test
        @DisplayName("one empty, one present: not equal")
        void testOneEmpty() {
            assertFalse(ObjectComparator.compare(Optional.empty(), Optional.of("value")));
            assertFalse(ObjectComparator.compare(Optional.of("value"), Optional.empty()));
        }

        @Test
        @DisplayName("both present with different values: not equal")
        void testDifferentValues() {
            assertFalse(ObjectComparator.compare(Optional.of("a"), Optional.of("b")));
        }
    }

    @Nested
    @DisplayName("Enum Comparison")
    class EnumTests {

        @Test
        @DisplayName("same enum values: equal")
        void testSameEnum() {
            assertTrue(ObjectComparator.compare(TestEnum.A, TestEnum.A));
        }

        @Test
        @DisplayName("different enum values: not equal")
        void testDifferentEnum() {
            assertFalse(ObjectComparator.compare(TestEnum.A, TestEnum.B));
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
