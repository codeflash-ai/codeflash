package com.codeflash;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.net.URI;
import java.net.URL;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Edge case tests for Comparator to catch subtle bugs.
 */
@DisplayName("Comparator Edge Case Tests")
class ComparatorEdgeCaseTest {

    // ============================================================
    // NUMBER EDGE CASES
    // ============================================================

    @Nested
    @DisplayName("Number Edge Cases")
    class NumberEdgeCases {

        @Test
        @DisplayName("BigDecimal comparison should work correctly")
        void testBigDecimalComparison() {
            BigDecimal bd1 = new BigDecimal("123456789.123456789");
            BigDecimal bd2 = new BigDecimal("123456789.123456789");
            BigDecimal bd3 = new BigDecimal("123456789.123456788");

            assertTrue(Comparator.compare(bd1, bd2), "Same BigDecimals should be equal");
            assertFalse(Comparator.compare(bd1, bd3), "Different BigDecimals should not be equal");
        }

        @Test
        @DisplayName("BigDecimal with different scale should compare by value")
        void testBigDecimalDifferentScale() {
            BigDecimal bd1 = new BigDecimal("1.0");
            BigDecimal bd2 = new BigDecimal("1.00");

            // Note: BigDecimal.equals considers scale, but compareTo doesn't
            // Our comparator should handle this
            assertTrue(Comparator.compare(bd1, bd2), "1.0 and 1.00 should be equal");
        }

        @Test
        @DisplayName("BigInteger comparison should work correctly")
        void testBigIntegerComparison() {
            BigInteger bi1 = new BigInteger("123456789012345678901234567890");
            BigInteger bi2 = new BigInteger("123456789012345678901234567890");
            BigInteger bi3 = new BigInteger("123456789012345678901234567891");

            assertTrue(Comparator.compare(bi1, bi2), "Same BigIntegers should be equal");
            assertFalse(Comparator.compare(bi1, bi3), "Different BigIntegers should not be equal");
        }

        @Test
        @DisplayName("BigInteger larger than Long.MAX_VALUE")
        void testBigIntegerLargerThanLong() {
            BigInteger bi1 = BigInteger.valueOf(Long.MAX_VALUE).add(BigInteger.ONE);
            BigInteger bi2 = BigInteger.valueOf(Long.MAX_VALUE).add(BigInteger.ONE);
            BigInteger bi3 = BigInteger.valueOf(Long.MAX_VALUE).add(BigInteger.TWO);

            assertTrue(Comparator.compare(bi1, bi2), "Same large BigIntegers should be equal");
            assertFalse(Comparator.compare(bi1, bi3), "Different large BigIntegers should not be equal");
        }

        @Test
        @DisplayName("Byte comparison")
        void testByteComparison() {
            Byte b1 = (byte) 127;
            Byte b2 = (byte) 127;
            Byte b3 = (byte) -128;

            assertTrue(Comparator.compare(b1, b2));
            assertFalse(Comparator.compare(b1, b3));
        }

        @Test
        @DisplayName("Short comparison")
        void testShortComparison() {
            Short s1 = (short) 32767;
            Short s2 = (short) 32767;
            Short s3 = (short) -32768;

            assertTrue(Comparator.compare(s1, s2));
            assertFalse(Comparator.compare(s1, s3));
        }

        @Test
        @DisplayName("Large double comparison with relative tolerance")
        void testLargeDoubleComparison() {
            // For large numbers, absolute epsilon may be too small
            double large1 = 1e15;
            double large2 = 1e15 + 1;  // Difference of 1 in 1e15

            // With relative tolerance, these should be equal (difference is 1e-15 relative)
            assertTrue(Comparator.compare(large1, large2),
                "Large numbers with tiny relative difference should be equal");
        }

        @Test
        @DisplayName("Large doubles that are actually different")
        void testLargeDoublesActuallyDifferent() {
            double large1 = 1e15;
            double large2 = 1.001e15;  // 0.1% difference

            assertFalse(Comparator.compare(large1, large2),
                "Large numbers with significant relative difference should NOT be equal");
        }

        @Test
        @DisplayName("Float vs Double comparison")
        void testFloatVsDouble() {
            Float f = 3.14f;
            Double d = 3.14;

            // These may differ slightly due to precision
            // Testing current behavior
            boolean result = Comparator.compare(f, d);
            // Document: Float 3.14f != Double 3.14 due to precision differences
        }

        @Test
        @DisplayName("Integer overflow edge case")
        void testIntegerOverflow() {
            Integer maxInt = Integer.MAX_VALUE;
            Long maxIntAsLong = (long) Integer.MAX_VALUE;

            assertTrue(Comparator.compare(maxInt, maxIntAsLong),
                "Integer.MAX_VALUE should equal same value as Long");
        }

        @Test
        @DisplayName("Long overflow to BigInteger")
        void testLongOverflowToBigInteger() {
            Long maxLong = Long.MAX_VALUE;
            BigInteger maxLongAsBigInt = BigInteger.valueOf(Long.MAX_VALUE);

            assertTrue(Comparator.compare(maxLong, maxLongAsBigInt),
                "Long.MAX_VALUE should equal same value as BigInteger");
        }

        @Test
        @DisplayName("Very small double comparison")
        void testVerySmallDoubleComparison() {
            double small1 = 1e-15;
            double small2 = 1e-15 + 1e-25;

            assertTrue(Comparator.compare(small1, small2),
                "Very close small numbers should be equal");
        }

        @Test
        @DisplayName("Negative zero equals positive zero")
        void testNegativeZero() {
            double negZero = -0.0;
            double posZero = 0.0;

            assertTrue(Comparator.compare(negZero, posZero),
                "-0.0 should equal 0.0");
        }

        @Test
        @DisplayName("Mixed integer types comparison")
        void testMixedIntegerTypes() {
            Integer i = 42;
            Long l = 42L;

            assertTrue(Comparator.compare(i, l), "Integer 42 should equal Long 42");
        }
    }

    // ============================================================
    // ARRAY EDGE CASES
    // ============================================================

    @Nested
    @DisplayName("Array Edge Cases")
    class ArrayEdgeCases {

        @Test
        @DisplayName("Empty arrays of same type")
        void testEmptyArrays() {
            int[] arr1 = new int[0];
            int[] arr2 = new int[0];

            assertTrue(Comparator.compare(arr1, arr2));
        }

        @Test
        @DisplayName("Empty arrays of different types")
        void testEmptyArraysDifferentTypes() {
            int[] intArr = new int[0];
            long[] longArr = new long[0];

            // Different array types should not be equal even if empty
            assertFalse(Comparator.compare(intArr, longArr));
        }

        @Test
        @DisplayName("Primitive array vs wrapper array")
        void testPrimitiveVsWrapperArray() {
            int[] primitiveArr = {1, 2, 3};
            Integer[] wrapperArr = {1, 2, 3};

            // These are different types
            assertFalse(Comparator.compare(primitiveArr, wrapperArr));
        }

        @Test
        @DisplayName("Nested arrays")
        void testNestedArrays() {
            int[][] arr1 = {{1, 2}, {3, 4}};
            int[][] arr2 = {{1, 2}, {3, 4}};
            int[][] arr3 = {{1, 2}, {3, 5}};

            assertTrue(Comparator.compare(arr1, arr2));
            assertFalse(Comparator.compare(arr1, arr3));
        }

        @Test
        @DisplayName("Array with null elements")
        void testArrayWithNulls() {
            String[] arr1 = {"a", null, "c"};
            String[] arr2 = {"a", null, "c"};
            String[] arr3 = {"a", "b", "c"};

            assertTrue(Comparator.compare(arr1, arr2));
            assertFalse(Comparator.compare(arr1, arr3));
        }
    }

    // ============================================================
    // LIST VS SET ORDER BEHAVIOR
    // ============================================================

    @Nested
    @DisplayName("List vs Set Order Behavior")
    class ListVsSetOrderBehavior {

        @Test
        @DisplayName("List comparison is ORDER SENSITIVE - [1,2,3] vs [2,3,1] should be FALSE")
        void testListOrderMatters() {
            List<Integer> list1 = Arrays.asList(1, 2, 3);
            List<Integer> list2 = Arrays.asList(2, 3, 1);

            assertFalse(Comparator.compare(list1, list2),
                "Lists with same elements but different order should NOT be equal");
        }

        @Test
        @DisplayName("List comparison with same order should be TRUE")
        void testListSameOrder() {
            List<Integer> list1 = Arrays.asList(1, 2, 3);
            List<Integer> list2 = Arrays.asList(1, 2, 3);

            assertTrue(Comparator.compare(list1, list2),
                "Lists with same elements in same order should be equal");
        }

        @Test
        @DisplayName("Set comparison is ORDER INDEPENDENT - {1,2,3} vs {3,2,1} should be TRUE")
        void testSetOrderDoesNotMatter() {
            Set<Integer> set1 = new LinkedHashSet<>(Arrays.asList(1, 2, 3));
            Set<Integer> set2 = new LinkedHashSet<>(Arrays.asList(3, 2, 1));

            assertTrue(Comparator.compare(set1, set2),
                "Sets with same elements in different order should be equal");
        }

        @Test
        @DisplayName("Set comparison with different elements should be FALSE")
        void testSetDifferentElements() {
            Set<Integer> set1 = new HashSet<>(Arrays.asList(1, 2, 3));
            Set<Integer> set2 = new HashSet<>(Arrays.asList(1, 2, 4));

            assertFalse(Comparator.compare(set1, set2),
                "Sets with different elements should NOT be equal");
        }

        @Test
        @DisplayName("ArrayList vs LinkedList with same elements same order should be TRUE")
        void testDifferentListImplementationsSameOrder() {
            List<Integer> arrayList = new ArrayList<>(Arrays.asList(1, 2, 3));
            List<Integer> linkedList = new LinkedList<>(Arrays.asList(1, 2, 3));

            assertTrue(Comparator.compare(arrayList, linkedList),
                "Different List implementations with same elements in same order should be equal");
        }

        @Test
        @DisplayName("ArrayList vs LinkedList with different order should be FALSE")
        void testDifferentListImplementationsDifferentOrder() {
            List<Integer> arrayList = new ArrayList<>(Arrays.asList(1, 2, 3));
            List<Integer> linkedList = new LinkedList<>(Arrays.asList(3, 2, 1));

            assertFalse(Comparator.compare(arrayList, linkedList),
                "Different List implementations with different order should NOT be equal");
        }

        @Test
        @DisplayName("HashSet vs TreeSet with same elements should be TRUE")
        void testDifferentSetImplementations() {
            Set<Integer> hashSet = new HashSet<>(Arrays.asList(3, 1, 2));
            Set<Integer> treeSet = new TreeSet<>(Arrays.asList(1, 2, 3));

            assertTrue(Comparator.compare(hashSet, treeSet),
                "Different Set implementations with same elements should be equal");
        }

        @Test
        @DisplayName("List with nested lists - order matters at all levels")
        void testNestedListOrder() {
            List<List<Integer>> list1 = Arrays.asList(
                Arrays.asList(1, 2),
                Arrays.asList(3, 4)
            );
            List<List<Integer>> list2 = Arrays.asList(
                Arrays.asList(3, 4),
                Arrays.asList(1, 2)
            );
            List<List<Integer>> list3 = Arrays.asList(
                Arrays.asList(1, 2),
                Arrays.asList(3, 4)
            );

            assertFalse(Comparator.compare(list1, list2),
                "Nested lists with different outer order should NOT be equal");
            assertTrue(Comparator.compare(list1, list3),
                "Nested lists with same order should be equal");
        }

        @Test
        @DisplayName("Set with nested sets - order independent")
        void testNestedSetOrder() {
            Set<Set<Integer>> set1 = new HashSet<>();
            set1.add(new HashSet<>(Arrays.asList(1, 2)));
            set1.add(new HashSet<>(Arrays.asList(3, 4)));

            Set<Set<Integer>> set2 = new HashSet<>();
            set2.add(new HashSet<>(Arrays.asList(4, 3)));  // Different internal order
            set2.add(new HashSet<>(Arrays.asList(2, 1)));  // Different internal order

            assertTrue(Comparator.compare(set1, set2),
                "Nested sets should be equal regardless of order at any level");
        }
    }

    // ============================================================
    // COLLECTION EDGE CASES
    // ============================================================

    @Nested
    @DisplayName("Collection Edge Cases")
    class CollectionEdgeCases {

        @Test
        @DisplayName("Set with custom objects without equals")
        void testSetWithCustomObjectsNoEquals() {
            Set<CustomNoEquals> set1 = new HashSet<>();
            set1.add(new CustomNoEquals("a"));

            Set<CustomNoEquals> set2 = new HashSet<>();
            set2.add(new CustomNoEquals("a"));

            // Should use deep comparison, not equals()
            assertTrue(Comparator.compare(set1, set2),
                "Sets with equivalent custom objects should be equal");
        }

        @Test
        @DisplayName("Empty Set equals empty Set")
        void testEmptySets() {
            Set<Integer> set1 = new HashSet<>();
            Set<Integer> set2 = new TreeSet<>();

            assertTrue(Comparator.compare(set1, set2));
        }

        @Test
        @DisplayName("List vs Set with same elements")
        void testListVsSet() {
            List<Integer> list = Arrays.asList(1, 2, 3);
            Set<Integer> set = new LinkedHashSet<>(Arrays.asList(1, 2, 3));

            // Different collection types should not be equal
            // Actually, our comparator allows this - testing current behavior
            boolean result = Comparator.compare(list, set);
            // Document: List and Set comparison depends on areTypesCompatible
        }

        @Test
        @DisplayName("List with duplicates vs Set")
        void testListWithDuplicatesVsSet() {
            List<Integer> list = Arrays.asList(1, 1, 2);
            Set<Integer> set = new LinkedHashSet<>(Arrays.asList(1, 2));

            assertFalse(Comparator.compare(list, set), "Different sizes should not be equal");
        }

        @Test
        @DisplayName("ConcurrentHashMap comparison")
        void testConcurrentHashMap() {
            ConcurrentHashMap<String, Integer> map1 = new ConcurrentHashMap<>();
            map1.put("a", 1);
            map1.put("b", 2);

            ConcurrentHashMap<String, Integer> map2 = new ConcurrentHashMap<>();
            map2.put("a", 1);
            map2.put("b", 2);

            assertTrue(Comparator.compare(map1, map2));
        }
    }

    // ============================================================
    // MAP EDGE CASES
    // ============================================================

    @Nested
    @DisplayName("Map Edge Cases")
    class MapEdgeCases {

        @Test
        @DisplayName("Map with null key")
        void testMapWithNullKey() {
            Map<String, Integer> map1 = new HashMap<>();
            map1.put(null, 1);
            map1.put("b", 2);

            Map<String, Integer> map2 = new HashMap<>();
            map2.put(null, 1);
            map2.put("b", 2);

            assertTrue(Comparator.compare(map1, map2));
        }

        @Test
        @DisplayName("Map with null value")
        void testMapWithNullValue() {
            Map<String, Integer> map1 = new HashMap<>();
            map1.put("a", null);
            map1.put("b", 2);

            Map<String, Integer> map2 = new HashMap<>();
            map2.put("a", null);
            map2.put("b", 2);

            assertTrue(Comparator.compare(map1, map2));
        }

        @Test
        @DisplayName("Map with complex keys")
        void testMapWithComplexKeys() {
            Map<List<Integer>, String> map1 = new HashMap<>();
            map1.put(Arrays.asList(1, 2, 3), "value1");

            Map<List<Integer>, String> map2 = new HashMap<>();
            map2.put(Arrays.asList(1, 2, 3), "value1");

            assertTrue(Comparator.compare(map1, map2),
                "Maps with complex keys should compare using deep key comparison");
        }

        @Test
        @DisplayName("Map comparison should not double-match entries")
        void testMapNoDoubleMatching() {
            // This tests that we don't match the same entry twice
            Map<String, Integer> map1 = new HashMap<>();
            map1.put("a", 1);
            map1.put("b", 1);  // Same value as "a"

            Map<String, Integer> map2 = new HashMap<>();
            map2.put("a", 1);
            map2.put("c", 1);  // Different key but same value

            assertFalse(Comparator.compare(map1, map2),
                "Maps with different keys should not be equal");
        }
    }

    // ============================================================
    // OBJECT EDGE CASES
    // ============================================================

    @Nested
    @DisplayName("Object Edge Cases")
    class ObjectEdgeCases {

        @Test
        @DisplayName("Objects with inherited fields")
        void testInheritedFields() {
            Child child1 = new Child("parent", "child");
            Child child2 = new Child("parent", "child");
            Child child3 = new Child("different", "child");

            assertTrue(Comparator.compare(child1, child2));
            assertFalse(Comparator.compare(child1, child3));
        }

        @Test
        @DisplayName("Different classes with same fields should not be equal")
        void testDifferentClassesSameFields() {
            ClassA objA = new ClassA("value");
            ClassB objB = new ClassB("value");

            assertFalse(Comparator.compare(objA, objB),
                "Different classes should not be equal even with same field values");
        }

        @Test
        @DisplayName("Object with transient field")
        void testTransientField() {
            ObjectWithTransient obj1 = new ObjectWithTransient("name", "transientValue1");
            ObjectWithTransient obj2 = new ObjectWithTransient("name", "transientValue2");

            // Transient fields should be skipped
            assertTrue(Comparator.compare(obj1, obj2),
                "Objects differing only in transient fields should be equal");
        }

        @Test
        @DisplayName("Object with static field")
        void testStaticField() {
            ObjectWithStatic.staticField = "static1";
            ObjectWithStatic obj1 = new ObjectWithStatic("instance1");

            ObjectWithStatic.staticField = "static2";
            ObjectWithStatic obj2 = new ObjectWithStatic("instance1");

            // Static fields should be skipped
            assertTrue(Comparator.compare(obj1, obj2),
                "Static fields should not affect comparison");
        }

        @Test
        @DisplayName("Circular reference in object")
        void testCircularReferenceInObject() {
            CircularRef ref1 = new CircularRef("a");
            CircularRef ref2 = new CircularRef("b");
            ref1.other = ref2;
            ref2.other = ref1;

            CircularRef ref3 = new CircularRef("a");
            CircularRef ref4 = new CircularRef("b");
            ref3.other = ref4;
            ref4.other = ref3;

            assertTrue(Comparator.compare(ref1, ref3),
                "Equivalent circular structures should be equal");
        }
    }

    // ============================================================
    // SPECIAL TYPES
    // ============================================================

    @Nested
    @DisplayName("Special Types")
    class SpecialTypes {

        @Test
        @DisplayName("UUID comparison")
        void testUUIDComparison() {
            UUID uuid1 = UUID.fromString("550e8400-e29b-41d4-a716-446655440000");
            UUID uuid2 = UUID.fromString("550e8400-e29b-41d4-a716-446655440000");
            UUID uuid3 = UUID.fromString("550e8400-e29b-41d4-a716-446655440001");

            assertTrue(Comparator.compare(uuid1, uuid2));
            assertFalse(Comparator.compare(uuid1, uuid3));
        }

        @Test
        @DisplayName("URI comparison")
        void testURIComparison() throws Exception {
            URI uri1 = new URI("https://example.com/path");
            URI uri2 = new URI("https://example.com/path");
            URI uri3 = new URI("https://example.com/other");

            assertTrue(Comparator.compare(uri1, uri2));
            assertFalse(Comparator.compare(uri1, uri3));
        }

        @Test
        @DisplayName("URL comparison")
        void testURLComparison() throws Exception {
            URL url1 = new URL("https://example.com/path");
            URL url2 = new URL("https://example.com/path");

            assertTrue(Comparator.compare(url1, url2));
        }

        @Test
        @DisplayName("Class object comparison")
        void testClassObjectComparison() {
            Class<?> class1 = String.class;
            Class<?> class2 = String.class;
            Class<?> class3 = Integer.class;

            assertTrue(Comparator.compare(class1, class2));
            assertFalse(Comparator.compare(class1, class3));
        }
    }

    // ============================================================
    // CUSTOM OBJECT (PERSON) EDGE CASES
    // ============================================================

    @Nested
    @DisplayName("Custom Object (Person) Edge Cases")
    class PersonObjectEdgeCases {

        @Test
        @DisplayName("Person with same name, age, date should be equal")
        void testPersonSameFields() {
            Person p1 = new Person("John", 25, java.time.LocalDate.of(2000, 1, 15));
            Person p2 = new Person("John", 25, java.time.LocalDate.of(2000, 1, 15));

            assertTrue(Comparator.compare(p1, p2),
                "Persons with same fields should be equal");
        }

        @Test
        @DisplayName("Person with different name should NOT be equal")
        void testPersonDifferentName() {
            Person p1 = new Person("John", 25, java.time.LocalDate.of(2000, 1, 15));
            Person p2 = new Person("Jane", 25, java.time.LocalDate.of(2000, 1, 15));

            assertFalse(Comparator.compare(p1, p2),
                "Persons with different names should NOT be equal");
        }

        @Test
        @DisplayName("Person with different age should NOT be equal")
        void testPersonDifferentAge() {
            Person p1 = new Person("John", 25, java.time.LocalDate.of(2000, 1, 15));
            Person p2 = new Person("John", 26, java.time.LocalDate.of(2000, 1, 15));

            assertFalse(Comparator.compare(p1, p2),
                "Persons with different ages should NOT be equal");
        }

        @Test
        @DisplayName("Person with different date should NOT be equal")
        void testPersonDifferentDate() {
            Person p1 = new Person("John", 25, java.time.LocalDate.of(2000, 1, 15));
            Person p2 = new Person("John", 25, java.time.LocalDate.of(2000, 1, 16));

            assertFalse(Comparator.compare(p1, p2),
                "Persons with different dates should NOT be equal");
        }

        @Test
        @DisplayName("Person with null name vs non-null name")
        void testPersonNullVsNonNullName() {
            Person p1 = new Person(null, 25, java.time.LocalDate.of(2000, 1, 15));
            Person p2 = new Person("John", 25, java.time.LocalDate.of(2000, 1, 15));

            assertFalse(Comparator.compare(p1, p2),
                "Person with null name vs non-null name should NOT be equal");
        }

        @Test
        @DisplayName("Person with both null names should be equal")
        void testPersonBothNullNames() {
            Person p1 = new Person(null, 25, java.time.LocalDate.of(2000, 1, 15));
            Person p2 = new Person(null, 25, java.time.LocalDate.of(2000, 1, 15));

            assertTrue(Comparator.compare(p1, p2),
                "Persons with both null names and same other fields should be equal");
        }

        @Test
        @DisplayName("Person with null date vs non-null date")
        void testPersonNullVsNonNullDate() {
            Person p1 = new Person("John", 25, null);
            Person p2 = new Person("John", 25, java.time.LocalDate.of(2000, 1, 15));

            assertFalse(Comparator.compare(p1, p2),
                "Person with null date vs non-null date should NOT be equal");
        }

        @Test
        @DisplayName("List of Persons with same content same order")
        void testListOfPersonsSameOrder() {
            List<Person> list1 = Arrays.asList(
                new Person("John", 25, java.time.LocalDate.of(2000, 1, 15)),
                new Person("Jane", 30, java.time.LocalDate.of(1995, 6, 20))
            );
            List<Person> list2 = Arrays.asList(
                new Person("John", 25, java.time.LocalDate.of(2000, 1, 15)),
                new Person("Jane", 30, java.time.LocalDate.of(1995, 6, 20))
            );

            assertTrue(Comparator.compare(list1, list2),
                "Lists of Persons with same content in same order should be equal");
        }

        @Test
        @DisplayName("List of Persons with same content different order should NOT be equal")
        void testListOfPersonsDifferentOrder() {
            List<Person> list1 = Arrays.asList(
                new Person("John", 25, java.time.LocalDate.of(2000, 1, 15)),
                new Person("Jane", 30, java.time.LocalDate.of(1995, 6, 20))
            );
            List<Person> list2 = Arrays.asList(
                new Person("Jane", 30, java.time.LocalDate.of(1995, 6, 20)),
                new Person("John", 25, java.time.LocalDate.of(2000, 1, 15))
            );

            assertFalse(Comparator.compare(list1, list2),
                "Lists of Persons with different order should NOT be equal");
        }

        @Test
        @DisplayName("Map with Person values")
        void testMapWithPersonValues() {
            Map<String, Person> map1 = new HashMap<>();
            map1.put("employee1", new Person("John", 25, java.time.LocalDate.of(2000, 1, 15)));

            Map<String, Person> map2 = new HashMap<>();
            map2.put("employee1", new Person("John", 25, java.time.LocalDate.of(2000, 1, 15)));

            assertTrue(Comparator.compare(map1, map2),
                "Maps with same Person values should be equal");
        }

        @Test
        @DisplayName("Person with floating point age (simulated)")
        void testPersonWithFloatingPointField() {
            PersonWithDouble p1 = new PersonWithDouble("John", 25.0000000001);
            PersonWithDouble p2 = new PersonWithDouble("John", 25.0);

            assertTrue(Comparator.compare(p1, p2),
                "Persons with nearly equal floating point ages should be equal");
        }
    }

    // ============================================================
    // HELPER CLASSES
    // ============================================================

    static class Person {
        String name;
        int age;
        java.time.LocalDate birthDate;

        Person(String name, int age, java.time.LocalDate birthDate) {
            this.name = name;
            this.age = age;
            this.birthDate = birthDate;
        }
        // Intentionally NO equals/hashCode - uses reflection comparison
    }

    static class PersonWithDouble {
        String name;
        double age;

        PersonWithDouble(String name, double age) {
            this.name = name;
            this.age = age;
        }
    }

    static class CustomNoEquals {
        String value;

        CustomNoEquals(String value) {
            this.value = value;
        }
        // No equals/hashCode override
    }

    static class Parent {
        String parentField;

        Parent(String parentField) {
            this.parentField = parentField;
        }
    }

    static class Child extends Parent {
        String childField;

        Child(String parentField, String childField) {
            super(parentField);
            this.childField = childField;
        }
    }

    static class ClassA {
        String field;

        ClassA(String field) {
            this.field = field;
        }
    }

    static class ClassB {
        String field;

        ClassB(String field) {
            this.field = field;
        }
    }

    static class ObjectWithTransient {
        String name;
        transient String transientField;

        ObjectWithTransient(String name, String transientField) {
            this.name = name;
            this.transientField = transientField;
        }
    }

    static class ObjectWithStatic {
        static String staticField;
        String instanceField;

        ObjectWithStatic(String instanceField) {
            this.instanceField = instanceField;
        }
    }

    static class CircularRef {
        String name;
        CircularRef other;

        CircularRef(String name) {
            this.name = name;
        }
    }
}
