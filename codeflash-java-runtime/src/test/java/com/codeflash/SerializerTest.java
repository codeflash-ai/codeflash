package com.codeflash;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Proxy;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the Serializer class.
 */
@DisplayName("Serializer Tests")
class SerializerTest {

    @Nested
    @DisplayName("Primitive Types")
    class PrimitiveTests {

        @Test
        @DisplayName("should serialize integers")
        void testInteger() {
            assertEquals("42", Serializer.toJson(42));
            assertEquals("-1", Serializer.toJson(-1));
            assertEquals("0", Serializer.toJson(0));
        }

        @Test
        @DisplayName("should serialize longs")
        void testLong() {
            assertEquals("9223372036854775807", Serializer.toJson(Long.MAX_VALUE));
        }

        @Test
        @DisplayName("should serialize doubles")
        void testDouble() {
            String json = Serializer.toJson(3.14159);
            assertTrue(json.startsWith("3.14"));
        }

        @Test
        @DisplayName("should serialize booleans")
        void testBoolean() {
            assertEquals("true", Serializer.toJson(true));
            assertEquals("false", Serializer.toJson(false));
        }

        @Test
        @DisplayName("should serialize strings")
        void testString() {
            assertEquals("\"hello\"", Serializer.toJson("hello"));
            assertEquals("\"with \\\"quotes\\\"\"", Serializer.toJson("with \"quotes\""));
        }

        @Test
        @DisplayName("should serialize null")
        void testNull() {
            assertEquals("null", Serializer.toJson((Object) null));
        }

        @Test
        @DisplayName("should serialize characters")
        void testCharacter() {
            assertEquals("\"a\"", Serializer.toJson('a'));
        }
    }

    @Nested
    @DisplayName("Array Types")
    class ArrayTests {

        @Test
        @DisplayName("should serialize int arrays")
        void testIntArray() {
            int[] arr = {1, 2, 3};
            assertEquals("[1,2,3]", Serializer.toJson((Object) arr));
        }

        @Test
        @DisplayName("should serialize String arrays")
        void testStringArray() {
            String[] arr = {"a", "b", "c"};
            assertEquals("[\"a\",\"b\",\"c\"]", Serializer.toJson((Object) arr));
        }

        @Test
        @DisplayName("should serialize empty arrays")
        void testEmptyArray() {
            int[] arr = {};
            assertEquals("[]", Serializer.toJson((Object) arr));
        }
    }

    @Nested
    @DisplayName("Collection Types")
    class CollectionTests {

        @Test
        @DisplayName("should serialize Lists")
        void testList() {
            List<Integer> list = Arrays.asList(1, 2, 3);
            assertEquals("[1,2,3]", Serializer.toJson(list));
        }

        @Test
        @DisplayName("should serialize Sets")
        void testSet() {
            Set<String> set = new LinkedHashSet<>(Arrays.asList("a", "b"));
            String json = Serializer.toJson(set);
            assertTrue(json.contains("\"a\""));
            assertTrue(json.contains("\"b\""));
        }

        @Test
        @DisplayName("should serialize Maps")
        void testMap() {
            Map<String, Integer> map = new LinkedHashMap<>();
            map.put("one", 1);
            map.put("two", 2);
            String json = Serializer.toJson(map);
            assertTrue(json.contains("\"one\":1"));
            assertTrue(json.contains("\"two\":2"));
        }

        @Test
        @DisplayName("should handle nested collections")
        void testNestedCollections() {
            List<List<Integer>> nested = Arrays.asList(
                Arrays.asList(1, 2),
                Arrays.asList(3, 4)
            );
            assertEquals("[[1,2],[3,4]]", Serializer.toJson(nested));
        }
    }

    @Nested
    @DisplayName("Varargs")
    class VarargsTests {

        @Test
        @DisplayName("should serialize multiple arguments")
        void testVarargs() {
            String json = Serializer.toJson(1, "hello", true);
            assertEquals("[1,\"hello\",true]", json);
        }

        @Test
        @DisplayName("should serialize mixed types")
        void testMixedVarargs() {
            String json = Serializer.toJson(42, Arrays.asList(1, 2), null);
            assertTrue(json.startsWith("[42,"));
            assertTrue(json.contains("null"));
        }
    }

    @Nested
    @DisplayName("Custom Objects")
    class CustomObjectTests {

        @Test
        @DisplayName("should serialize simple objects")
        void testSimpleObject() {
            TestPerson person = new TestPerson("John", 30);
            String json = Serializer.toJson(person);

            assertTrue(json.contains("\"name\":\"John\""));
            assertTrue(json.contains("\"age\":30"));
            assertTrue(json.contains("\"__type__\""));
        }

        @Test
        @DisplayName("should serialize nested objects")
        void testNestedObject() {
            TestAddress address = new TestAddress("123 Main St", "NYC");
            TestPersonWithAddress person = new TestPersonWithAddress("Jane", address);
            String json = Serializer.toJson(person);

            assertTrue(json.contains("\"name\":\"Jane\""));
            assertTrue(json.contains("\"city\":\"NYC\""));
        }
    }

    @Nested
    @DisplayName("Exception Serialization")
    class ExceptionTests {

        @Test
        @DisplayName("should serialize exception with type and message")
        void testException() {
            Exception e = new IllegalArgumentException("test error");
            String json = Serializer.exceptionToJson(e);

            assertTrue(json.contains("\"__exception__\":true"));
            assertTrue(json.contains("\"type\":\"java.lang.IllegalArgumentException\""));
            assertTrue(json.contains("\"message\":\"test error\""));
        }

        @Test
        @DisplayName("should include stack trace")
        void testExceptionStackTrace() {
            Exception e = new RuntimeException("test");
            String json = Serializer.exceptionToJson(e);

            assertTrue(json.contains("\"stackTrace\""));
        }

        @Test
        @DisplayName("should include cause")
        void testExceptionWithCause() {
            Exception cause = new NullPointerException("root cause");
            Exception e = new RuntimeException("wrapper", cause);
            String json = Serializer.exceptionToJson(e);

            assertTrue(json.contains("\"causeType\":\"java.lang.NullPointerException\""));
            assertTrue(json.contains("\"causeMessage\":\"root cause\""));
        }
    }

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCaseTests {

        @Test
        @DisplayName("should handle Optional with value")
        void testOptionalPresent() {
            Optional<String> opt = Optional.of("value");
            assertEquals("\"value\"", Serializer.toJson(opt));
        }

        @Test
        @DisplayName("should handle Optional empty")
        void testOptionalEmpty() {
            Optional<String> opt = Optional.empty();
            assertEquals("null", Serializer.toJson(opt));
        }

        @Test
        @DisplayName("should handle enums")
        void testEnum() {
            assertEquals("\"MONDAY\"", Serializer.toJson(java.time.DayOfWeek.MONDAY));
        }

        @Test
        @DisplayName("should handle Date")
        void testDate() {
            Date date = new Date(0); // Epoch
            String json = Serializer.toJson(date);
            assertTrue(json.contains("1970"));
        }
    }

    @Nested
    @DisplayName("Map Key Collision")
    class MapKeyCollisionTests {

        @Test
        @DisplayName("should handle duplicate toString keys without losing data")
        void testDuplicateToStringKeys() {
            Map<Object, String> map = new LinkedHashMap<>();
            map.put(new SameToString("A"), "first");
            map.put(new SameToString("B"), "second");

            String json = Serializer.toJson(map);
            // Both values should be present, not overwritten
            assertTrue(json.contains("first"), "First value should be present, got: " + json);
            assertTrue(json.contains("second"), "Second value should be present, got: " + json);
        }

        @Test
        @DisplayName("should append index to duplicate keys")
        void testDuplicateKeysGetIndex() {
            Map<Object, String> map = new LinkedHashMap<>();
            map.put(new SameToString("A"), "first");
            map.put(new SameToString("B"), "second");
            map.put(new SameToString("C"), "third");

            String json = Serializer.toJson(map);
            // Should have same-key, same-key_1, same-key_2
            assertTrue(json.contains("\"same-key\""), "Original key should be present");
            assertTrue(json.contains("\"same-key_1\""), "First duplicate should have _1 suffix");
            assertTrue(json.contains("\"same-key_2\""), "Second duplicate should have _2 suffix");
        }
    }

    static class SameToString {
        String internalValue;

        SameToString(String value) {
            this.internalValue = value;
        }

        @Override
        public String toString() {
            return "same-key";
        }
    }

    @Nested
    @DisplayName("Class and Proxy Types")
    class ClassAndProxyTests {

        @Test
        @DisplayName("should serialize Class objects cleanly")
        void testClassObject() {
            String json = Serializer.toJson(String.class);
            // Should output just the class name, not internal JVM fields
            assertEquals("\"java.lang.String\"", json);
        }

        @Test
        @DisplayName("should serialize primitive Class objects")
        void testPrimitiveClassObject() {
            String json = Serializer.toJson(int.class);
            assertEquals("\"int\"", json);
        }

        @Test
        @DisplayName("should serialize array Class objects")
        void testArrayClassObject() {
            String json = Serializer.toJson(String[].class);
            assertEquals("\"java.lang.String[]\"", json);
        }

        @Test
        @DisplayName("should handle dynamic proxy")
        void testProxy() {
            Runnable proxy = (Runnable) Proxy.newProxyInstance(
                Runnable.class.getClassLoader(),
                new Class<?>[] { Runnable.class },
                (p, method, args) -> null
            );
            String json = Serializer.toJson(proxy);
            assertNotNull(json);
            // Should indicate it's a proxy cleanly, not dump handler internals or error
            // Current behavior: produces __serialization_error__ due to module access
            assertFalse(json.contains("__serialization_error__"),
                "Proxy should be serialized cleanly, got: " + json);
            assertTrue(json.contains("proxy") || json.contains("Proxy"),
                "Proxy should be identified as such, got: " + json);
        }
    }

    // Test helper classes
    static class TestPerson {
        private final String name;
        private final int age;

        TestPerson(String name, int age) {
            this.name = name;
            this.age = age;
        }
    }

    static class TestAddress {
        private final String street;
        private final String city;

        TestAddress(String street, String city) {
            this.street = street;
            this.city = city;
        }
    }

    static class TestPersonWithAddress {
        private final String name;
        private final TestAddress address;

        TestPersonWithAddress(String name, TestAddress address) {
            this.name = name;
            this.address = address;
        }
    }
}
