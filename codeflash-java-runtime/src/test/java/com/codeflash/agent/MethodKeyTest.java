package com.codeflash.agent;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class MethodKeyTest {

    @Nested
    class EqualityTests {

        @Test
        void equalKeysAreEqual() {
            MethodKey a = new MethodKey("Foo.java", 10, "bar", "com.example.Foo");
            MethodKey b = new MethodKey("Foo.java", 10, "bar", "com.example.Foo");
            assertEquals(a, b);
            assertEquals(a.hashCode(), b.hashCode());
        }

        @Test
        void differentFileNameNotEqual() {
            MethodKey a = new MethodKey("Foo.java", 10, "bar", "com.example.Foo");
            MethodKey b = new MethodKey("Bar.java", 10, "bar", "com.example.Foo");
            assertNotEquals(a, b);
        }

        @Test
        void differentLineNumberNotEqual() {
            MethodKey a = new MethodKey("Foo.java", 10, "bar", "com.example.Foo");
            MethodKey b = new MethodKey("Foo.java", 20, "bar", "com.example.Foo");
            assertNotEquals(a, b);
        }

        @Test
        void differentMethodNameNotEqual() {
            MethodKey a = new MethodKey("Foo.java", 10, "bar", "com.example.Foo");
            MethodKey b = new MethodKey("Foo.java", 10, "baz", "com.example.Foo");
            assertNotEquals(a, b);
        }

        @Test
        void differentClassNameNotEqual() {
            MethodKey a = new MethodKey("Foo.java", 10, "bar", "com.example.Foo");
            MethodKey b = new MethodKey("Foo.java", 10, "bar", "com.example.Bar");
            assertNotEquals(a, b);
        }

        @Test
        void notEqualToNull() {
            MethodKey key = new MethodKey("Foo.java", 10, "bar", "com.example.Foo");
            assertNotEquals(null, key);
        }

        @Test
        void notEqualToOtherType() {
            MethodKey key = new MethodKey("Foo.java", 10, "bar", "com.example.Foo");
            assertNotEquals("not a key", key);
        }
    }

    @Nested
    class GetterTests {

        @Test
        void gettersReturnCorrectValues() {
            MethodKey key = new MethodKey("src/main/java/com/example/Foo.java", 42, "compute", "com.example.Foo");
            assertEquals("src/main/java/com/example/Foo.java", key.getFileName());
            assertEquals(42, key.getLineNumber());
            assertEquals("compute", key.getMethodName());
            assertEquals("com.example.Foo", key.getClassName());
        }
    }

    @Nested
    class HashMapUsageTests {

        @Test
        void worksAsHashMapKey() {
            Map<MethodKey, String> map = new HashMap<>();
            MethodKey key1 = new MethodKey("A.java", 1, "foo", "A");
            MethodKey key2 = new MethodKey("A.java", 1, "foo", "A");
            map.put(key1, "value1");
            assertEquals("value1", map.get(key2));
        }

        @Test
        void distinctKeysAreSeparateInMap() {
            Map<MethodKey, String> map = new HashMap<>();
            MethodKey key1 = new MethodKey("A.java", 1, "foo", "A");
            MethodKey key2 = new MethodKey("A.java", 2, "bar", "A");
            map.put(key1, "first");
            map.put(key2, "second");
            assertEquals(2, map.size());
            assertEquals("first", map.get(key1));
            assertEquals("second", map.get(key2));
        }
    }

    @Nested
    class ToStringTests {

        @Test
        void toStringContainsAllFields() {
            MethodKey key = new MethodKey("Foo.java", 10, "bar", "com.example.Foo");
            String str = key.toString();
            assertTrue(str.contains("com.example.Foo"));
            assertTrue(str.contains("bar"));
            assertTrue(str.contains("Foo.java"));
            assertTrue(str.contains("10"));
        }
    }
}
