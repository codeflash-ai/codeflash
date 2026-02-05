package com.codeflash;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for KryoPlaceholder class.
 */
@DisplayName("KryoPlaceholder Tests")
class KryoPlaceholderTest {

    @Nested
    @DisplayName("Metadata Storage")
    class MetadataTests {

        @Test
        @DisplayName("should store all metadata correctly")
        void testMetadataStorage() {
            KryoPlaceholder placeholder = new KryoPlaceholder(
                "java.net.Socket",
                "<socket instance>",
                "Cannot serialize socket",
                "data.connection.socket"
            );

            assertEquals("java.net.Socket", placeholder.getObjType());
            assertEquals("<socket instance>", placeholder.getObjStr());
            assertEquals("Cannot serialize socket", placeholder.getErrorMsg());
            assertEquals("data.connection.socket", placeholder.getPath());
        }

        @Test
        @DisplayName("should truncate long string representations")
        void testStringTruncation() {
            String longStr = "x".repeat(200);
            KryoPlaceholder placeholder = new KryoPlaceholder(
                "SomeType", longStr, "error", "path"
            );

            assertTrue(placeholder.getObjStr().length() <= 103); // 100 + "..."
            assertTrue(placeholder.getObjStr().endsWith("..."));
        }

        @Test
        @DisplayName("should handle null string representation")
        void testNullStringRepresentation() {
            KryoPlaceholder placeholder = new KryoPlaceholder(
                "SomeType", null, "error", "path"
            );

            assertNull(placeholder.getObjStr());
        }
    }

    @Nested
    @DisplayName("Factory Method")
    class FactoryTests {

        @Test
        @DisplayName("should create placeholder from object")
        void testCreateFromObject() {
            Object obj = new StringBuilder("test");
            KryoPlaceholder placeholder = KryoPlaceholder.create(
                obj, "Cannot serialize", "root"
            );

            assertEquals("java.lang.StringBuilder", placeholder.getObjType());
            assertEquals("test", placeholder.getObjStr());
            assertEquals("Cannot serialize", placeholder.getErrorMsg());
            assertEquals("root", placeholder.getPath());
        }

        @Test
        @DisplayName("should handle null object")
        void testCreateFromNull() {
            KryoPlaceholder placeholder = KryoPlaceholder.create(
                null, "Null object", "path"
            );

            assertEquals("null", placeholder.getObjType());
            assertEquals("null", placeholder.getObjStr());
        }

        @Test
        @DisplayName("should handle object with failing toString")
        void testCreateFromObjectWithBadToString() {
            Object badObj = new Object() {
                @Override
                public String toString() {
                    throw new RuntimeException("toString failed!");
                }
            };

            KryoPlaceholder placeholder = KryoPlaceholder.create(
                badObj, "error", "path"
            );

            assertTrue(placeholder.getObjStr().contains("toString failed"));
        }
    }

    @Nested
    @DisplayName("Serialization")
    class SerializationTests {

        @Test
        @DisplayName("placeholder should be serializable itself")
        void testPlaceholderSerializable() {
            KryoPlaceholder original = new KryoPlaceholder(
                "java.net.Socket",
                "<socket>",
                "Cannot serialize socket",
                "data.socket"
            );

            // Serialize and deserialize the placeholder
            byte[] serialized = KryoSerializer.serialize(original);
            assertNotNull(serialized);
            assertTrue(serialized.length > 0);

            Object deserialized = KryoSerializer.deserialize(serialized);
            assertInstanceOf(KryoPlaceholder.class, deserialized);

            KryoPlaceholder restored = (KryoPlaceholder) deserialized;
            assertEquals(original.getObjType(), restored.getObjType());
            assertEquals(original.getObjStr(), restored.getObjStr());
            assertEquals(original.getErrorMsg(), restored.getErrorMsg());
            assertEquals(original.getPath(), restored.getPath());
        }
    }

    @Nested
    @DisplayName("toString")
    class ToStringTests {

        @Test
        @DisplayName("should produce readable toString")
        void testToString() {
            KryoPlaceholder placeholder = new KryoPlaceholder(
                "java.net.Socket",
                "<socket instance>",
                "error",
                "data.socket"
            );

            String str = placeholder.toString();
            assertTrue(str.contains("KryoPlaceholder"));
            assertTrue(str.contains("java.net.Socket"));
            assertTrue(str.contains("data.socket"));
        }
    }

    @Nested
    @DisplayName("Equality")
    class EqualityTests {

        @Test
        @DisplayName("placeholders with same type and path should be equal")
        void testEquality() {
            KryoPlaceholder p1 = new KryoPlaceholder("Type", "str1", "error1", "path");
            KryoPlaceholder p2 = new KryoPlaceholder("Type", "str2", "error2", "path");

            assertEquals(p1, p2);
            assertEquals(p1.hashCode(), p2.hashCode());
        }

        @Test
        @DisplayName("placeholders with different paths should not be equal")
        void testInequality() {
            KryoPlaceholder p1 = new KryoPlaceholder("Type", "str", "error", "path1");
            KryoPlaceholder p2 = new KryoPlaceholder("Type", "str", "error", "path2");

            assertNotEquals(p1, p2);
        }
    }
}
