package com.codeflash;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for KryoSerializer following Python's dill/patcher test patterns.
 *
 * Test pattern: Create object -> Serialize -> Deserialize -> Compare with original
 */
@DisplayName("KryoSerializer Tests")
class KryoSerializerTest {

    @BeforeEach
    void setUp() {
        KryoSerializer.clearUnserializableTypesCache();
    }

    // ============================================================
    // ROUNDTRIP TESTS - Following Python's test patterns
    // ============================================================

    @Nested
    @DisplayName("Roundtrip Tests - Simple Nested Structures")
    class RoundtripSimpleNestedTests {

        @Test
        @DisplayName("simple nested data structure serializes and deserializes correctly")
        void testSimpleNested() {
            Map<String, Object> originalData = new LinkedHashMap<>();
            originalData.put("numbers", Arrays.asList(1, 2, 3));
            Map<String, Object> nestedDict = new LinkedHashMap<>();
            nestedDict.put("key", "value");
            nestedDict.put("another", 42);
            originalData.put("nested_dict", nestedDict);

            byte[] dumped = KryoSerializer.serialize(originalData);
            Object reloaded = KryoSerializer.deserialize(dumped);

            assertTrue(ObjectComparator.compare(originalData, reloaded),
                "Reloaded data should equal original data");
        }

        @Test
        @DisplayName("integers roundtrip correctly")
        void testIntegers() {
            int[] testCases = {5, 0, -1, Integer.MAX_VALUE, Integer.MIN_VALUE};
            for (int original : testCases) {
                byte[] dumped = KryoSerializer.serialize(original);
                Object reloaded = KryoSerializer.deserialize(dumped);
                assertTrue(ObjectComparator.compare(original, reloaded),
                    "Failed for: " + original);
            }
        }

        @Test
        @DisplayName("floats roundtrip correctly with epsilon tolerance")
        void testFloats() {
            double[] testCases = {5.0, 0.0, -1.0, 3.14159, Double.MAX_VALUE};
            for (double original : testCases) {
                byte[] dumped = KryoSerializer.serialize(original);
                Object reloaded = KryoSerializer.deserialize(dumped);
                assertTrue(ObjectComparator.compare(original, reloaded),
                    "Failed for: " + original);
            }
        }

        @Test
        @DisplayName("strings roundtrip correctly")
        void testStrings() {
            String[] testCases = {"Hello", "", "World", "unicode: \u00e9\u00e8"};
            for (String original : testCases) {
                byte[] dumped = KryoSerializer.serialize(original);
                Object reloaded = KryoSerializer.deserialize(dumped);
                assertTrue(ObjectComparator.compare(original, reloaded),
                    "Failed for: " + original);
            }
        }

        @Test
        @DisplayName("lists roundtrip correctly")
        void testLists() {
            List<Integer> original = Arrays.asList(1, 2, 3);
            byte[] dumped = KryoSerializer.serialize(original);
            Object reloaded = KryoSerializer.deserialize(dumped);
            assertTrue(ObjectComparator.compare(original, reloaded));
        }

        @Test
        @DisplayName("maps roundtrip correctly")
        void testMaps() {
            Map<String, Integer> original = new LinkedHashMap<>();
            original.put("a", 1);
            original.put("b", 2);

            byte[] dumped = KryoSerializer.serialize(original);
            Object reloaded = KryoSerializer.deserialize(dumped);
            assertTrue(ObjectComparator.compare(original, reloaded));
        }

        @Test
        @DisplayName("sets roundtrip correctly")
        void testSets() {
            Set<Integer> original = new LinkedHashSet<>(Arrays.asList(1, 2, 3));
            byte[] dumped = KryoSerializer.serialize(original);
            Object reloaded = KryoSerializer.deserialize(dumped);
            assertTrue(ObjectComparator.compare(original, reloaded));
        }

        @Test
        @DisplayName("null roundtrips correctly")
        void testNull() {
            byte[] dumped = KryoSerializer.serialize(null);
            Object reloaded = KryoSerializer.deserialize(dumped);
            assertNull(reloaded);
        }
    }

    // ============================================================
    // UNSERIALIZABLE OBJECT TESTS
    // ============================================================

    @Nested
    @DisplayName("Unserializable Object Tests")
    class UnserializableObjectTests {

        @Test
        @DisplayName("socket replaced by KryoPlaceholder")
        void testSocketReplacedByPlaceholder() throws Exception {
            try (Socket socket = new Socket()) {
                Map<String, Object> dataWithSocket = new LinkedHashMap<>();
                dataWithSocket.put("safe_value", 123);
                dataWithSocket.put("raw_socket", socket);

                byte[] dumped = KryoSerializer.serialize(dataWithSocket);
                Map<?, ?> reloaded = (Map<?, ?>) KryoSerializer.deserialize(dumped);

                assertInstanceOf(Map.class, reloaded);
                assertEquals(123, reloaded.get("safe_value"));
                assertInstanceOf(KryoPlaceholder.class, reloaded.get("raw_socket"));
            }
        }

        @Test
        @DisplayName("database connection replaced by KryoPlaceholder")
        void testDatabaseConnectionReplacedByPlaceholder() throws Exception {
            try (Connection conn = DriverManager.getConnection("jdbc:sqlite::memory:")) {
                Map<String, Object> dataWithDb = new LinkedHashMap<>();
                dataWithDb.put("description", "Database connection");
                dataWithDb.put("connection", conn);

                byte[] dumped = KryoSerializer.serialize(dataWithDb);
                Map<?, ?> reloaded = (Map<?, ?>) KryoSerializer.deserialize(dumped);

                assertInstanceOf(Map.class, reloaded);
                assertEquals("Database connection", reloaded.get("description"));
                assertInstanceOf(KryoPlaceholder.class, reloaded.get("connection"));
            }
        }

        @Test
        @DisplayName("InputStream replaced by KryoPlaceholder")
        void testInputStreamReplacedByPlaceholder() {
            InputStream stream = new ByteArrayInputStream("test".getBytes());
            Map<String, Object> data = new LinkedHashMap<>();
            data.put("description", "Contains stream");
            data.put("stream", stream);

            byte[] dumped = KryoSerializer.serialize(data);
            Map<?, ?> reloaded = (Map<?, ?>) KryoSerializer.deserialize(dumped);

            assertEquals("Contains stream", reloaded.get("description"));
            assertInstanceOf(KryoPlaceholder.class, reloaded.get("stream"));
        }

        @Test
        @DisplayName("OutputStream replaced by KryoPlaceholder")
        void testOutputStreamReplacedByPlaceholder() {
            OutputStream stream = new ByteArrayOutputStream();
            Map<String, Object> data = new LinkedHashMap<>();
            data.put("stream", stream);

            byte[] dumped = KryoSerializer.serialize(data);
            Map<?, ?> reloaded = (Map<?, ?>) KryoSerializer.deserialize(dumped);

            assertInstanceOf(KryoPlaceholder.class, reloaded.get("stream"));
        }

        @Test
        @DisplayName("deeply nested unserializable object")
        void testDeeplyNestedUnserializable() throws Exception {
            try (Socket socket = new Socket()) {
                Map<String, Object> level3 = new LinkedHashMap<>();
                level3.put("normal", "value");
                level3.put("socket", socket);

                Map<String, Object> level2 = new LinkedHashMap<>();
                level2.put("level3", level3);

                Map<String, Object> level1 = new LinkedHashMap<>();
                level1.put("level2", level2);

                Map<String, Object> deepNested = new LinkedHashMap<>();
                deepNested.put("level1", level1);

                byte[] dumped = KryoSerializer.serialize(deepNested);
                Map<?, ?> reloaded = (Map<?, ?>) KryoSerializer.deserialize(dumped);

                Map<?, ?> l1 = (Map<?, ?>) reloaded.get("level1");
                Map<?, ?> l2 = (Map<?, ?>) l1.get("level2");
                Map<?, ?> l3 = (Map<?, ?>) l2.get("level3");

                assertEquals("value", l3.get("normal"));
                assertInstanceOf(KryoPlaceholder.class, l3.get("socket"));
            }
        }

        @Test
        @DisplayName("class with unserializable attribute - field becomes placeholder")
        void testClassWithUnserializableAttribute() throws Exception {
            Socket socket = new Socket();
            try {
                TestClassWithSocket obj = new TestClassWithSocket();
                obj.normal = "normal value";
                obj.unserializable = socket;

                byte[] dumped = KryoSerializer.serialize(obj);
                Object reloaded = KryoSerializer.deserialize(dumped);

                // The object itself is serializable - only the socket field becomes a placeholder
                // This matches Python's pickle_patcher behavior which preserves object structure
                assertInstanceOf(TestClassWithSocket.class, reloaded);
                TestClassWithSocket reloadedObj = (TestClassWithSocket) reloaded;

                assertEquals("normal value", reloadedObj.normal);
                assertInstanceOf(KryoPlaceholder.class, reloadedObj.unserializable);
            } finally {
                socket.close();
            }
        }
    }

    // ============================================================
    // PLACEHOLDER ACCESS TESTS
    // ============================================================

    @Nested
    @DisplayName("Placeholder Access Tests")
    class PlaceholderAccessTests {

        @Test
        @DisplayName("comparing objects with placeholder throws KryoPlaceholderAccessException")
        void testPlaceholderComparisonThrowsException() throws Exception {
            try (Socket socket = new Socket()) {
                Map<String, Object> data = new LinkedHashMap<>();
                data.put("socket", socket);

                byte[] dumped = KryoSerializer.serialize(data);
                Map<?, ?> reloaded = (Map<?, ?>) KryoSerializer.deserialize(dumped);

                KryoPlaceholder placeholder = (KryoPlaceholder) reloaded.get("socket");

                assertThrows(KryoPlaceholderAccessException.class, () -> {
                    ObjectComparator.compare(placeholder, "anything");
                });
            }
        }
    }

    // ============================================================
    // EXCEPTION SERIALIZATION TESTS
    // ============================================================

    @Nested
    @DisplayName("Exception Serialization Tests")
    class ExceptionSerializationTests {

        @Test
        @DisplayName("exception serializes with type and message")
        void testExceptionSerialization() {
            Exception original = new IllegalArgumentException("test error");

            byte[] dumped = KryoSerializer.serializeException(original);
            Map<?, ?> reloaded = (Map<?, ?>) KryoSerializer.deserialize(dumped);

            assertEquals(true, reloaded.get("__exception__"));
            assertEquals("java.lang.IllegalArgumentException", reloaded.get("type"));
            assertEquals("test error", reloaded.get("message"));
            assertNotNull(reloaded.get("stackTrace"));
        }

        @Test
        @DisplayName("exception with cause includes cause info")
        void testExceptionWithCause() {
            Exception cause = new NullPointerException("root cause");
            Exception original = new RuntimeException("wrapper", cause);

            byte[] dumped = KryoSerializer.serializeException(original);
            Map<?, ?> reloaded = (Map<?, ?>) KryoSerializer.deserialize(dumped);

            assertEquals("java.lang.NullPointerException", reloaded.get("causeType"));
            assertEquals("root cause", reloaded.get("causeMessage"));
        }
    }

    // ============================================================
    // CIRCULAR REFERENCE TESTS
    // ============================================================

    @Nested
    @DisplayName("Circular Reference Tests")
    class CircularReferenceTests {

        @Test
        @DisplayName("circular reference handled without stack overflow")
        void testCircularReference() {
            Node a = new Node("A");
            Node b = new Node("B");
            a.next = b;
            b.next = a;

            byte[] dumped = KryoSerializer.serialize(a);
            assertNotNull(dumped);

            Object reloaded = KryoSerializer.deserialize(dumped);
            assertNotNull(reloaded);
        }

        @Test
        @DisplayName("self-referencing object handled gracefully")
        void testSelfReference() {
            SelfReferencing obj = new SelfReferencing();
            obj.self = obj;

            byte[] dumped = KryoSerializer.serialize(obj);
            assertNotNull(dumped);

            Object reloaded = KryoSerializer.deserialize(dumped);
            assertNotNull(reloaded);
        }

        @Test
        @DisplayName("deeply nested structure respects max depth")
        void testDeeplyNested() {
            Map<String, Object> current = new HashMap<>();
            Map<String, Object> root = current;

            for (int i = 0; i < 20; i++) {
                Map<String, Object> next = new HashMap<>();
                current.put("nested", next);
                current = next;
            }
            current.put("value", "deep");

            byte[] dumped = KryoSerializer.serialize(root);
            assertNotNull(dumped);
        }
    }

    // ============================================================
    // FULL FLOW TESTS - SQLite Integration
    // ============================================================

    @Nested
    @DisplayName("Full Flow Tests - SQLite Integration")
    class FullFlowTests {

        @Test
        @DisplayName("serialize -> store in SQLite BLOB -> read -> deserialize -> compare")
        void testFullFlowWithSQLite() throws Exception {
            Path dbPath = Files.createTempFile("kryo_test_", ".db");

            try {
                Map<String, Object> inputArgs = new LinkedHashMap<>();
                inputArgs.put("numbers", Arrays.asList(3, 1, 4, 1, 5));
                inputArgs.put("name", "test");

                List<Integer> result = Arrays.asList(1, 1, 3, 4, 5);

                byte[] argsBlob = KryoSerializer.serialize(inputArgs);
                byte[] resultBlob = KryoSerializer.serialize(result);

                try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + dbPath)) {
                    conn.createStatement().execute(
                        "CREATE TABLE test_results (id INTEGER PRIMARY KEY, args BLOB, result BLOB)"
                    );

                    try (PreparedStatement ps = conn.prepareStatement(
                        "INSERT INTO test_results (id, args, result) VALUES (?, ?, ?)")) {
                        ps.setInt(1, 1);
                        ps.setBytes(2, argsBlob);
                        ps.setBytes(3, resultBlob);
                        ps.executeUpdate();
                    }

                    try (PreparedStatement ps = conn.prepareStatement(
                        "SELECT args, result FROM test_results WHERE id = ?")) {
                        ps.setInt(1, 1);
                        try (ResultSet rs = ps.executeQuery()) {
                            assertTrue(rs.next());

                            byte[] storedArgs = rs.getBytes("args");
                            byte[] storedResult = rs.getBytes("result");

                            Object deserializedArgs = KryoSerializer.deserialize(storedArgs);
                            Object deserializedResult = KryoSerializer.deserialize(storedResult);

                            assertTrue(ObjectComparator.compare(inputArgs, deserializedArgs),
                                "Args should match after full SQLite round-trip");
                            assertTrue(ObjectComparator.compare(result, deserializedResult),
                                "Result should match after full SQLite round-trip");
                        }
                    }
                }
            } finally {
                Files.deleteIfExists(dbPath);
            }
        }

        @Test
        @DisplayName("full flow with custom objects")
        void testFullFlowWithCustomObjects() throws Exception {
            Path dbPath = Files.createTempFile("kryo_custom_", ".db");

            try {
                TestPerson original = new TestPerson("Alice", 25);

                byte[] blob = KryoSerializer.serialize(original);

                try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + dbPath)) {
                    conn.createStatement().execute(
                        "CREATE TABLE objects (id INTEGER PRIMARY KEY, data BLOB)"
                    );

                    try (PreparedStatement ps = conn.prepareStatement(
                        "INSERT INTO objects (id, data) VALUES (?, ?)")) {
                        ps.setInt(1, 1);
                        ps.setBytes(2, blob);
                        ps.executeUpdate();
                    }

                    try (PreparedStatement ps = conn.prepareStatement(
                        "SELECT data FROM objects WHERE id = ?")) {
                        ps.setInt(1, 1);
                        try (ResultSet rs = ps.executeQuery()) {
                            assertTrue(rs.next());
                            byte[] stored = rs.getBytes("data");
                            Object deserialized = KryoSerializer.deserialize(stored);

                            assertTrue(ObjectComparator.compare(original, deserialized));
                        }
                    }
                }
            } finally {
                Files.deleteIfExists(dbPath);
            }
        }
    }

    // ============================================================
    // DATE/TIME AND ENUM TESTS
    // ============================================================

    @Nested
    @DisplayName("Date/Time and Enum Tests")
    class DateTimeEnumTests {

        @Test
        @DisplayName("LocalDate roundtrips correctly")
        void testLocalDate() {
            LocalDate original = LocalDate.of(2024, 1, 15);
            byte[] dumped = KryoSerializer.serialize(original);
            Object reloaded = KryoSerializer.deserialize(dumped);
            assertTrue(ObjectComparator.compare(original, reloaded));
        }

        @Test
        @DisplayName("LocalDateTime roundtrips correctly")
        void testLocalDateTime() {
            LocalDateTime original = LocalDateTime.of(2024, 1, 15, 10, 30, 45);
            byte[] dumped = KryoSerializer.serialize(original);
            Object reloaded = KryoSerializer.deserialize(dumped);
            assertTrue(ObjectComparator.compare(original, reloaded));
        }

        @Test
        @DisplayName("Date roundtrips correctly")
        void testDate() {
            Date original = new Date();
            byte[] dumped = KryoSerializer.serialize(original);
            Object reloaded = KryoSerializer.deserialize(dumped);
            assertTrue(ObjectComparator.compare(original, reloaded));
        }

        @Test
        @DisplayName("enum roundtrips correctly")
        void testEnum() {
            TestEnum original = TestEnum.VALUE_B;
            byte[] dumped = KryoSerializer.serialize(original);
            Object reloaded = KryoSerializer.deserialize(dumped);
            assertTrue(ObjectComparator.compare(original, reloaded));
        }
    }

    // ============================================================
    // TEST HELPER CLASSES
    // ============================================================

    static class TestPerson {
        String name;
        int age;

        TestPerson() {}

        TestPerson(String name, int age) {
            this.name = name;
            this.age = age;
        }
    }

    static class TestClassWithSocket {
        String normal;
        Object unserializable;  // Using Object to allow placeholder substitution

        TestClassWithSocket() {}
    }

    static class Node {
        String value;
        Node next;

        Node() {}

        Node(String value) {
            this.value = value;
        }
    }

    static class SelfReferencing {
        SelfReferencing self;

        SelfReferencing() {}
    }

    enum TestEnum {
        VALUE_A, VALUE_B, VALUE_C
    }
}
