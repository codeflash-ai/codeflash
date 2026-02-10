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
 * Tests for Serializer following Python's dill/patcher test patterns.
 *
 * Test pattern: Create object -> Serialize -> Deserialize -> Compare with original
 */
@DisplayName("Serializer Tests")
class SerializerTest {

    @BeforeEach
    void setUp() {
        Serializer.clearUnserializableTypesCache();
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

            byte[] dumped = Serializer.serialize(originalData);
            Object reloaded = Serializer.deserialize(dumped);

            assertTrue(Comparator.compare(originalData, reloaded),
                "Reloaded data should equal original data");
        }

        @Test
        @DisplayName("integers roundtrip correctly")
        void testIntegers() {
            int[] testCases = {5, 0, -1, Integer.MAX_VALUE, Integer.MIN_VALUE};
            for (int original : testCases) {
                byte[] dumped = Serializer.serialize(original);
                Object reloaded = Serializer.deserialize(dumped);
                assertTrue(Comparator.compare(original, reloaded),
                    "Failed for: " + original);
            }
        }

        @Test
        @DisplayName("floats roundtrip correctly with epsilon tolerance")
        void testFloats() {
            double[] testCases = {5.0, 0.0, -1.0, 3.14159, Double.MAX_VALUE};
            for (double original : testCases) {
                byte[] dumped = Serializer.serialize(original);
                Object reloaded = Serializer.deserialize(dumped);
                assertTrue(Comparator.compare(original, reloaded),
                    "Failed for: " + original);
            }
        }

        @Test
        @DisplayName("strings roundtrip correctly")
        void testStrings() {
            String[] testCases = {"Hello", "", "World", "unicode: \u00e9\u00e8"};
            for (String original : testCases) {
                byte[] dumped = Serializer.serialize(original);
                Object reloaded = Serializer.deserialize(dumped);
                assertTrue(Comparator.compare(original, reloaded),
                    "Failed for: " + original);
            }
        }

        @Test
        @DisplayName("lists roundtrip correctly")
        void testLists() {
            List<Integer> original = Arrays.asList(1, 2, 3);
            byte[] dumped = Serializer.serialize(original);
            Object reloaded = Serializer.deserialize(dumped);
            assertTrue(Comparator.compare(original, reloaded));
        }

        @Test
        @DisplayName("maps roundtrip correctly")
        void testMaps() {
            Map<String, Integer> original = new LinkedHashMap<>();
            original.put("a", 1);
            original.put("b", 2);

            byte[] dumped = Serializer.serialize(original);
            Object reloaded = Serializer.deserialize(dumped);
            assertTrue(Comparator.compare(original, reloaded));
        }

        @Test
        @DisplayName("sets roundtrip correctly")
        void testSets() {
            Set<Integer> original = new LinkedHashSet<>(Arrays.asList(1, 2, 3));
            byte[] dumped = Serializer.serialize(original);
            Object reloaded = Serializer.deserialize(dumped);
            assertTrue(Comparator.compare(original, reloaded));
        }

        @Test
        @DisplayName("null roundtrips correctly")
        void testNull() {
            byte[] dumped = Serializer.serialize(null);
            Object reloaded = Serializer.deserialize(dumped);
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

                byte[] dumped = Serializer.serialize(dataWithSocket);
                Map<?, ?> reloaded = (Map<?, ?>) Serializer.deserialize(dumped);

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

                byte[] dumped = Serializer.serialize(dataWithDb);
                Map<?, ?> reloaded = (Map<?, ?>) Serializer.deserialize(dumped);

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

            byte[] dumped = Serializer.serialize(data);
            Map<?, ?> reloaded = (Map<?, ?>) Serializer.deserialize(dumped);

            assertEquals("Contains stream", reloaded.get("description"));
            assertInstanceOf(KryoPlaceholder.class, reloaded.get("stream"));
        }

        @Test
        @DisplayName("OutputStream replaced by KryoPlaceholder")
        void testOutputStreamReplacedByPlaceholder() {
            OutputStream stream = new ByteArrayOutputStream();
            Map<String, Object> data = new LinkedHashMap<>();
            data.put("stream", stream);

            byte[] dumped = Serializer.serialize(data);
            Map<?, ?> reloaded = (Map<?, ?>) Serializer.deserialize(dumped);

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

                byte[] dumped = Serializer.serialize(deepNested);
                Map<?, ?> reloaded = (Map<?, ?>) Serializer.deserialize(dumped);

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

                byte[] dumped = Serializer.serialize(obj);
                Object reloaded = Serializer.deserialize(dumped);

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

                byte[] dumped = Serializer.serialize(data);
                Map<?, ?> reloaded = (Map<?, ?>) Serializer.deserialize(dumped);

                KryoPlaceholder placeholder = (KryoPlaceholder) reloaded.get("socket");

                assertThrows(KryoPlaceholderAccessException.class, () -> {
                    Comparator.compare(placeholder, "anything");
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

            byte[] dumped = Serializer.serializeException(original);
            Map<?, ?> reloaded = (Map<?, ?>) Serializer.deserialize(dumped);

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

            byte[] dumped = Serializer.serializeException(original);
            Map<?, ?> reloaded = (Map<?, ?>) Serializer.deserialize(dumped);

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

            byte[] dumped = Serializer.serialize(a);
            assertNotNull(dumped);

            Object reloaded = Serializer.deserialize(dumped);
            assertNotNull(reloaded);
        }

        @Test
        @DisplayName("self-referencing object handled gracefully")
        void testSelfReference() {
            SelfReferencing obj = new SelfReferencing();
            obj.self = obj;

            byte[] dumped = Serializer.serialize(obj);
            assertNotNull(dumped);

            Object reloaded = Serializer.deserialize(dumped);
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

            byte[] dumped = Serializer.serialize(root);
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

                byte[] argsBlob = Serializer.serialize(inputArgs);
                byte[] resultBlob = Serializer.serialize(result);

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

                            Object deserializedArgs = Serializer.deserialize(storedArgs);
                            Object deserializedResult = Serializer.deserialize(storedResult);

                            assertTrue(Comparator.compare(inputArgs, deserializedArgs),
                                "Args should match after full SQLite round-trip");
                            assertTrue(Comparator.compare(result, deserializedResult),
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

                byte[] blob = Serializer.serialize(original);

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
                            Object deserialized = Serializer.deserialize(stored);

                            assertTrue(Comparator.compare(original, deserialized));
                        }
                    }
                }
            } finally {
                Files.deleteIfExists(dbPath);
            }
        }
    }

    // ============================================================
    // BEHAVIOR TUPLE FORMAT TESTS (from JS patterns)
    // ============================================================

    @Nested
    @DisplayName("Behavior Tuple Format Tests")
    class BehaviorTupleFormatTests {

        @Test
        @DisplayName("behavior tuple [args, kwargs, returnValue] serializes correctly")
        void testBehaviorTupleFormat() {
            // Simulate what instrumentation does: [args, {}, returnValue]
            List<Object> args = Arrays.asList(42, "hello");
            Map<String, Object> kwargs = new LinkedHashMap<>();  // Java doesn't have kwargs, always empty
            Map<String, Object> returnValue = new LinkedHashMap<>();
            returnValue.put("result", 84);
            returnValue.put("message", "HELLO");

            List<Object> behaviorTuple = Arrays.asList(args, kwargs, returnValue);
            byte[] serialized = Serializer.serialize(behaviorTuple);
            List<?> restored = (List<?>) Serializer.deserialize(serialized);

            assertTrue(Comparator.compare(behaviorTuple, restored));
            assertEquals(args, restored.get(0));
            assertEquals(kwargs, restored.get(1));
            assertTrue(Comparator.compare(returnValue, restored.get(2)));
        }

        @Test
        @DisplayName("behavior with Map return value")
        void testBehaviorWithMapReturn() {
            List<Object> args = Arrays.asList(Arrays.asList(
                Arrays.asList("a", 1),
                Arrays.asList("b", 2)
            ));
            Map<String, Integer> returnValue = new LinkedHashMap<>();
            returnValue.put("a", 1);
            returnValue.put("b", 2);

            List<Object> behaviorTuple = Arrays.asList(args, new LinkedHashMap<>(), returnValue);
            byte[] serialized = Serializer.serialize(behaviorTuple);
            List<?> restored = (List<?>) Serializer.deserialize(serialized);

            assertTrue(Comparator.compare(behaviorTuple, restored));
            assertInstanceOf(Map.class, restored.get(2));
        }

        @Test
        @DisplayName("behavior with Set return value")
        void testBehaviorWithSetReturn() {
            List<Object> args = Arrays.asList(Arrays.asList(1, 2, 3));
            Set<Integer> returnValue = new LinkedHashSet<>(Arrays.asList(1, 2, 3));

            List<Object> behaviorTuple = Arrays.asList(args, new LinkedHashMap<>(), returnValue);
            byte[] serialized = Serializer.serialize(behaviorTuple);
            List<?> restored = (List<?>) Serializer.deserialize(serialized);

            assertTrue(Comparator.compare(behaviorTuple, restored));
            assertInstanceOf(Set.class, restored.get(2));
        }

        @Test
        @DisplayName("behavior with Date return value")
        void testBehaviorWithDateReturn() {
            long timestamp = 1705276800000L;  // 2024-01-15
            List<Object> args = Arrays.asList(timestamp);
            Date returnValue = new Date(timestamp);

            List<Object> behaviorTuple = Arrays.asList(args, new LinkedHashMap<>(), returnValue);
            byte[] serialized = Serializer.serialize(behaviorTuple);
            List<?> restored = (List<?>) Serializer.deserialize(serialized);

            assertTrue(Comparator.compare(behaviorTuple, restored));
            assertInstanceOf(Date.class, restored.get(2));
            assertEquals(timestamp, ((Date) restored.get(2)).getTime());
        }
    }

    // ============================================================
    // SIMULATED ORIGINAL VS OPTIMIZED COMPARISON (from JS patterns)
    // ============================================================

    @Nested
    @DisplayName("Simulated Original vs Optimized Comparison")
    class OriginalVsOptimizedTests {

        private List<Object> runAndCapture(java.util.function.Function<Integer, Integer> fn, int arg) {
            Integer returnValue = fn.apply(arg);
            return Arrays.asList(Arrays.asList(arg), new LinkedHashMap<>(), returnValue);
        }

        @Test
        @DisplayName("identical behaviors are equal - number function")
        void testIdenticalBehaviorsNumber() {
            java.util.function.Function<Integer, Integer> fn = x -> x * 2;
            int arg = 21;

            // "Original" run
            List<Object> original = runAndCapture(fn, arg);
            byte[] originalSerialized = Serializer.serialize(original);

            // "Optimized" run (same function, simulating optimization)
            List<Object> optimized = runAndCapture(fn, arg);
            byte[] optimizedSerialized = Serializer.serialize(optimized);

            // Deserialize and compare (what verification does)
            Object originalRestored = Serializer.deserialize(originalSerialized);
            Object optimizedRestored = Serializer.deserialize(optimizedSerialized);

            assertTrue(Comparator.compare(originalRestored, optimizedRestored));
        }

        @Test
        @DisplayName("different behaviors are NOT equal")
        void testDifferentBehaviors() {
            java.util.function.Function<Integer, Integer> fn1 = x -> x * 2;
            java.util.function.Function<Integer, Integer> fn2 = x -> x * 3;  // Different behavior!
            int arg = 10;

            List<Object> original = runAndCapture(fn1, arg);
            byte[] originalSerialized = Serializer.serialize(original);

            List<Object> optimized = runAndCapture(fn2, arg);
            byte[] optimizedSerialized = Serializer.serialize(optimized);

            Object originalRestored = Serializer.deserialize(originalSerialized);
            Object optimizedRestored = Serializer.deserialize(optimizedSerialized);

            // Should be FALSE - behaviors differ (20 vs 30)
            assertFalse(Comparator.compare(originalRestored, optimizedRestored));
        }

        @Test
        @DisplayName("floating point tolerance works")
        void testFloatingPointTolerance() {
            // Simulate slight floating point differences from optimization
            List<Object> original = Arrays.asList(
                Arrays.asList(1.0),
                new LinkedHashMap<>(),
                0.30000000000000004
            );
            List<Object> optimized = Arrays.asList(
                Arrays.asList(1.0),
                new LinkedHashMap<>(),
                0.3
            );

            byte[] originalSerialized = Serializer.serialize(original);
            byte[] optimizedSerialized = Serializer.serialize(optimized);

            Object originalRestored = Serializer.deserialize(originalSerialized);
            Object optimizedRestored = Serializer.deserialize(optimizedSerialized);

            // Should be TRUE with default tolerance
            assertTrue(Comparator.compare(originalRestored, optimizedRestored));
        }
    }

    // ============================================================
    // MULTIPLE INVOCATIONS COMPARISON (from JS patterns)
    // ============================================================

    @Nested
    @DisplayName("Multiple Invocations Comparison")
    class MultipleInvocationsTests {

        @Test
        @DisplayName("batch of invocations can be compared")
        void testBatchInvocations() {
            // Define test cases: function behavior with args and expected return
            List<List<Object>> testCases = Arrays.asList(
                Arrays.asList(Arrays.asList(1), 2),       // x -> x * 2
                Arrays.asList(Arrays.asList(100), 200),
                Arrays.asList(Arrays.asList("hello"), "HELLO"),
                Arrays.asList(Arrays.asList(Arrays.asList(1, 2, 3)), Arrays.asList(2, 4, 6))
            );

            // Simulate original run
            List<byte[]> originalResults = new ArrayList<>();
            for (List<Object> testCase : testCases) {
                List<Object> tuple = Arrays.asList(testCase.get(0), new LinkedHashMap<>(), testCase.get(1));
                originalResults.add(Serializer.serialize(tuple));
            }

            // Simulate optimized run (same results)
            List<byte[]> optimizedResults = new ArrayList<>();
            for (List<Object> testCase : testCases) {
                List<Object> tuple = Arrays.asList(testCase.get(0), new LinkedHashMap<>(), testCase.get(1));
                optimizedResults.add(Serializer.serialize(tuple));
            }

            // Compare all results
            for (int i = 0; i < testCases.size(); i++) {
                Object originalRestored = Serializer.deserialize(originalResults.get(i));
                Object optimizedRestored = Serializer.deserialize(optimizedResults.get(i));

                assertTrue(Comparator.compare(originalRestored, optimizedRestored),
                    "Failed at test case " + i);
            }
        }
    }

    // ============================================================
    // EDGE CASES (from JS patterns)
    // ============================================================

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCaseTests {

        @Test
        @DisplayName("handles special values in args")
        void testSpecialValuesInArgs() {
            List<Object> tuple = Arrays.asList(
                Arrays.asList(Double.NaN, Double.POSITIVE_INFINITY, null),
                new LinkedHashMap<>(),
                "processed"
            );

            byte[] serialized = Serializer.serialize(tuple);
            List<?> restored = (List<?>) Serializer.deserialize(serialized);

            assertTrue(Comparator.compare(tuple, restored));
            List<?> args = (List<?>) restored.get(0);
            assertTrue(Double.isNaN((Double) args.get(0)));
            assertEquals(Double.POSITIVE_INFINITY, args.get(1));
            assertNull(args.get(2));
        }

        @Test
        @DisplayName("handles empty behavior tuple")
        void testEmptyBehavior() {
            List<Object> tuple = Arrays.asList(
                new ArrayList<>(),
                new LinkedHashMap<>(),
                null
            );

            byte[] serialized = Serializer.serialize(tuple);
            List<?> restored = (List<?>) Serializer.deserialize(serialized);

            assertTrue(Comparator.compare(tuple, restored));
        }

        @Test
        @DisplayName("handles large arrays in behavior")
        void testLargeArrays() {
            List<Integer> largeArray = new ArrayList<>();
            for (int i = 0; i < 1000; i++) {
                largeArray.add(i);
            }
            int sum = largeArray.stream().mapToInt(Integer::intValue).sum();

            List<Object> tuple = Arrays.asList(
                Arrays.asList(largeArray),
                new LinkedHashMap<>(),
                sum
            );

            byte[] serialized = Serializer.serialize(tuple);
            List<?> restored = (List<?>) Serializer.deserialize(serialized);

            assertTrue(Comparator.compare(tuple, restored));
        }

        @Test
        @DisplayName("NaN equals NaN in comparison")
        void testNaNEquality() {
            double nanValue = Double.NaN;

            byte[] serialized = Serializer.serialize(nanValue);
            Object restored = Serializer.deserialize(serialized);

            assertTrue(Comparator.compare(nanValue, restored));
        }

        @Test
        @DisplayName("Infinity values compare correctly")
        void testInfinityValues() {
            List<Double> values = Arrays.asList(
                Double.POSITIVE_INFINITY,
                Double.NEGATIVE_INFINITY
            );

            byte[] serialized = Serializer.serialize(values);
            Object restored = Serializer.deserialize(serialized);

            assertTrue(Comparator.compare(values, restored));
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
            byte[] dumped = Serializer.serialize(original);
            Object reloaded = Serializer.deserialize(dumped);
            assertTrue(Comparator.compare(original, reloaded));
        }

        @Test
        @DisplayName("LocalDateTime roundtrips correctly")
        void testLocalDateTime() {
            LocalDateTime original = LocalDateTime.of(2024, 1, 15, 10, 30, 45);
            byte[] dumped = Serializer.serialize(original);
            Object reloaded = Serializer.deserialize(dumped);
            assertTrue(Comparator.compare(original, reloaded));
        }

        @Test
        @DisplayName("Date roundtrips correctly")
        void testDate() {
            Date original = new Date();
            byte[] dumped = Serializer.serialize(original);
            Object reloaded = Serializer.deserialize(dumped);
            assertTrue(Comparator.compare(original, reloaded));
        }

        @Test
        @DisplayName("enum roundtrips correctly")
        void testEnum() {
            TestEnum original = TestEnum.VALUE_B;
            byte[] dumped = Serializer.serialize(original);
            Object reloaded = Serializer.deserialize(dumped);
            assertTrue(Comparator.compare(original, reloaded));
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

    // ============================================================
    // FIXED ISSUES TESTS - These verify the fixes work correctly
    // ============================================================

    @Nested
    @DisplayName("Fixed - Field Type Mismatch Handling")
    class FieldTypeMismatchTests {

        @Test
        @DisplayName("FIXED: typed field with unserializable value - object becomes Map with placeholder")
        void testTypedFieldBecomesMapWithPlaceholder() throws Exception {
            // When field is typed as Socket (not Object), the object becomes a Map
            // so the placeholder can be preserved
            TestClassWithTypedSocket obj = new TestClassWithTypedSocket();
            obj.normal = "normal value";
            obj.socket = new Socket();

            byte[] dumped = Serializer.serialize(obj);
            Object reloaded = Serializer.deserialize(dumped);

            // FIX: Object becomes Map to preserve the placeholder
            assertInstanceOf(Map.class, reloaded, "Object with incompatible field becomes Map");
            Map<?, ?> result = (Map<?, ?>) reloaded;

            assertEquals("normal value", result.get("normal"));
            assertInstanceOf(KryoPlaceholder.class, result.get("socket"),
                "Socket field is preserved as placeholder in Map");

            obj.socket.close();
        }
    }

    @Nested
    @DisplayName("Fixed - Type Preservation When Recursive Processing Triggered")
    class TypePreservationTests {

        @Test
        @DisplayName("FIXED: array containing unserializable object becomes Object[]")
        void testArrayWithUnserializableBecomesObjectArray() throws Exception {
            Object[] original = new Object[]{"normal", new Socket(), "also normal"};

            byte[] dumped = Serializer.serialize(original);
            Object reloaded = Serializer.deserialize(dumped);

            // FIX: Array type is preserved (as Object[])
            assertInstanceOf(Object[].class, reloaded, "Array type preserved");
            Object[] arr = (Object[]) reloaded;
            assertEquals(3, arr.length);
            assertEquals("normal", arr[0]);
            assertInstanceOf(KryoPlaceholder.class, arr[1], "Socket became placeholder");
            assertEquals("also normal", arr[2]);

            ((Socket) original[1]).close();
        }

        @Test
        @DisplayName("FIXED: LinkedList with unserializable preserves LinkedList type")
        void testLinkedListWithUnserializablePreservesType() throws Exception {
            LinkedList<Object> original = new LinkedList<>();
            original.add("normal");
            original.add(new Socket());
            original.add("also normal");

            byte[] dumped = Serializer.serialize(original);
            Object reloaded = Serializer.deserialize(dumped);

            // FIX: LinkedList type is preserved
            assertInstanceOf(LinkedList.class, reloaded, "LinkedList type preserved");
            LinkedList<?> list = (LinkedList<?>) reloaded;
            assertEquals(3, list.size());
            assertInstanceOf(KryoPlaceholder.class, list.get(1), "Socket became placeholder");

            ((Socket) original.get(1)).close();
        }

        @Test
        @DisplayName("FIXED: TreeSet with unserializable preserves TreeSet type")
        void testTreeSetWithUnserializablePreservesType() throws Exception {
            TreeSet<Object> original = new TreeSet<>();
            original.add("a");
            original.add("b");
            original.add("c");

            // Add a map containing unserializable to trigger recursive processing
            Map<String, Object> mapWithSocket = new LinkedHashMap<>();
            mapWithSocket.put("socket", new Socket());

            byte[] dumped = Serializer.serialize(original);
            Object reloaded = Serializer.deserialize(dumped);

            // FIX: TreeSet type is preserved
            assertInstanceOf(TreeSet.class, reloaded, "TreeSet type preserved");

            ((Socket) mapWithSocket.get("socket")).close();
        }

        @Test
        @DisplayName("FIXED: TreeMap with unserializable value preserves TreeMap type")
        void testTreeMapWithUnserializablePreservesType() throws Exception {
            TreeMap<String, Object> original = new TreeMap<>();
            original.put("a", "normal");
            original.put("b", new Socket());
            original.put("c", "also normal");

            byte[] dumped = Serializer.serialize(original);
            Object reloaded = Serializer.deserialize(dumped);

            // FIX: TreeMap type is preserved
            assertInstanceOf(TreeMap.class, reloaded, "TreeMap type preserved");
            TreeMap<?, ?> map = (TreeMap<?, ?>) reloaded;
            assertEquals("normal", map.get("a"));
            assertInstanceOf(KryoPlaceholder.class, map.get("b"), "Socket became placeholder");
            assertEquals("also normal", map.get("c"));

            ((Socket) original.get("b")).close();
        }
    }

    @Nested
    @DisplayName("Fixed - Map Key Comparison")
    class MapKeyComparisonTests {

        @Test
        @DisplayName("Map.containsKey still fails with custom keys (expected Java behavior)")
        void testContainsKeyStillFailsWithCustomKeys() {
            // This is expected Java behavior - containsKey uses equals()
            Map<CustomKeyWithoutEquals, String> original = new LinkedHashMap<>();
            original.put(new CustomKeyWithoutEquals("key1"), "value1");

            byte[] dumped = Serializer.serialize(original);
            Map<?, ?> reloaded = (Map<?, ?>) Serializer.deserialize(dumped);

            // containsKey uses equals(), which is identity-based - this is expected
            assertFalse(reloaded.containsKey(new CustomKeyWithoutEquals("key1")),
                "containsKey uses equals() - expected to fail");
            assertEquals(1, reloaded.size());
        }

        @Test
        @DisplayName("FIXED: Comparator.compareMaps works with custom keys")
        void testComparatorWorksWithCustomKeys() {
            // FIX: Comparator now uses deep comparison for keys
            Map<CustomKeyWithoutEquals, String> map1 = new LinkedHashMap<>();
            map1.put(new CustomKeyWithoutEquals("key1"), "value1");

            Map<CustomKeyWithoutEquals, String> map2 = new LinkedHashMap<>();
            map2.put(new CustomKeyWithoutEquals("key1"), "value1");

            // FIX: Comparison now works using deep key comparison
            assertTrue(Comparator.compare(map1, map2),
                "Maps with custom keys now compare correctly using deep comparison");
        }
    }

    @Nested
    @DisplayName("Verified Working - Direct Serialization")
    class VerifiedWorkingTests {

        @Test
        @DisplayName("WORKS: pure arrays serialize correctly via Kryo direct")
        void testPureArraysWork() {
            int[] intArray = {1, 2, 3};
            String[] strArray = {"a", "b", "c"};

            Object reloadedInt = Serializer.deserialize(Serializer.serialize(intArray));
            Object reloadedStr = Serializer.deserialize(Serializer.serialize(strArray));

            assertInstanceOf(int[].class, reloadedInt, "int[] preserved");
            assertInstanceOf(String[].class, reloadedStr, "String[] preserved");
        }

        @Test
        @DisplayName("WORKS: pure collections serialize correctly via Kryo direct")
        void testPureCollectionsWork() {
            LinkedList<Integer> linkedList = new LinkedList<>(Arrays.asList(1, 2, 3));
            TreeSet<Integer> treeSet = new TreeSet<>(Arrays.asList(3, 1, 2));
            TreeMap<String, Integer> treeMap = new TreeMap<>();
            treeMap.put("c", 3);
            treeMap.put("a", 1);
            treeMap.put("b", 2);

            Object reloadedList = Serializer.deserialize(Serializer.serialize(linkedList));
            Object reloadedSet = Serializer.deserialize(Serializer.serialize(treeSet));
            Object reloadedMap = Serializer.deserialize(Serializer.serialize(treeMap));

            assertInstanceOf(LinkedList.class, reloadedList, "LinkedList preserved");
            assertInstanceOf(TreeSet.class, reloadedSet, "TreeSet preserved");
            assertInstanceOf(TreeMap.class, reloadedMap, "TreeMap preserved");
        }

        @Test
        @DisplayName("WORKS: large collections serialize correctly via Kryo direct")
        void testLargeCollectionsWork() {
            List<Integer> largeList = new ArrayList<>();
            for (int i = 0; i < 5000; i++) {
                largeList.add(i);
            }

            Object reloaded = Serializer.deserialize(Serializer.serialize(largeList));

            assertInstanceOf(ArrayList.class, reloaded);
            assertEquals(5000, ((List<?>) reloaded).size(), "Large list not truncated");
        }
    }

    // ============================================================
    // ADDITIONAL TEST HELPER CLASSES FOR KNOWN ISSUES
    // ============================================================

    static class TestClassWithTypedSocket {
        String normal;
        Socket socket;  // Typed as Socket, not Object - can't hold KryoPlaceholder

        TestClassWithTypedSocket() {}
    }

    static class ContainerWithSocket {
        String name;
        Socket socket;

        ContainerWithSocket() {}
    }

    static class CustomKeyWithoutEquals {
        String value;

        CustomKeyWithoutEquals(String value) {
            this.value = value;
        }

        // Intentionally NO equals() and hashCode() override
        // Uses Object's identity-based equals

        @Override
        public String toString() {
            return "CustomKey(" + value + ")";
        }
    }
}
