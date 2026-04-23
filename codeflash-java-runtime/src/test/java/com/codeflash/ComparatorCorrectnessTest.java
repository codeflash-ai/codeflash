package com.codeflash;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.Statement;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Comparator Correctness Tests")
class ComparatorCorrectnessTest {

    @TempDir
    Path tempDir;

    private Path originalDb;
    private Path candidateDb;

    @BeforeEach
    void setUp() {
        originalDb = tempDir.resolve("original.db");
        candidateDb = tempDir.resolve("candidate.db");
    }

    @Test
    @DisplayName("empty databases → not equivalent (vacuous equivalence guard)")
    void testEmptyDatabases() throws Exception {
        createTestDb(originalDb);
        createTestDb(candidateDb);

        String json = Comparator.compareDatabases(originalDb.toString(), candidateDb.toString());
        Map<String, Object> result = parseJson(json);

        assertFalse((Boolean) result.get("equivalent"));
        assertEquals(0, ((Number) result.get("actualComparisons")).intValue());
        assertEquals(0, ((Number) result.get("totalInvocations")).intValue());
    }

    @Test
    @DisplayName("all placeholder skips → not equivalent")
    void testAllPlaceholderSkips() throws Exception {
        createTestDb(originalDb);
        createTestDb(candidateDb);

        byte[] placeholderBytes = Serializer.serialize(
            KryoPlaceholder.create(new Object(), "unserializable", "root")
        );

        insertRow(originalDb, "1", 1, placeholderBytes);
        insertRow(candidateDb, "1", 1, placeholderBytes);

        String json = Comparator.compareDatabases(originalDb.toString(), candidateDb.toString());
        Map<String, Object> result = parseJson(json);

        assertFalse((Boolean) result.get("equivalent"));
        assertEquals(0, ((Number) result.get("actualComparisons")).intValue());
        assertTrue(((Number) result.get("skippedPlaceholders")).intValue() > 0);
    }

    @Test
    @DisplayName("deserialization errors on both sides → skipped, not equivalent")
    void testDeserializationErrorSkipped() throws Exception {
        createTestDb(originalDb);
        createTestDb(candidateDb);

        // Insert corrupted byte data that will fail Kryo deserialization
        byte[] corruptedBytes = new byte[]{0x01, 0x02, 0x03, (byte) 0xFF, (byte) 0xFE};

        insertRow(originalDb, "1", 1, corruptedBytes);
        insertRow(candidateDb, "1", 1, corruptedBytes);

        String json = Comparator.compareDatabases(originalDb.toString(), candidateDb.toString());
        Map<String, Object> result = parseJson(json);

        assertFalse((Boolean) result.get("equivalent"));
        assertEquals(0, ((Number) result.get("actualComparisons")).intValue());
        assertTrue(((Number) result.get("skippedDeserializationErrors")).intValue() > 0);
    }

    @Test
    @DisplayName("mix of real comparisons and placeholder skips → equivalent if real ones match")
    void testMixedRealAndPlaceholder() throws Exception {
        createTestDb(originalDb);
        createTestDb(candidateDb);

        byte[] realBytes1 = Serializer.serialize(42);
        byte[] realBytes2 = Serializer.serialize("hello");
        byte[] placeholderBytes = Serializer.serialize(
            KryoPlaceholder.create(new Object(), "unserializable", "root")
        );

        insertRow(originalDb, "1", 1, realBytes1);
        insertRow(candidateDb, "1", 1, realBytes1);
        insertRow(originalDb, "2", 1, realBytes2);
        insertRow(candidateDb, "2", 1, realBytes2);
        insertRow(originalDb, "3", 1, placeholderBytes);
        insertRow(candidateDb, "3", 1, placeholderBytes);

        String json = Comparator.compareDatabases(originalDb.toString(), candidateDb.toString());
        Map<String, Object> result = parseJson(json);

        assertTrue((Boolean) result.get("equivalent"));
        assertEquals(2, ((Number) result.get("actualComparisons")).intValue());
        assertEquals(1, ((Number) result.get("skippedPlaceholders")).intValue());
    }

    @Test
    @DisplayName("normal happy path — matching results → equivalent")
    void testNormalHappyPath() throws Exception {
        createTestDb(originalDb);
        createTestDb(candidateDb);

        byte[] bytes1 = Serializer.serialize(100);
        byte[] bytes2 = Serializer.serialize("world");

        insertRow(originalDb, "1", 1, bytes1);
        insertRow(candidateDb, "1", 1, bytes1);
        insertRow(originalDb, "2", 1, bytes2);
        insertRow(candidateDb, "2", 1, bytes2);

        String json = Comparator.compareDatabases(originalDb.toString(), candidateDb.toString());
        Map<String, Object> result = parseJson(json);

        assertTrue((Boolean) result.get("equivalent"));
        assertEquals(2, ((Number) result.get("actualComparisons")).intValue());
        assertEquals(0, ((Number) result.get("skippedPlaceholders")).intValue());
        assertEquals(0, ((Number) result.get("skippedDeserializationErrors")).intValue());
    }

    @Test
    @DisplayName("normal mismatch — different results → not equivalent with diffs")
    void testNormalMismatch() throws Exception {
        createTestDb(originalDb);
        createTestDb(candidateDb);

        byte[] origBytes = Serializer.serialize(42);
        byte[] candBytes = Serializer.serialize(99);

        insertRow(originalDb, "1", 1, origBytes);
        insertRow(candidateDb, "1", 1, candBytes);

        String json = Comparator.compareDatabases(originalDb.toString(), candidateDb.toString());
        Map<String, Object> result = parseJson(json);

        assertFalse((Boolean) result.get("equivalent"));
        assertTrue(((Number) result.get("actualComparisons")).intValue() > 0);
    }

    @Test
    @DisplayName("void methods (both null) → equivalent with actual comparison counted")
    void testVoidMethodsBothNull() throws Exception {
        createTestDb(originalDb);
        createTestDb(candidateDb);

        // Insert rows with NULL return_value (void methods)
        insertRow(originalDb, "1", 1, null);
        insertRow(candidateDb, "1", 1, null);

        String json = Comparator.compareDatabases(originalDb.toString(), candidateDb.toString());
        Map<String, Object> result = parseJson(json);

        assertTrue((Boolean) result.get("equivalent"));
        assertEquals(1, ((Number) result.get("actualComparisons")).intValue());
    }

    @Test
    @DisplayName("one side empty — original has rows, candidate empty → not equivalent")
    void testOneSideEmpty() throws Exception {
        createTestDb(originalDb);
        createTestDb(candidateDb);

        byte[] bytes = Serializer.serialize(42);
        insertRow(originalDb, "1", 1, bytes);
        // candidateDb has no rows

        String json = Comparator.compareDatabases(originalDb.toString(), candidateDb.toString());
        Map<String, Object> result = parseJson(json);

        assertFalse((Boolean) result.get("equivalent"));
        // The missing invocation counts as an actual comparison (it produces a diff)
        assertEquals(1, ((Number) result.get("actualComparisons")).intValue());
    }

    @Test
    @DisplayName("isDeserializationError correctly identifies error maps")
    void testIsDeserializationError() {
        Map<String, String> errorMap = new HashMap<>();
        errorMap.put("__type", "DeserializationError");
        errorMap.put("error", "some error");
        assertTrue(Comparator.isDeserializationError(errorMap));

        Map<String, String> normalMap = new HashMap<>();
        normalMap.put("__type", "SomethingElse");
        assertFalse(Comparator.isDeserializationError(normalMap));

        Map<String, String> emptyMap = new HashMap<>();
        assertFalse(Comparator.isDeserializationError(emptyMap));

        assertFalse(Comparator.isDeserializationError("not a map"));
        assertFalse(Comparator.isDeserializationError(null));
        assertFalse(Comparator.isDeserializationError(42));
    }

    // ============================================================
    // VOID METHOD STATE COMPARISON — proves we actually compare
    // post-call state for void methods, not just skip them
    // ============================================================

    @Test
    @DisplayName("void state: both sides sorted identically → equivalent")
    void testVoidState_identicalMutation_equivalent() throws Exception {
        createTestDb(originalDb);
        createTestDb(candidateDb);

        // Simulate: bubbleSortInPlace(arr) — both original and candidate sort correctly
        // Post-call state: Object[]{sorted_array}
        int[] sortedArr = {1, 2, 3, 4, 5};
        byte[] origState = Serializer.serialize(new Object[]{sortedArr});
        byte[] candState = Serializer.serialize(new Object[]{new int[]{1, 2, 3, 4, 5}});

        insertRow(originalDb, "L1_1", 1, origState);
        insertRow(candidateDb, "L1_1", 1, candState);

        String json = Comparator.compareDatabases(originalDb.toString(), candidateDb.toString());
        Map<String, Object> result = parseJson(json);

        assertTrue((Boolean) result.get("equivalent"),
            "Both sides produce same sorted array — should be equivalent");
        assertEquals(1, ((Number) result.get("actualComparisons")).intValue());
    }

    @Test
    @DisplayName("void state: candidate mutates array differently → NOT equivalent")
    void testVoidState_differentMutation_rejected() throws Exception {
        createTestDb(originalDb);
        createTestDb(candidateDb);

        // Simulate: original sorts [3,1,2] → [1,2,3]
        // Bad optimization doesn't sort correctly → [3,1,2] unchanged
        byte[] origState = Serializer.serialize(new Object[]{new int[]{1, 2, 3}});
        byte[] candState = Serializer.serialize(new Object[]{new int[]{3, 1, 2}});

        insertRow(originalDb, "L1_1", 1, origState);
        insertRow(candidateDb, "L1_1", 1, candState);

        String json = Comparator.compareDatabases(originalDb.toString(), candidateDb.toString());
        Map<String, Object> result = parseJson(json);

        assertFalse((Boolean) result.get("equivalent"),
            "Candidate produced wrong array — must be rejected");
        assertEquals(1, ((Number) result.get("actualComparisons")).intValue());
    }

    @Test
    @DisplayName("void state: receiver + args both compared — wrong receiver state rejected")
    void testVoidState_receiverAndArgs_wrongReceiverRejected() throws Exception {
        createTestDb(originalDb);
        createTestDb(candidateDb);

        // Simulate: instance method sorter.sort(data)
        // Post-call state is Object[]{receiver_fields_map, mutated_data}
        // Original: receiver has size=3, data is [1,2,3]
        // Candidate: receiver has size=0 (wrong), data is [1,2,3]
        Map<String, Object> origReceiver = new HashMap<>();
        origReceiver.put("size", 3);
        origReceiver.put("sorted", true);
        Map<String, Object> candReceiver = new HashMap<>();
        candReceiver.put("size", 0);
        candReceiver.put("sorted", true);

        byte[] origState = Serializer.serialize(new Object[]{origReceiver, new int[]{1, 2, 3}});
        byte[] candState = Serializer.serialize(new Object[]{candReceiver, new int[]{1, 2, 3}});

        insertRow(originalDb, "L1_1", 1, origState);
        insertRow(candidateDb, "L1_1", 1, candState);

        String json = Comparator.compareDatabases(originalDb.toString(), candidateDb.toString());
        Map<String, Object> result = parseJson(json);

        assertFalse((Boolean) result.get("equivalent"),
            "Receiver state differs (size 3 vs 0) — must be rejected even though args match");
        assertEquals(1, ((Number) result.get("actualComparisons")).intValue());
    }

    // --- Helpers ---

    private void createTestDb(Path dbPath) throws Exception {
        String url = "jdbc:sqlite:" + dbPath;
        try (Connection conn = DriverManager.getConnection(url);
             Statement stmt = conn.createStatement()) {
            stmt.execute("CREATE TABLE IF NOT EXISTS test_results ("
                + "test_module_path TEXT NOT NULL, "
                + "test_class_name TEXT NOT NULL, "
                + "test_function_name TEXT NOT NULL, "
                + "iteration_id TEXT NOT NULL, "
                + "loop_index INTEGER NOT NULL, "
                + "return_value BLOB, "
                + "verification_type TEXT)");
        }
    }

    private void insertRow(Path dbPath, String iterationId, int loopIndex, byte[] returnValue) throws Exception {
        String url = "jdbc:sqlite:" + dbPath;
        try (Connection conn = DriverManager.getConnection(url);
             PreparedStatement ps = conn.prepareStatement(
                 "INSERT INTO test_results (test_module_path, test_class_name, test_function_name, iteration_id, loop_index, return_value) VALUES (?, ?, ?, ?, ?, ?)")) {
            ps.setString(1, "src/test/java/com/example/TestClass.java");
            ps.setString(2, "TestClass");
            ps.setString(3, "testMethod");
            ps.setString(4, iterationId);
            ps.setInt(5, loopIndex);
            ps.setBytes(6, returnValue);
            ps.executeUpdate();
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> parseJson(String json) {
        // Minimal JSON parsing for test assertions — handles the flat structure from compareDatabases
        Map<String, Object> result = new HashMap<>();

        // Remove outer braces
        json = json.trim();
        if (json.startsWith("{")) json = json.substring(1);
        if (json.endsWith("}")) json = json.substring(0, json.length() - 1);

        // Extract known fields
        result.put("equivalent", extractBoolean(json, "equivalent"));
        result.put("totalInvocations", extractInt(json, "totalInvocations"));
        result.put("actualComparisons", extractInt(json, "actualComparisons"));
        result.put("skippedPlaceholders", extractInt(json, "skippedPlaceholders"));
        result.put("skippedDeserializationErrors", extractInt(json, "skippedDeserializationErrors"));

        return result;
    }

    private Boolean extractBoolean(String json, String key) {
        String pattern = "\"" + key + "\":";
        int idx = json.indexOf(pattern);
        if (idx < 0) return null;
        String after = json.substring(idx + pattern.length()).trim();
        return after.startsWith("true");
    }

    private Integer extractInt(String json, String key) {
        String pattern = "\"" + key + "\":";
        int idx = json.indexOf(pattern);
        if (idx < 0) return null;
        String after = json.substring(idx + pattern.length()).trim();
        StringBuilder sb = new StringBuilder();
        for (char c : after.toCharArray()) {
            if (Character.isDigit(c) || c == '-') sb.append(c);
            else break;
        }
        return sb.length() > 0 ? Integer.parseInt(sb.toString()) : null;
    }
}
