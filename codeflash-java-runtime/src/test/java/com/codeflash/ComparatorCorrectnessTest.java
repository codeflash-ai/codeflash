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

        insertRow(originalDb, "iter_1_0", 1, placeholderBytes);
        insertRow(candidateDb, "iter_1_1", 1, placeholderBytes);

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

        insertRow(originalDb, "iter_1_0", 1, corruptedBytes);
        insertRow(candidateDb, "iter_1_1", 1, corruptedBytes);

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

        insertRow(originalDb, "iter_1_0", 1, realBytes1);
        insertRow(candidateDb, "iter_1_1", 1, realBytes1);
        insertRow(originalDb, "iter_2_0", 1, realBytes2);
        insertRow(candidateDb, "iter_2_1", 1, realBytes2);
        insertRow(originalDb, "iter_3_0", 1, placeholderBytes);
        insertRow(candidateDb, "iter_3_1", 1, placeholderBytes);

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

        insertRow(originalDb, "iter_1_0", 1, bytes1);
        insertRow(candidateDb, "iter_1_1", 1, bytes1);
        insertRow(originalDb, "iter_2_0", 1, bytes2);
        insertRow(candidateDb, "iter_2_1", 1, bytes2);

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

        insertRow(originalDb, "iter_1_0", 1, origBytes);
        insertRow(candidateDb, "iter_1_1", 1, candBytes);

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
        insertRow(originalDb, "iter_1_0", 1, null);
        insertRow(candidateDb, "iter_1_1", 1, null);

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
        insertRow(originalDb, "iter_1_0", 1, bytes);
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

    // --- Helpers ---

    private void createTestDb(Path dbPath) throws Exception {
        String url = "jdbc:sqlite:" + dbPath;
        try (Connection conn = DriverManager.getConnection(url);
             Statement stmt = conn.createStatement()) {
            stmt.execute("CREATE TABLE IF NOT EXISTS test_results ("
                + "iteration_id TEXT NOT NULL, "
                + "loop_index INTEGER NOT NULL, "
                + "return_value BLOB, "
                + "PRIMARY KEY (iteration_id, loop_index))");
        }
    }

    private void insertRow(Path dbPath, String iterationId, int loopIndex, byte[] returnValue) throws Exception {
        String url = "jdbc:sqlite:" + dbPath;
        try (Connection conn = DriverManager.getConnection(url);
             PreparedStatement ps = conn.prepareStatement(
                 "INSERT INTO test_results (iteration_id, loop_index, return_value) VALUES (?, ?, ?)")) {
            ps.setString(1, iterationId);
            ps.setInt(2, loopIndex);
            ps.setBytes(3, returnValue);
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
