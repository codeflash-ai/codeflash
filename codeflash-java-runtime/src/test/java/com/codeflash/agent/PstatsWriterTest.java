package com.codeflash.agent;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class PstatsWriterTest {

    @TempDir
    Path tempDir;

    private CallTracker tracker;

    @BeforeEach
    void setUp() {
        tracker = CallTracker.getInstance();
        tracker.reset();
    }

    @Nested
    class SchemaTests {

        @Test
        void createsAllRequiredTables() throws Exception {
            String dbPath = tempDir.resolve("test_trace.sqlite").toString();
            PstatsWriter.flush(dbPath);

            try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + dbPath);
                 Statement stmt = conn.createStatement()) {

                // Check pstats table exists with correct columns
                ResultSet rs = stmt.executeQuery(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='pstats'");
                assertTrue(rs.next(), "pstats table should exist");
                String sql = rs.getString("sql");
                assertTrue(sql.contains("filename TEXT"), "pstats should have filename column");
                assertTrue(sql.contains("line_number INTEGER"), "pstats should have line_number column");
                assertTrue(sql.contains("function TEXT"), "pstats should have function column");
                assertTrue(sql.contains("class_name TEXT"), "pstats should have class_name column");
                assertTrue(sql.contains("call_count_nonrecursive INTEGER"));
                assertTrue(sql.contains("num_callers INTEGER"));
                assertTrue(sql.contains("total_time_ns INTEGER"));
                assertTrue(sql.contains("cumulative_time_ns INTEGER"));
                assertTrue(sql.contains("callers BLOB"));

                // Check function_calls table
                rs = stmt.executeQuery(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='function_calls'");
                assertTrue(rs.next(), "function_calls table should exist");
                sql = rs.getString("sql");
                assertTrue(sql.contains("type TEXT"));
                assertTrue(sql.contains("function TEXT"));
                assertTrue(sql.contains("classname TEXT"));
                assertTrue(sql.contains("filename TEXT"));
                assertTrue(sql.contains("args BLOB"));

                // Check metadata table
                rs = stmt.executeQuery(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='metadata'");
                assertTrue(rs.next(), "metadata table should exist");

                // Check total_time table
                rs = stmt.executeQuery(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='total_time'");
                assertTrue(rs.next(), "total_time table should exist");
            }
        }
    }

    @Nested
    class PstatsDataTests {

        @Test
        void writesTimingDataCorrectly() throws Exception {
            tracker.markStart();
            tracker.enter("com.example.Foo", "compute", "/src/main/java/com/example/Foo.java", 42, null);
            tracker.exit();
            tracker.markEnd();

            String dbPath = tempDir.resolve("timing_trace.sqlite").toString();
            PstatsWriter.flush(dbPath);

            try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + dbPath);
                 Statement stmt = conn.createStatement()) {

                ResultSet rs = stmt.executeQuery("SELECT * FROM pstats");
                assertTrue(rs.next(), "Should have at least one pstats row");

                assertEquals("/src/main/java/com/example/Foo.java", rs.getString("filename"));
                assertEquals(42, rs.getInt("line_number"));
                assertEquals("compute", rs.getString("function"));
                assertEquals("com.example.Foo", rs.getString("class_name"));
                assertEquals(1, rs.getLong("call_count_nonrecursive"));
                assertTrue(rs.getLong("total_time_ns") >= 0);
                assertTrue(rs.getLong("cumulative_time_ns") >= 0);
            }
        }

        @Test
        void writesCallerRelationships() throws Exception {
            // A calls B
            tracker.enter("com.example.A", "methodA", "A.java", 1, null);
            tracker.enter("com.example.B", "methodB", "B.java", 10, null);
            tracker.exit();
            tracker.exit();

            String dbPath = tempDir.resolve("callers_trace.sqlite").toString();
            PstatsWriter.flush(dbPath);

            try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + dbPath);
                 Statement stmt = conn.createStatement()) {

                // Get B's callers
                ResultSet rs = stmt.executeQuery(
                    "SELECT callers, num_callers FROM pstats WHERE function='methodB'");
                assertTrue(rs.next());
                assertEquals(1, rs.getInt("num_callers"));

                String callersJson = rs.getString("callers");
                assertNotNull(callersJson);

                // Parse and verify the JSON format: [{"key": [filename, line, funcname], "value": count}]
                Gson gson = new Gson();
                List<Map<String, Object>> callersList = gson.fromJson(callersJson,
                    new TypeToken<List<Map<String, Object>>>(){}.getType());
                assertEquals(1, callersList.size());

                Map<String, Object> callerEntry = callersList.get(0);
                assertTrue(callerEntry.containsKey("key"));
                assertTrue(callerEntry.containsKey("value"));

                @SuppressWarnings("unchecked")
                List<Object> keyList = (List<Object>) callerEntry.get("key");
                assertEquals("A.java", keyList.get(0));
                assertEquals(1.0, keyList.get(1)); // Gson parses ints as doubles
                assertEquals("methodA", keyList.get(2));
                assertEquals(1.0, callerEntry.get("value")); // call count
            }
        }

        @Test
        void multipleFunctionsWritten() throws Exception {
            tracker.enter("com.example.A", "a", "A.java", 1, null);
            tracker.exit();
            tracker.enter("com.example.B", "b", "B.java", 1, null);
            tracker.exit();
            tracker.enter("com.example.C", "c", "C.java", 1, null);
            tracker.exit();

            String dbPath = tempDir.resolve("multi_trace.sqlite").toString();
            PstatsWriter.flush(dbPath);

            try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + dbPath);
                 Statement stmt = conn.createStatement()) {

                ResultSet rs = stmt.executeQuery("SELECT COUNT(*) as cnt FROM pstats");
                assertTrue(rs.next());
                assertEquals(3, rs.getInt("cnt"));
            }
        }
    }

    @Nested
    class FunctionCallsDataTests {

        @Test
        void writesArgumentData() throws Exception {
            tracker.enter("com.example.Foo", "bar", "Foo.java", 10, new Object[]{"hello", 42});
            tracker.exit();

            String dbPath = tempDir.resolve("args_trace.sqlite").toString();
            PstatsWriter.flush(dbPath);

            try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + dbPath);
                 Statement stmt = conn.createStatement()) {

                ResultSet rs = stmt.executeQuery("SELECT * FROM function_calls");
                assertTrue(rs.next(), "Should have function_calls entry");

                assertEquals("call", rs.getString("type"));
                assertEquals("bar", rs.getString("function"));
                assertEquals("com.example.Foo", rs.getString("classname"));
                assertEquals("Foo.java", rs.getString("filename"));
                assertEquals(10, rs.getInt("line_number"));

                byte[] args = rs.getBytes("args");
                assertNotNull(args);
                assertTrue(args.length > 0, "Serialized args should be non-empty");
            }
        }

        @Test
        void multipleCallsCaptured() throws Exception {
            tracker.setMaxFunctionCount(10);
            for (int i = 0; i < 3; i++) {
                tracker.enter("com.example.Foo", "bar", "Foo.java", 10, new Object[]{i});
                tracker.exit();
            }

            String dbPath = tempDir.resolve("multi_args_trace.sqlite").toString();
            PstatsWriter.flush(dbPath);

            try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + dbPath);
                 Statement stmt = conn.createStatement()) {

                ResultSet rs = stmt.executeQuery("SELECT COUNT(*) as cnt FROM function_calls");
                assertTrue(rs.next());
                assertEquals(3, rs.getInt("cnt"));
            }
        }
    }

    @Nested
    class TotalTimeTests {

        @Test
        void writesTotalTime() throws Exception {
            tracker.markStart();
            tracker.enter("com.example.Foo", "bar", "Foo.java", 10, null);
            tracker.exit();
            tracker.markEnd();

            String dbPath = tempDir.resolve("total_time_trace.sqlite").toString();
            PstatsWriter.flush(dbPath);

            try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + dbPath);
                 Statement stmt = conn.createStatement()) {

                ResultSet rs = stmt.executeQuery("SELECT time_ns FROM total_time");
                assertTrue(rs.next(), "Should have total_time entry");
                long timeNs = rs.getLong("time_ns");
                assertTrue(timeNs >= 0, "Total time should be non-negative");
            }
        }
    }

    @Nested
    class PythonCompatibilityTests {

        @Test
        void schemaMatchesPythonTracer() throws Exception {
            // This test verifies the SQLite schema matches what Python's ProfileStats expects
            tracker.enter("com.example.Foo", "bar", "/project/src/main/java/com/example/Foo.java", 10, null);
            tracker.exit();

            String dbPath = tempDir.resolve("compat_trace.sqlite").toString();
            PstatsWriter.flush(dbPath);

            try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + dbPath);
                 Statement stmt = conn.createStatement()) {

                // ProfileStats does: SELECT * FROM pstats
                // and expects columns in this exact order:
                // filename, line_number, function, class_name,
                // call_count_nonrecursive, num_callers, total_time_ns, cumulative_time_ns, callers
                ResultSet rs = stmt.executeQuery("SELECT * FROM pstats LIMIT 1");
                assertTrue(rs.next());

                // Verify column names match Python expectations
                var meta = rs.getMetaData();
                assertEquals("filename", meta.getColumnName(1));
                assertEquals("line_number", meta.getColumnName(2));
                assertEquals("function", meta.getColumnName(3));
                assertEquals("class_name", meta.getColumnName(4));
                assertEquals("call_count_nonrecursive", meta.getColumnName(5));
                assertEquals("num_callers", meta.getColumnName(6));
                assertEquals("total_time_ns", meta.getColumnName(7));
                assertEquals("cumulative_time_ns", meta.getColumnName(8));
                assertEquals("callers", meta.getColumnName(9));

                // Verify callers is valid JSON in the expected format
                String callers = rs.getString("callers");
                assertNotNull(callers);
                Gson gson = new Gson();
                List<Map<String, Object>> parsed = gson.fromJson(callers,
                    new TypeToken<List<Map<String, Object>>>(){}.getType());
                assertNotNull(parsed);

                // get_trace_total_run_time_ns does: SELECT time_ns FROM total_time
                rs = stmt.executeQuery("SELECT time_ns FROM total_time");
                assertTrue(rs.next());
                assertTrue(rs.getLong("time_ns") >= 0);
            }
        }
    }
}
