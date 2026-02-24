package com.codeflash.agent;

import com.google.gson.Gson;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Writes profiling data to SQLite, matching the Python tracer's schema exactly.
 *
 * Tables:
 * - pstats: per-function timing stats with caller relationships
 * - function_calls: captured arguments for replay test generation
 * - metadata: key-value pairs for tracer configuration
 * - total_time: total wall-clock time of the traced execution
 */
public final class PstatsWriter {

    private static final Gson GSON = new Gson();

    private PstatsWriter() {}

    /**
     * Write all accumulated profiling data to the specified SQLite file.
     */
    public static void flush(String outputPath) {
        CallTracker tracker = CallTracker.getInstance();
        tracker.markEnd();

        try {
            Class.forName("org.sqlite.JDBC");
        } catch (ClassNotFoundException e) {
            System.err.println("[codeflash-agent] SQLite JDBC driver not found: " + e.getMessage());
            return;
        }

        try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + outputPath)) {
            // PRAGMAs must be set before starting a transaction (autocommit=true)
            try (Statement stmt = conn.createStatement()) {
                stmt.execute("PRAGMA synchronous = OFF");
                stmt.execute("PRAGMA journal_mode = WAL");
            }

            conn.setAutoCommit(false);

            createTables(conn);
            writePstats(conn, tracker.getTimings());
            writeFunctionCalls(conn, tracker.getCapturedArgs());
            writeTotalTime(conn, tracker.getTotalTimeNs());

            conn.commit();
        } catch (SQLException e) {
            System.err.println("[codeflash-agent] Failed to write trace data: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void createTables(Connection conn) throws SQLException {
        try (Statement stmt = conn.createStatement()) {
            stmt.execute(
                "CREATE TABLE IF NOT EXISTS pstats ("
                + "filename TEXT, "
                + "line_number INTEGER, "
                + "function TEXT, "
                + "class_name TEXT, "
                + "call_count_nonrecursive INTEGER, "
                + "num_callers INTEGER, "
                + "total_time_ns INTEGER, "
                + "cumulative_time_ns INTEGER, "
                + "callers BLOB"
                + ")"
            );
            stmt.execute(
                "CREATE TABLE IF NOT EXISTS function_calls ("
                + "type TEXT, "
                + "function TEXT, "
                + "classname TEXT, "
                + "filename TEXT, "
                + "line_number INTEGER, "
                + "last_frame_address INTEGER, "
                + "time_ns INTEGER, "
                + "args BLOB"
                + ")"
            );
            stmt.execute(
                "CREATE TABLE IF NOT EXISTS metadata ("
                + "key TEXT PRIMARY KEY, "
                + "value TEXT"
                + ")"
            );
            stmt.execute(
                "CREATE TABLE IF NOT EXISTS total_time ("
                + "time_ns INTEGER"
                + ")"
            );
        }
    }

    private static void writePstats(Connection conn, ConcurrentHashMap<MethodKey, MethodStats> timings)
            throws SQLException {
        String sql = "INSERT INTO pstats "
            + "(filename, line_number, function, class_name, "
            + "call_count_nonrecursive, num_callers, total_time_ns, cumulative_time_ns, callers) "
            + "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)";

        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            int batchCount = 0;
            for (Map.Entry<MethodKey, MethodStats> entry : timings.entrySet()) {
                MethodKey key = entry.getKey();
                MethodStats stats = entry.getValue();

                ConcurrentHashMap<MethodKey, AtomicLong> callers = stats.getCallers();
                String callersJson = serializeCallers(callers);

                ps.setString(1, key.getFileName());
                ps.setInt(2, key.getLineNumber());
                ps.setString(3, key.getMethodName());
                ps.setString(4, key.getClassName());
                ps.setLong(5, stats.getCallCount());
                ps.setInt(6, callers.size());
                ps.setLong(7, stats.getTotalTimeNs());
                ps.setLong(8, stats.getCumulativeTimeNs());
                ps.setString(9, callersJson);

                ps.addBatch();
                batchCount++;

                if (batchCount % 1000 == 0) {
                    ps.executeBatch();
                }
            }
            if (batchCount % 1000 != 0) {
                ps.executeBatch();
            }
        }
    }

    private static void writeFunctionCalls(Connection conn, ConcurrentHashMap<MethodKey, List<byte[]>> capturedArgs)
            throws SQLException {
        String sql = "INSERT INTO function_calls "
            + "(type, function, classname, filename, line_number, last_frame_address, time_ns, args) "
            + "VALUES (?, ?, ?, ?, ?, ?, ?, ?)";

        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            int batchCount = 0;
            for (Map.Entry<MethodKey, List<byte[]>> entry : capturedArgs.entrySet()) {
                MethodKey key = entry.getKey();
                List<byte[]> argsList = entry.getValue();

                for (byte[] args : argsList) {
                    ps.setString(1, "call");
                    ps.setString(2, key.getMethodName());
                    ps.setString(3, key.getClassName());
                    ps.setString(4, key.getFileName());
                    ps.setInt(5, key.getLineNumber());
                    ps.setLong(6, 0);
                    ps.setLong(7, 0);
                    ps.setBytes(8, args);

                    ps.addBatch();
                    batchCount++;

                    if (batchCount % 1000 == 0) {
                        ps.executeBatch();
                    }
                }
            }
            if (batchCount % 1000 != 0) {
                ps.executeBatch();
            }
        }
    }

    private static void writeTotalTime(Connection conn, long totalTimeNs) throws SQLException {
        try (PreparedStatement ps = conn.prepareStatement("INSERT INTO total_time (time_ns) VALUES (?)")) {
            ps.setLong(1, totalTimeNs);
            ps.executeUpdate();
        }
    }

    /**
     * Serialize callers map to JSON matching the Python format:
     * [{"key": [filename, line, funcname], "value": count}, ...]
     */
    private static String serializeCallers(ConcurrentHashMap<MethodKey, AtomicLong> callers) {
        List<Map<String, Object>> entries = new ArrayList<>();
        for (Map.Entry<MethodKey, AtomicLong> entry : callers.entrySet()) {
            MethodKey caller = entry.getKey();
            long count = entry.getValue().get();

            Map<String, Object> jsonEntry = new HashMap<>();
            List<Object> keyList = new ArrayList<>();
            keyList.add(caller.getFileName());
            keyList.add(caller.getLineNumber());
            keyList.add(caller.getMethodName());
            jsonEntry.put("key", keyList);
            jsonEntry.put("value", count);
            entries.add(jsonEntry);
        }
        return GSON.toJson(entries);
    }
}
