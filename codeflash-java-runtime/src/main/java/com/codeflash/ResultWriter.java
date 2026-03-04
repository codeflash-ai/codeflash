package com.codeflash;

import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Writes benchmark and behavior capture results to SQLite database.
 *
 * Uses a background thread for non-blocking writes to minimize
 * impact on benchmark measurements.
 *
 * Database schema:
 * - invocations: call_id, method_id, args_blob, result_blob, error_blob, start_time, end_time
 * - benchmarks: method_id, duration_ns, timestamp
 * - benchmark_results: method_id, mean_ns, stddev_ns, min_ns, max_ns, p50_ns, p90_ns, p99_ns, iterations
 */
public final class ResultWriter {

    private final Path dbPath;
    private final Connection connection;
    private final BlockingQueue<WriteTask> writeQueue;
    private final Thread writerThread;
    private final AtomicBoolean running;

    // Prepared statements for performance
    private PreparedStatement insertInvocationInput;
    private PreparedStatement updateInvocationOutput;
    private PreparedStatement updateInvocationError;
    private PreparedStatement insertBenchmark;
    private PreparedStatement insertBenchmarkResult;

    /**
     * Create a new ResultWriter that writes to the specified database file.
     *
     * @param dbPath Path to SQLite database file (will be created if not exists)
     */
    public ResultWriter(Path dbPath) {
        this.dbPath = dbPath;
        this.writeQueue = new LinkedBlockingQueue<>();
        this.running = new AtomicBoolean(true);

        try {
            // Create connection and initialize schema
            this.connection = DriverManager.getConnection("jdbc:sqlite:" + dbPath.toAbsolutePath());
            initializeSchema();
            prepareStatements();

            // Start background writer thread
            this.writerThread = new Thread(this::writerLoop, "codeflash-writer");
            this.writerThread.setDaemon(true);
            this.writerThread.start();

        } catch (SQLException e) {
            throw new RuntimeException("Failed to initialize ResultWriter: " + e.getMessage(), e);
        }
    }

    private void initializeSchema() throws SQLException {
        try (Statement stmt = connection.createStatement()) {
            // Invocations table - stores input/output/error for each function call as BLOBs
            stmt.execute(
                "CREATE TABLE IF NOT EXISTS invocations (" +
                "call_id INTEGER PRIMARY KEY, " +
                "method_id TEXT NOT NULL, " +
                "args_blob BLOB, " +
                "result_blob BLOB, " +
                "error_blob BLOB, " +
                "start_time INTEGER, " +
                "end_time INTEGER)"
            );

            // Benchmarks table - stores individual benchmark timings
            stmt.execute(
                "CREATE TABLE IF NOT EXISTS benchmarks (" +
                "id INTEGER PRIMARY KEY AUTOINCREMENT, " +
                "method_id TEXT NOT NULL, " +
                "duration_ns INTEGER NOT NULL, " +
                "timestamp INTEGER NOT NULL)"
            );

            // Benchmark results table - stores aggregated statistics
            stmt.execute(
                "CREATE TABLE IF NOT EXISTS benchmark_results (" +
                "method_id TEXT PRIMARY KEY, " +
                "mean_ns INTEGER NOT NULL, " +
                "stddev_ns INTEGER NOT NULL, " +
                "min_ns INTEGER NOT NULL, " +
                "max_ns INTEGER NOT NULL, " +
                "p50_ns INTEGER NOT NULL, " +
                "p90_ns INTEGER NOT NULL, " +
                "p99_ns INTEGER NOT NULL, " +
                "iterations INTEGER NOT NULL, " +
                "coefficient_of_variation REAL NOT NULL)"
            );

            // Create indexes for faster queries
            stmt.execute("CREATE INDEX IF NOT EXISTS idx_invocations_method ON invocations(method_id)");
            stmt.execute("CREATE INDEX IF NOT EXISTS idx_benchmarks_method ON benchmarks(method_id)");
        }
    }

    private void prepareStatements() throws SQLException {
        insertInvocationInput = connection.prepareStatement(
            "INSERT INTO invocations (call_id, method_id, args_blob, start_time) VALUES (?, ?, ?, ?)"
        );
        updateInvocationOutput = connection.prepareStatement(
            "UPDATE invocations SET result_blob = ?, end_time = ? WHERE call_id = ?"
        );
        updateInvocationError = connection.prepareStatement(
            "UPDATE invocations SET error_blob = ?, end_time = ? WHERE call_id = ?"
        );
        insertBenchmark = connection.prepareStatement(
            "INSERT INTO benchmarks (method_id, duration_ns, timestamp) VALUES (?, ?, ?)"
        );
        insertBenchmarkResult = connection.prepareStatement(
            "INSERT OR REPLACE INTO benchmark_results " +
            "(method_id, mean_ns, stddev_ns, min_ns, max_ns, p50_ns, p90_ns, p99_ns, iterations, coefficient_of_variation) " +
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        );
    }

    /**
     * Record function input (beginning of invocation).
     */
    public void recordInput(long callId, String methodId, byte[] argsBlob, long startTime) {
        writeQueue.offer(new WriteTask(WriteType.INPUT, callId, methodId, argsBlob, null, null, startTime, 0, null));
    }

    /**
     * Record function output (successful completion).
     */
    public void recordOutput(long callId, String methodId, byte[] resultBlob, long endTime) {
        writeQueue.offer(new WriteTask(WriteType.OUTPUT, callId, methodId, null, resultBlob, null, 0, endTime, null));
    }

    /**
     * Record function error (exception thrown).
     */
    public void recordError(long callId, String methodId, byte[] errorBlob, long endTime) {
        writeQueue.offer(new WriteTask(WriteType.ERROR, callId, methodId, null, null, errorBlob, 0, endTime, null));
    }

    /**
     * Record a single benchmark timing.
     */
    public void recordBenchmark(String methodId, long durationNs, long timestamp) {
        writeQueue.offer(new WriteTask(WriteType.BENCHMARK, 0, methodId, null, null, null, durationNs, timestamp, null));
    }

    /**
     * Record aggregated benchmark results.
     */
    public void recordBenchmarkResult(String methodId, BenchmarkResult result) {
        writeQueue.offer(new WriteTask(WriteType.BENCHMARK_RESULT, 0, methodId, null, null, null, 0, 0, result));
    }

    /**
     * Background writer loop - processes write tasks from queue.
     */
    private void writerLoop() {
        while (running.get() || !writeQueue.isEmpty()) {
            try {
                WriteTask task = writeQueue.poll(100, TimeUnit.MILLISECONDS);
                if (task != null) {
                    executeTask(task);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (SQLException e) {
                System.err.println("CodeFlash ResultWriter error: " + e.getMessage());
            }
        }

        // Process remaining tasks
        WriteTask task;
        while ((task = writeQueue.poll()) != null) {
            try {
                executeTask(task);
            } catch (SQLException e) {
                System.err.println("CodeFlash ResultWriter error: " + e.getMessage());
            }
        }
    }

    private void executeTask(WriteTask task) throws SQLException {
        switch (task.type) {
            case INPUT:
                insertInvocationInput.setLong(1, task.callId);
                insertInvocationInput.setString(2, task.methodId);
                insertInvocationInput.setBytes(3, task.argsBlob);
                insertInvocationInput.setLong(4, task.startTime);
                insertInvocationInput.executeUpdate();
                break;

            case OUTPUT:
                updateInvocationOutput.setBytes(1, task.resultBlob);
                updateInvocationOutput.setLong(2, task.endTime);
                updateInvocationOutput.setLong(3, task.callId);
                updateInvocationOutput.executeUpdate();
                break;

            case ERROR:
                updateInvocationError.setBytes(1, task.errorBlob);
                updateInvocationError.setLong(2, task.endTime);
                updateInvocationError.setLong(3, task.callId);
                updateInvocationError.executeUpdate();
                break;

            case BENCHMARK:
                insertBenchmark.setString(1, task.methodId);
                insertBenchmark.setLong(2, task.startTime); // duration stored in startTime field
                insertBenchmark.setLong(3, task.endTime);   // timestamp stored in endTime field
                insertBenchmark.executeUpdate();
                break;

            case BENCHMARK_RESULT:
                BenchmarkResult r = task.benchmarkResult;
                insertBenchmarkResult.setString(1, task.methodId);
                insertBenchmarkResult.setLong(2, r.getMean());
                insertBenchmarkResult.setLong(3, r.getStdDev());
                insertBenchmarkResult.setLong(4, r.getMin());
                insertBenchmarkResult.setLong(5, r.getMax());
                insertBenchmarkResult.setLong(6, r.getP50());
                insertBenchmarkResult.setLong(7, r.getP90());
                insertBenchmarkResult.setLong(8, r.getP99());
                insertBenchmarkResult.setInt(9, r.getIterationCount());
                insertBenchmarkResult.setDouble(10, r.getCoefficientOfVariation());
                insertBenchmarkResult.executeUpdate();
                break;
        }
    }

    /**
     * Flush all pending writes synchronously.
     */
    public void flush() {
        // Wait for queue to drain
        while (!writeQueue.isEmpty()) {
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }

    /**
     * Close the writer and database connection.
     */
    public void close() {
        running.set(false);

        try {
            writerThread.join(5000); // Wait up to 5 seconds
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        try {
            if (insertInvocationInput != null) insertInvocationInput.close();
            if (updateInvocationOutput != null) updateInvocationOutput.close();
            if (updateInvocationError != null) updateInvocationError.close();
            if (insertBenchmark != null) insertBenchmark.close();
            if (insertBenchmarkResult != null) insertBenchmarkResult.close();
            if (connection != null) connection.close();
        } catch (SQLException e) {
            System.err.println("Error closing ResultWriter: " + e.getMessage());
        }
    }

    /**
     * Get the database path.
     */
    public Path getDbPath() {
        return dbPath;
    }

    // Internal task class for queue
    private enum WriteType {
        INPUT, OUTPUT, ERROR, BENCHMARK, BENCHMARK_RESULT
    }

    private static class WriteTask {
        final WriteType type;
        final long callId;
        final String methodId;
        final byte[] argsBlob;
        final byte[] resultBlob;
        final byte[] errorBlob;
        final long startTime;
        final long endTime;
        final BenchmarkResult benchmarkResult;

        WriteTask(WriteType type, long callId, String methodId, byte[] argsBlob,
                  byte[] resultBlob, byte[] errorBlob, long startTime, long endTime,
                  BenchmarkResult benchmarkResult) {
            this.type = type;
            this.callId = callId;
            this.methodId = methodId;
            this.argsBlob = argsBlob;
            this.resultBlob = resultBlob;
            this.errorBlob = errorBlob;
            this.startTime = startTime;
            this.endTime = endTime;
            this.benchmarkResult = benchmarkResult;
        }
    }
}
