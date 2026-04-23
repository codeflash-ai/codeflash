package com.codeflash.tracer;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public final class TraceWriter {

    private static final int BATCH_SIZE = 256;
    private static final int QUEUE_CAPACITY = 65536;

    private final Connection connection;
    private final Path diskPath;
    private final boolean inMemory;
    private final BlockingQueue<WriteTask> writeQueue;
    private final Thread writerThread;
    private final AtomicBoolean running;

    private PreparedStatement insertFunctionCall;
    private PreparedStatement insertMetadata;

    public TraceWriter(String dbPath, boolean inMemory) {
        this.diskPath = Paths.get(dbPath).toAbsolutePath();
        this.diskPath.getParent().toFile().mkdirs();
        this.inMemory = inMemory;
        this.writeQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
        this.running = new AtomicBoolean(true);

        try {
            if (inMemory) {
                // In-memory database for maximum write performance; flushed to disk via VACUUM INTO at close()
                this.connection = DriverManager.getConnection("jdbc:sqlite::memory:");
            } else {
                this.connection = DriverManager.getConnection("jdbc:sqlite:" + this.diskPath);
            }
            initializeSchema();
            prepareStatements();

            this.writerThread = new Thread(this::writerLoop, "codeflash-trace-writer");
            this.writerThread.setDaemon(true);
            this.writerThread.start();

        } catch (SQLException e) {
            throw new RuntimeException("Failed to initialize TraceWriter: " + e.getMessage(), e);
        }
    }

    private void initializeSchema() throws SQLException {
        try (Statement stmt = connection.createStatement()) {
            if (!inMemory) {
                stmt.execute("PRAGMA journal_mode=WAL");
                stmt.execute("PRAGMA synchronous=NORMAL");
                stmt.execute("PRAGMA cache_size=-16000");
                stmt.execute("PRAGMA temp_store=MEMORY");
            }

            stmt.execute(
                "CREATE TABLE IF NOT EXISTS function_calls(" +
                "id INTEGER PRIMARY KEY AUTOINCREMENT, " +
                "type TEXT, " +
                "function TEXT, " +
                "classname TEXT, " +
                "filename TEXT, " +
                "line_number INTEGER, " +
                "descriptor TEXT, " +
                "time_ns INTEGER, " +
                "args BLOB)"
            );

            stmt.execute(
                "CREATE TABLE IF NOT EXISTS metadata(" +
                "key TEXT PRIMARY KEY, " +
                "value TEXT)"
            );

            stmt.execute("CREATE INDEX IF NOT EXISTS idx_fc_class_func ON function_calls(classname, function)");
        }
        // Keep autocommit off for writer performance — commit explicitly per batch
        connection.setAutoCommit(false);
    }

    private void prepareStatements() throws SQLException {
        insertFunctionCall = connection.prepareStatement(
            "INSERT INTO function_calls (type, function, classname, filename, line_number, descriptor, time_ns, args) " +
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        );
        insertMetadata = connection.prepareStatement(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)"
        );
    }

    public void recordFunctionCall(String type, String function, String classname,
                                   String filename, int lineNumber, String descriptor,
                                   long timeNs, byte[] argsBlob) {
        writeQueue.offer(new FunctionCallTask(type, function, classname, filename,
                lineNumber, descriptor, timeNs, argsBlob));
    }

    public void writeMetadata(Map<String, String> metadata) {
        for (Map.Entry<String, String> entry : metadata.entrySet()) {
            writeQueue.offer(new MetadataTask(entry.getKey(), entry.getValue()));
        }
    }

    private void writerLoop() {
        List<WriteTask> batch = new ArrayList<>(BATCH_SIZE);

        while (running.get() || !writeQueue.isEmpty()) {
            try {
                WriteTask task = writeQueue.poll(100, TimeUnit.MILLISECONDS);
                if (task == null) {
                    continue;
                }
                batch.add(task);
                writeQueue.drainTo(batch, BATCH_SIZE - 1);
                executeBatch(batch);
                batch.clear();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }

        // Drain remaining
        writeQueue.drainTo(batch);
        if (!batch.isEmpty()) {
            executeBatch(batch);
        }
    }

    private void executeBatch(List<WriteTask> batch) {
        if (batch.isEmpty()) {
            return;
        }

        boolean hasFunctionCalls = false;
        try {
            for (WriteTask task : batch) {
                if (task instanceof FunctionCallTask) {
                    ((FunctionCallTask) task).bindParameters(this);
                    insertFunctionCall.addBatch();
                    hasFunctionCalls = true;
                } else {
                    task.execute(this);
                }
            }

            if (hasFunctionCalls) {
                insertFunctionCall.executeBatch();
            }

            connection.commit();
        } catch (SQLException e) {
            System.err.println("[codeflash-tracer] Batch write error (" + batch.size() + " tasks): " + e.getMessage());
            try {
                connection.rollback();
            } catch (SQLException re) {
                System.err.println("[codeflash-tracer] Rollback failed: " + re.getMessage());
            }
        }
    }

    public void flush() {
        while (!writeQueue.isEmpty()) {
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }

    public void close() {
        running.set(false);
        try {
            writerThread.join(5000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        // Close prepared statements first — required before VACUUM
        try {
            if (insertFunctionCall != null) insertFunctionCall.close();
            if (insertMetadata != null) insertMetadata.close();
        } catch (SQLException e) {
            System.err.println("[codeflash-tracer] Error closing statements: " + e.getMessage());
        }

        if (inMemory) {
            try {
                connection.commit();
                connection.setAutoCommit(true);
                try (Statement stmt = connection.createStatement()) {
                    stmt.execute("VACUUM INTO '" + diskPath.toString().replace("'", "''") + "'");
                }
            } catch (SQLException e) {
                System.err.println("[codeflash-tracer] Failed to write trace DB to disk: " + e.getMessage());
            }
        }

        try {
            if (connection != null) connection.close();
        } catch (SQLException e) {
            System.err.println("[codeflash-tracer] Error closing TraceWriter: " + e.getMessage());
        }
    }

    // Task types

    private interface WriteTask {
        void execute(TraceWriter writer) throws SQLException;
    }

    private static class FunctionCallTask implements WriteTask {
        final String type;
        final String function;
        final String classname;
        final String filename;
        final int lineNumber;
        final String descriptor;
        final long timeNs;
        final byte[] argsBlob;

        FunctionCallTask(String type, String function, String classname,
                         String filename, int lineNumber, String descriptor,
                         long timeNs, byte[] argsBlob) {
            this.type = type;
            this.function = function;
            this.classname = classname;
            this.filename = filename;
            this.lineNumber = lineNumber;
            this.descriptor = descriptor;
            this.timeNs = timeNs;
            this.argsBlob = argsBlob;
        }

        void bindParameters(TraceWriter writer) throws SQLException {
            writer.insertFunctionCall.setString(1, type);
            writer.insertFunctionCall.setString(2, function);
            writer.insertFunctionCall.setString(3, classname);
            writer.insertFunctionCall.setString(4, filename);
            writer.insertFunctionCall.setInt(5, lineNumber);
            writer.insertFunctionCall.setString(6, descriptor);
            writer.insertFunctionCall.setLong(7, timeNs);
            writer.insertFunctionCall.setBytes(8, argsBlob);
        }

        @Override
        public void execute(TraceWriter writer) throws SQLException {
            bindParameters(writer);
            writer.insertFunctionCall.executeUpdate();
        }
    }

    private static class MetadataTask implements WriteTask {
        final String key;
        final String value;

        MetadataTask(String key, String value) {
            this.key = key;
            this.value = value;
        }

        @Override
        public void execute(TraceWriter writer) throws SQLException {
            writer.insertMetadata.setString(1, key);
            writer.insertMetadata.setString(2, value);
            writer.insertMetadata.executeUpdate();
        }
    }
}
