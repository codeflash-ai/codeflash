package com.codeflash.tracer;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public final class TraceWriter {

    private final Connection connection;
    private final BlockingQueue<WriteTask> writeQueue;
    private final Thread writerThread;
    private final AtomicBoolean running;

    private PreparedStatement insertFunctionCall;
    private PreparedStatement insertMetadata;

    public TraceWriter(String dbPath) {
        this.writeQueue = new LinkedBlockingQueue<>();
        this.running = new AtomicBoolean(true);

        try {
            Path path = Paths.get(dbPath).toAbsolutePath();
            path.getParent().toFile().mkdirs();
            this.connection = DriverManager.getConnection("jdbc:sqlite:" + path);
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
            stmt.execute("PRAGMA journal_mode=WAL");
            stmt.execute("PRAGMA synchronous=NORMAL");

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
        while (running.get() || !writeQueue.isEmpty()) {
            try {
                WriteTask task = writeQueue.poll(100, TimeUnit.MILLISECONDS);
                if (task != null) {
                    task.execute(this);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (SQLException e) {
                System.err.println("[codeflash-tracer] Write error: " + e.getMessage());
            }
        }

        // Drain remaining
        WriteTask task;
        while ((task = writeQueue.poll()) != null) {
            try {
                task.execute(this);
            } catch (SQLException e) {
                System.err.println("[codeflash-tracer] Write error: " + e.getMessage());
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

        try {
            if (insertFunctionCall != null) insertFunctionCall.close();
            if (insertMetadata != null) insertMetadata.close();
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

        @Override
        public void execute(TraceWriter writer) throws SQLException {
            writer.insertFunctionCall.setString(1, type);
            writer.insertFunctionCall.setString(2, function);
            writer.insertFunctionCall.setString(3, classname);
            writer.insertFunctionCall.setString(4, filename);
            writer.insertFunctionCall.setInt(5, lineNumber);
            writer.insertFunctionCall.setString(6, descriptor);
            writer.insertFunctionCall.setLong(7, timeNs);
            writer.insertFunctionCall.setBytes(8, argsBlob);
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
