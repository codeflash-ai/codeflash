package codeflash.runtime;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Codeflash Helper - Test Instrumentation for Java
 *
 * This class provides timing instrumentation for Java tests, mirroring the
 * behavior of the JavaScript codeflash package.
 *
 * Usage in instrumented tests:
 *   import codeflash.runtime.CodeflashHelper;
 *
 *   // For behavior verification (writes to SQLite):
 *   Object result = CodeflashHelper.capture("testModule", "testClass", "testFunc",
 *       "funcName", () -> targetMethod(arg1, arg2));
 *
 *   // For performance benchmarking:
 *   Object result = CodeflashHelper.capturePerf("testModule", "testClass", "testFunc",
 *       "funcName", () -> targetMethod(arg1, arg2));
 *
 * Environment Variables:
 *   CODEFLASH_OUTPUT_FILE - Path to write results SQLite file
 *   CODEFLASH_LOOP_INDEX - Current benchmark loop iteration (default: 1)
 *   CODEFLASH_TEST_ITERATION - Test iteration number (default: 0)
 *   CODEFLASH_MODE - "behavior" or "performance"
 */
public class CodeflashHelper {

    private static final String OUTPUT_FILE = System.getenv("CODEFLASH_OUTPUT_FILE");
    private static final int LOOP_INDEX = parseIntOrDefault(System.getenv("CODEFLASH_LOOP_INDEX"), 1);
    private static final String MODE = System.getenv("CODEFLASH_MODE");

    // Track invocation counts per test method for unique iteration IDs
    private static final ConcurrentHashMap<String, AtomicInteger> invocationCounts = new ConcurrentHashMap<>();

    // Database connection (lazily initialized)
    private static Connection dbConnection = null;
    private static boolean dbInitialized = false;

    /**
     * Functional interface for wrapping void method calls.
     */
    @FunctionalInterface
    public interface VoidCallable {
        void call() throws Exception;
    }

    /**
     * Functional interface for wrapping method calls that return a value.
     */
    @FunctionalInterface
    public interface Callable<T> {
        T call() throws Exception;
    }

    /**
     * Capture behavior and timing for a method call that returns a value.
     */
    public static <T> T capture(
            String testModulePath,
            String testClassName,
            String testFunctionName,
            String functionGettingTested,
            Callable<T> callable
    ) throws Exception {
        String invocationKey = testModulePath + ":" + testClassName + ":" + testFunctionName + ":" + functionGettingTested;
        int iterationId = getNextIterationId(invocationKey);

        long startTime = System.nanoTime();
        T result;
        try {
            result = callable.call();
        } finally {
            long endTime = System.nanoTime();
            long durationNs = endTime - startTime;

            // Write to SQLite for behavior verification
            writeResultToSqlite(
                    testModulePath,
                    testClassName,
                    testFunctionName,
                    functionGettingTested,
                    LOOP_INDEX,
                    iterationId,
                    durationNs,
                    null, // return_value - TODO: serialize if needed
                    "output"
            );

            // Print timing marker for stdout parsing (backup method)
            printTimingMarker(testModulePath, testClassName, functionGettingTested, LOOP_INDEX, iterationId, durationNs);
        }
        return result;
    }

    /**
     * Capture behavior and timing for a void method call.
     */
    public static void captureVoid(
            String testModulePath,
            String testClassName,
            String testFunctionName,
            String functionGettingTested,
            VoidCallable callable
    ) throws Exception {
        String invocationKey = testModulePath + ":" + testClassName + ":" + testFunctionName + ":" + functionGettingTested;
        int iterationId = getNextIterationId(invocationKey);

        long startTime = System.nanoTime();
        try {
            callable.call();
        } finally {
            long endTime = System.nanoTime();
            long durationNs = endTime - startTime;

            // Write to SQLite
            writeResultToSqlite(
                    testModulePath,
                    testClassName,
                    testFunctionName,
                    functionGettingTested,
                    LOOP_INDEX,
                    iterationId,
                    durationNs,
                    null,
                    "output"
            );

            // Print timing marker
            printTimingMarker(testModulePath, testClassName, functionGettingTested, LOOP_INDEX, iterationId, durationNs);
        }
    }

    /**
     * Capture timing for performance benchmarking (method with return value).
     */
    public static <T> T capturePerf(
            String testModulePath,
            String testClassName,
            String testFunctionName,
            String functionGettingTested,
            Callable<T> callable
    ) throws Exception {
        String invocationKey = testModulePath + ":" + testClassName + ":" + testFunctionName + ":" + functionGettingTested;
        int iterationId = getNextIterationId(invocationKey);

        // Print start marker
        printStartMarker(testModulePath, testClassName, functionGettingTested, LOOP_INDEX, iterationId);

        long startTime = System.nanoTime();
        T result;
        try {
            result = callable.call();
        } finally {
            long endTime = System.nanoTime();
            long durationNs = endTime - startTime;

            // Write to SQLite for performance data
            writeResultToSqlite(
                    testModulePath,
                    testClassName,
                    testFunctionName,
                    functionGettingTested,
                    LOOP_INDEX,
                    iterationId,
                    durationNs,
                    null,
                    "output"
            );

            // Print end marker with timing
            printTimingMarker(testModulePath, testClassName, functionGettingTested, LOOP_INDEX, iterationId, durationNs);
        }
        return result;
    }

    /**
     * Capture timing for performance benchmarking (void method).
     */
    public static void capturePerfVoid(
            String testModulePath,
            String testClassName,
            String testFunctionName,
            String functionGettingTested,
            VoidCallable callable
    ) throws Exception {
        String invocationKey = testModulePath + ":" + testClassName + ":" + testFunctionName + ":" + functionGettingTested;
        int iterationId = getNextIterationId(invocationKey);

        // Print start marker
        printStartMarker(testModulePath, testClassName, functionGettingTested, LOOP_INDEX, iterationId);

        long startTime = System.nanoTime();
        try {
            callable.call();
        } finally {
            long endTime = System.nanoTime();
            long durationNs = endTime - startTime;

            // Write to SQLite
            writeResultToSqlite(
                    testModulePath,
                    testClassName,
                    testFunctionName,
                    functionGettingTested,
                    LOOP_INDEX,
                    iterationId,
                    durationNs,
                    null,
                    "output"
            );

            // Print end marker with timing
            printTimingMarker(testModulePath, testClassName, functionGettingTested, LOOP_INDEX, iterationId, durationNs);
        }
    }

    /**
     * Get the next iteration ID for a given invocation key.
     */
    private static int getNextIterationId(String invocationKey) {
        return invocationCounts.computeIfAbsent(invocationKey, k -> new AtomicInteger(0)).incrementAndGet();
    }

    /**
     * Print timing marker to stdout (format matches Python/JS).
     * Format: !######testModule:testClass:funcName:loopIndex:iterationId:durationNs######!
     */
    private static void printTimingMarker(
            String testModule,
            String testClass,
            String funcName,
            int loopIndex,
            int iterationId,
            long durationNs
    ) {
        System.out.println("!######" + testModule + ":" + testClass + ":" + funcName + ":" +
                loopIndex + ":" + iterationId + ":" + durationNs + "######!");
    }

    /**
     * Print start marker for performance tests.
     * Format: !$######testModule:testClass:funcName:loopIndex:iterationId######$!
     */
    private static void printStartMarker(
            String testModule,
            String testClass,
            String funcName,
            int loopIndex,
            int iterationId
    ) {
        System.out.println("!$######" + testModule + ":" + testClass + ":" + funcName + ":" +
                loopIndex + ":" + iterationId + "######$!");
    }

    /**
     * Write test result to SQLite database.
     */
    private static synchronized void writeResultToSqlite(
            String testModulePath,
            String testClassName,
            String testFunctionName,
            String functionGettingTested,
            int loopIndex,
            int iterationId,
            long runtime,
            byte[] returnValue,
            String verificationType
    ) {
        if (OUTPUT_FILE == null || OUTPUT_FILE.isEmpty()) {
            return;
        }

        try {
            ensureDbInitialized();
            if (dbConnection == null) {
                return;
            }

            String sql = "INSERT INTO test_results " +
                    "(test_module_path, test_class_name, test_function_name, function_getting_tested, " +
                    "loop_index, iteration_id, runtime, return_value, verification_type) " +
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)";

            try (PreparedStatement stmt = dbConnection.prepareStatement(sql)) {
                stmt.setString(1, testModulePath);
                stmt.setString(2, testClassName);
                stmt.setString(3, testFunctionName);
                stmt.setString(4, functionGettingTested);
                stmt.setInt(5, loopIndex);
                stmt.setInt(6, iterationId);
                stmt.setLong(7, runtime);
                stmt.setBytes(8, returnValue);
                stmt.setString(9, verificationType);
                stmt.executeUpdate();
            }
        } catch (SQLException e) {
            System.err.println("CodeflashHelper: Failed to write to SQLite: " + e.getMessage());
        }
    }

    /**
     * Ensure the database is initialized.
     */
    private static void ensureDbInitialized() {
        if (dbInitialized) {
            return;
        }
        dbInitialized = true;

        if (OUTPUT_FILE == null || OUTPUT_FILE.isEmpty()) {
            return;
        }

        try {
            // Load SQLite JDBC driver
            Class.forName("org.sqlite.JDBC");

            // Create parent directories if needed
            File dbFile = new File(OUTPUT_FILE);
            File parentDir = dbFile.getParentFile();
            if (parentDir != null && !parentDir.exists()) {
                parentDir.mkdirs();
            }

            // Connect to database
            dbConnection = DriverManager.getConnection("jdbc:sqlite:" + OUTPUT_FILE);

            // Create table if not exists
            String createTableSql = "CREATE TABLE IF NOT EXISTS test_results (" +
                    "test_module_path TEXT, " +
                    "test_class_name TEXT, " +
                    "test_function_name TEXT, " +
                    "function_getting_tested TEXT, " +
                    "loop_index INTEGER, " +
                    "iteration_id INTEGER, " +
                    "runtime INTEGER, " +
                    "return_value BLOB, " +
                    "verification_type TEXT" +
                    ")";

            try (Statement stmt = dbConnection.createStatement()) {
                stmt.execute(createTableSql);
            }

            // Register shutdown hook to close connection
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                try {
                    if (dbConnection != null && !dbConnection.isClosed()) {
                        dbConnection.close();
                    }
                } catch (SQLException e) {
                    // Ignore
                }
            }));

        } catch (ClassNotFoundException e) {
            System.err.println("CodeflashHelper: SQLite JDBC driver not found. " +
                    "Add sqlite-jdbc to your dependencies. Timing will still be captured via stdout.");
        } catch (SQLException e) {
            System.err.println("CodeflashHelper: Failed to initialize SQLite: " + e.getMessage());
        }
    }

    /**
     * Parse int with default value.
     */
    private static int parseIntOrDefault(String value, int defaultValue) {
        if (value == null || value.isEmpty()) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }
}
