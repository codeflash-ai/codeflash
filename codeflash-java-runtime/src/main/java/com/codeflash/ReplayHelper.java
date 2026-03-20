package com.codeflash;

import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

import org.objectweb.asm.Type;

public class ReplayHelper {

    private final Connection traceDb;

    // Codeflash instrumentation state — read from environment variables once
    private final String mode;           // "behavior", "performance", or null
    private final int loopIndex;
    private final String testIteration;
    private final String outputFile;     // SQLite path for behavior capture
    private final int innerIterations;   // for performance looping

    // Behavior mode: lazily opened SQLite connection for writing results
    private Connection behaviorDb;
    private boolean behaviorDbInitialized;

    public ReplayHelper(String traceDbPath) {
        try {
            this.traceDb = DriverManager.getConnection("jdbc:sqlite:" + traceDbPath);
        } catch (SQLException e) {
            throw new RuntimeException("Failed to open trace database: " + traceDbPath, e);
        }

        // Read codeflash instrumentation env vars (set by the test runner)
        this.mode = System.getenv("CODEFLASH_MODE");
        this.loopIndex = parseIntEnv("CODEFLASH_LOOP_INDEX", 1);
        this.testIteration = getEnvOrDefault("CODEFLASH_TEST_ITERATION", "0");
        this.outputFile = System.getenv("CODEFLASH_OUTPUT_FILE");
        this.innerIterations = parseIntEnv("CODEFLASH_INNER_ITERATIONS", 10);
    }

    public void replay(String className, String methodName, String descriptor, int invocationIndex) throws Exception {
        // Deserialize args and resolve method (done once, outside timing)
        Object[] allArgs = loadArgs(className, methodName, descriptor, invocationIndex);
        Class<?> targetClass = Class.forName(className);

        Type[] paramTypes = Type.getArgumentTypes(descriptor);
        Class<?>[] paramClasses = new Class<?>[paramTypes.length];
        for (int i = 0; i < paramTypes.length; i++) {
            paramClasses[i] = typeToClass(paramTypes[i]);
        }

        Method method = targetClass.getDeclaredMethod(methodName, paramClasses);
        method.setAccessible(true);
        boolean isStatic = Modifier.isStatic(method.getModifiers());

        Object instance = null;
        if (!isStatic) {
            try {
                java.lang.reflect.Constructor<?> ctor = targetClass.getDeclaredConstructor();
                ctor.setAccessible(true);
                instance = ctor.newInstance();
            } catch (NoSuchMethodException e) {
                instance = new org.objenesis.ObjenesisStd().newInstance(targetClass);
            }
        }

        // Get the calling test method name from the stack trace
        String testMethodName = getCallingTestMethodName();
        // Module name = the test class that called us
        String testClassName = getCallingTestClassName();

        if ("behavior".equals(mode)) {
            replayBehavior(method, instance, allArgs, className, methodName, testClassName, testMethodName);
        } else if ("performance".equals(mode)) {
            replayPerformance(method, instance, allArgs, className, methodName, testClassName, testMethodName);
        } else {
            // No codeflash mode — just invoke (trace-only or manual testing)
            method.invoke(instance, allArgs);
        }
    }

    private void replayBehavior(Method method, Object instance, Object[] args,
                                 String className, String methodName,
                                 String testClassName, String testMethodName) throws Exception {
        String invId = testIteration + "_" + testMethodName;

        // Print start marker (same format as behavior instrumentation)
        System.out.println("!$######" + testClassName + ":" + testClassName + "." + testMethodName
                + ":" + methodName + ":" + loopIndex + ":" + invId + "######$!");

        long startNs = System.nanoTime();
        Object result;
        try {
            result = method.invoke(instance, args);
        } catch (java.lang.reflect.InvocationTargetException e) {
            throw (Exception) e.getCause();
        }
        long durationNs = System.nanoTime() - startNs;

        // Print end marker
        System.out.println("!######" + testClassName + ":" + testClassName + "." + testMethodName
                + ":" + methodName + ":" + loopIndex + ":" + invId + ":" + durationNs + "######!");

        // Write return value to SQLite for correctness comparison
        if (outputFile != null && !outputFile.isEmpty()) {
            writeBehaviorResult(testClassName, testMethodName, methodName, invId, durationNs, result);
        }
    }

    private void replayPerformance(Method method, Object instance, Object[] args,
                                    String className, String methodName,
                                    String testClassName, String testMethodName) throws Exception {
        // Performance mode: run inner loop for JIT warmup, print timing for each iteration
        int maxInner = innerIterations;
        for (int inner = 0; inner < maxInner; inner++) {
            int loopId = (loopIndex - 1) * maxInner + inner;
            String invId = testMethodName;

            // Print start marker
            System.out.println("!$######" + testClassName + ":" + testClassName + "." + testMethodName
                    + ":" + methodName + ":" + loopId + ":" + invId + "######$!");

            long startNs = System.nanoTime();
            try {
                method.invoke(instance, args);
            } catch (java.lang.reflect.InvocationTargetException e) {
                // Swallow — performance mode doesn't check correctness
            }
            long durationNs = System.nanoTime() - startNs;

            // Print end marker
            System.out.println("!######" + testClassName + ":" + testClassName + "." + testMethodName
                    + ":" + methodName + ":" + loopId + ":" + invId + ":" + durationNs + "######!");
        }
    }

    private void writeBehaviorResult(String testClassName, String testMethodName,
                                      String functionName, String invId,
                                      long durationNs, Object result) {
        try {
            ensureBehaviorDb();
            String sql = "INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)";
            try (PreparedStatement ps = behaviorDb.prepareStatement(sql)) {
                ps.setString(1, testClassName);              // test_module_path
                ps.setString(2, testClassName);              // test_class_name
                ps.setString(3, testMethodName);             // test_function_name
                ps.setString(4, functionName);               // function_getting_tested
                ps.setInt(5, loopIndex);                     // loop_index
                ps.setString(6, invId);                      // iteration_id
                ps.setLong(7, durationNs);                   // runtime
                ps.setBytes(8, serializeResult(result));     // return_value
                ps.setString(9, "function_call");            // verification_type
                ps.executeUpdate();
            }
        } catch (Exception e) {
            System.err.println("ReplayHelper: SQLite behavior write error: " + e.getMessage());
        }
    }

    private void ensureBehaviorDb() throws SQLException {
        if (behaviorDbInitialized) return;
        behaviorDbInitialized = true;
        behaviorDb = DriverManager.getConnection("jdbc:sqlite:" + outputFile);
        try (java.sql.Statement stmt = behaviorDb.createStatement()) {
            stmt.execute("CREATE TABLE IF NOT EXISTS test_results (" +
                    "test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, " +
                    "function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, " +
                    "runtime INTEGER, return_value BLOB, verification_type TEXT)");
        }
    }

    private byte[] serializeResult(Object result) {
        if (result == null) return null;
        try {
            return Serializer.serialize(result);
        } catch (Exception e) {
            // Fall back to String.valueOf if Kryo fails
            return String.valueOf(result).getBytes(java.nio.charset.StandardCharsets.UTF_8);
        }
    }

    private Object[] loadArgs(String className, String methodName, String descriptor, int invocationIndex)
            throws SQLException {
        byte[] argsBlob;
        try (PreparedStatement stmt = traceDb.prepareStatement(
                "SELECT args FROM function_calls " +
                "WHERE classname = ? AND function = ? AND descriptor = ? " +
                "ORDER BY time_ns LIMIT 1 OFFSET ?")) {
            stmt.setString(1, className);
            stmt.setString(2, methodName);
            stmt.setString(3, descriptor);
            stmt.setInt(4, invocationIndex);

            try (ResultSet rs = stmt.executeQuery()) {
                if (!rs.next()) {
                    throw new RuntimeException("No invocation found at index " + invocationIndex
                            + " for " + className + "." + methodName + descriptor);
                }
                argsBlob = rs.getBytes("args");
            }
        }

        Object deserialized = Serializer.deserialize(argsBlob);
        if (!(deserialized instanceof Object[])) {
            throw new RuntimeException("Deserialized args is not Object[], got: "
                    + (deserialized == null ? "null" : deserialized.getClass().getName()));
        }
        return (Object[]) deserialized;
    }

    private static String getCallingTestMethodName() {
        StackTraceElement[] stack = Thread.currentThread().getStackTrace();
        // Walk up: [0]=getStackTrace, [1]=this method, [2]=replay(), [3]=calling test method
        for (int i = 3; i < stack.length; i++) {
            String method = stack[i].getMethodName();
            if (method.startsWith("replay_")) {
                return method;
            }
        }
        return stack.length > 3 ? stack[3].getMethodName() : "unknown";
    }

    private static String getCallingTestClassName() {
        StackTraceElement[] stack = Thread.currentThread().getStackTrace();
        for (int i = 3; i < stack.length; i++) {
            String cls = stack[i].getClassName();
            if (cls.contains("ReplayTest") || cls.contains("replay")) {
                return cls;
            }
        }
        return stack.length > 3 ? stack[3].getClassName() : "unknown";
    }

    private static Class<?> typeToClass(Type type) throws ClassNotFoundException {
        switch (type.getSort()) {
            case Type.BOOLEAN: return boolean.class;
            case Type.BYTE:    return byte.class;
            case Type.CHAR:    return char.class;
            case Type.SHORT:   return short.class;
            case Type.INT:     return int.class;
            case Type.LONG:    return long.class;
            case Type.FLOAT:   return float.class;
            case Type.DOUBLE:  return double.class;
            case Type.VOID:    return void.class;
            case Type.ARRAY:
                Class<?> elementClass = typeToClass(type.getElementType());
                return java.lang.reflect.Array.newInstance(elementClass, 0).getClass();
            case Type.OBJECT:
                return Class.forName(type.getClassName());
            default:
                throw new ClassNotFoundException("Unknown type: " + type);
        }
    }

    private static int parseIntEnv(String name, int defaultValue) {
        String val = System.getenv(name);
        if (val == null || val.isEmpty()) return defaultValue;
        try { return Integer.parseInt(val); } catch (NumberFormatException e) { return defaultValue; }
    }

    private static String getEnvOrDefault(String name, String defaultValue) {
        String val = System.getenv(name);
        return (val != null && !val.isEmpty()) ? val : defaultValue;
    }

    public void close() {
        try { if (traceDb != null) traceDb.close(); } catch (SQLException e) {
            System.err.println("Error closing ReplayHelper trace db: " + e.getMessage());
        }
        try { if (behaviorDb != null) behaviorDb.close(); } catch (SQLException e) {
            System.err.println("Error closing ReplayHelper behavior db: " + e.getMessage());
        }
    }
}
