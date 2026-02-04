package com.codeflash;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Compares test results between original and optimized code.
 *
 * Used by CodeFlash to verify that optimized code produces the
 * same outputs as the original code for the same inputs.
 *
 * Can be run as a CLI tool:
 * java -jar codeflash-runtime.jar original.db candidate.db
 */
public final class Comparator {

    private static final Gson GSON = new GsonBuilder()
            .serializeNulls()
            .setPrettyPrinting()
            .create();

    // Tolerance for floating point comparison
    private static final double EPSILON = 1e-9;

    private Comparator() {
        // Utility class
    }

    /**
     * Main entry point for CLI usage.
     *
     * @param args [originalDb, candidateDb]
     */
    public static void main(String[] args) {
        if (args.length != 2) {
            System.err.println("Usage: java -jar codeflash-runtime.jar <original.db> <candidate.db>");
            System.exit(1);
        }

        try {
            ComparisonResult result = compare(args[0], args[1]);
            System.out.println(GSON.toJson(result));
            System.exit(result.isEquivalent() ? 0 : 1);
        } catch (Exception e) {
            JsonObject error = new JsonObject();
            error.addProperty("error", e.getMessage());
            System.out.println(GSON.toJson(error));
            System.exit(2);
        }
    }

    /**
     * Compare two result databases.
     *
     * @param originalDbPath Path to original results database
     * @param candidateDbPath Path to candidate results database
     * @return Comparison result with list of differences
     */
    public static ComparisonResult compare(String originalDbPath, String candidateDbPath) throws SQLException {
        List<Diff> diffs = new ArrayList<>();

        try (Connection originalConn = DriverManager.getConnection("jdbc:sqlite:" + originalDbPath);
             Connection candidateConn = DriverManager.getConnection("jdbc:sqlite:" + candidateDbPath)) {

            // Get all invocations from original
            List<Invocation> originalInvocations = getInvocations(originalConn);
            List<Invocation> candidateInvocations = getInvocations(candidateConn);

            // Create lookup map for candidate invocations
            java.util.Map<Long, Invocation> candidateMap = new java.util.HashMap<>();
            for (Invocation inv : candidateInvocations) {
                candidateMap.put(inv.callId, inv);
            }

            // Compare each original invocation with candidate
            for (Invocation original : originalInvocations) {
                Invocation candidate = candidateMap.get(original.callId);

                if (candidate == null) {
                    diffs.add(new Diff(
                        original.callId,
                        original.methodId,
                        DiffType.MISSING_IN_CANDIDATE,
                        "Invocation not found in candidate",
                        original.resultJson,
                        null
                    ));
                    continue;
                }

                // Compare results
                if (!compareJsonValues(original.resultJson, candidate.resultJson)) {
                    diffs.add(new Diff(
                        original.callId,
                        original.methodId,
                        DiffType.RETURN_VALUE,
                        "Return values differ",
                        original.resultJson,
                        candidate.resultJson
                    ));
                }

                // Compare errors
                boolean originalHasError = original.errorJson != null && !original.errorJson.isEmpty();
                boolean candidateHasError = candidate.errorJson != null && !candidate.errorJson.isEmpty();

                if (originalHasError != candidateHasError) {
                    diffs.add(new Diff(
                        original.callId,
                        original.methodId,
                        DiffType.EXCEPTION,
                        originalHasError ? "Original threw exception, candidate did not" :
                                          "Candidate threw exception, original did not",
                        original.errorJson,
                        candidate.errorJson
                    ));
                } else if (originalHasError && !compareExceptions(original.errorJson, candidate.errorJson)) {
                    diffs.add(new Diff(
                        original.callId,
                        original.methodId,
                        DiffType.EXCEPTION,
                        "Exception details differ",
                        original.errorJson,
                        candidate.errorJson
                    ));
                }

                // Remove from map to track extra invocations
                candidateMap.remove(original.callId);
            }

            // Check for extra invocations in candidate
            for (Invocation extra : candidateMap.values()) {
                diffs.add(new Diff(
                    extra.callId,
                    extra.methodId,
                    DiffType.EXTRA_IN_CANDIDATE,
                    "Extra invocation in candidate",
                    null,
                    extra.resultJson
                ));
            }
        }

        return new ComparisonResult(diffs.isEmpty(), diffs);
    }

    private static List<Invocation> getInvocations(Connection conn) throws SQLException {
        List<Invocation> invocations = new ArrayList<>();
        String sql = "SELECT test_class_name, function_getting_tested, loop_index, iteration_id, return_value " +
                     "FROM test_results ORDER BY loop_index, iteration_id";

        try (PreparedStatement stmt = conn.prepareStatement(sql);
             ResultSet rs = stmt.executeQuery()) {

            while (rs.next()) {
                String testClassName = rs.getString("test_class_name");
                String functionName = rs.getString("function_getting_tested");
                int loopIndex = rs.getInt("loop_index");
                String iterationId = rs.getString("iteration_id");
                String returnValue = rs.getString("return_value");

                // Create unique call_id from loop_index and iteration_id
                // Parse iteration_id which is in format "iter_testIteration" (e.g., "1_0")
                long callId = (loopIndex * 10000L) + parseIterationId(iterationId);

                // Construct method_id as "ClassName.methodName"
                String methodId = testClassName + "." + functionName;

                invocations.add(new Invocation(
                    callId,
                    methodId,
                    null,  // args_json not captured in test_results schema
                    returnValue,  // return_value maps to resultJson
                    null   // error_json not captured in test_results schema
                ));
            }
        }

        return invocations;
    }

    /**
     * Parse iteration_id string to extract the numeric iteration number.
     * Format: "iter_testIteration" (e.g., "1_0" → 1)
     */
    private static long parseIterationId(String iterationId) {
        if (iterationId == null || iterationId.isEmpty()) {
            return 0;
        }
        try {
            // Split by underscore and take the first part
            String[] parts = iterationId.split("_");
            return Long.parseLong(parts[0]);
        } catch (Exception e) {
            // If parsing fails, try to parse the whole string
            try {
                return Long.parseLong(iterationId);
            } catch (Exception ex) {
                return 0;
            }
        }
    }

    /**
     * Compare two JSON values for equivalence.
     */
    private static boolean compareJsonValues(String json1, String json2) {
        if (json1 == null && json2 == null) return true;
        if (json1 == null || json2 == null) return false;
        if (json1.equals(json2)) return true;

        try {
            JsonElement elem1 = JsonParser.parseString(json1);
            JsonElement elem2 = JsonParser.parseString(json2);
            return compareJsonElements(elem1, elem2);
        } catch (Exception e) {
            // If parsing fails, fall back to string comparison
            return json1.equals(json2);
        }
    }

    private static boolean compareJsonElements(JsonElement elem1, JsonElement elem2) {
        if (elem1 == null && elem2 == null) return true;
        if (elem1 == null || elem2 == null) return false;
        if (elem1.isJsonNull() && elem2.isJsonNull()) return true;

        // Compare primitives
        if (elem1.isJsonPrimitive() && elem2.isJsonPrimitive()) {
            return comparePrimitives(elem1.getAsJsonPrimitive(), elem2.getAsJsonPrimitive());
        }

        // Compare arrays
        if (elem1.isJsonArray() && elem2.isJsonArray()) {
            return compareArrays(elem1.getAsJsonArray(), elem2.getAsJsonArray());
        }

        // Compare objects
        if (elem1.isJsonObject() && elem2.isJsonObject()) {
            return compareObjects(elem1.getAsJsonObject(), elem2.getAsJsonObject());
        }

        return false;
    }

    private static boolean comparePrimitives(com.google.gson.JsonPrimitive p1, com.google.gson.JsonPrimitive p2) {
        // Handle numeric comparison with epsilon
        if (p1.isNumber() && p2.isNumber()) {
            double d1 = p1.getAsDouble();
            double d2 = p2.getAsDouble();
            // Handle NaN
            if (Double.isNaN(d1) && Double.isNaN(d2)) return true;
            // Handle infinity
            if (Double.isInfinite(d1) && Double.isInfinite(d2)) {
                return (d1 > 0) == (d2 > 0);
            }
            // Compare with epsilon
            return Math.abs(d1 - d2) < EPSILON;
        }

        return Objects.equals(p1, p2);
    }

    private static boolean compareArrays(JsonArray arr1, JsonArray arr2) {
        if (arr1.size() != arr2.size()) return false;

        for (int i = 0; i < arr1.size(); i++) {
            if (!compareJsonElements(arr1.get(i), arr2.get(i))) {
                return false;
            }
        }
        return true;
    }

    private static boolean compareObjects(JsonObject obj1, JsonObject obj2) {
        // Skip type metadata for comparison
        java.util.Set<String> keys1 = new java.util.HashSet<>(obj1.keySet());
        java.util.Set<String> keys2 = new java.util.HashSet<>(obj2.keySet());
        keys1.remove("__type__");
        keys2.remove("__type__");

        if (!keys1.equals(keys2)) return false;

        for (String key : keys1) {
            if (!compareJsonElements(obj1.get(key), obj2.get(key))) {
                return false;
            }
        }
        return true;
    }

    private static boolean compareExceptions(String error1, String error2) {
        try {
            JsonObject e1 = JsonParser.parseString(error1).getAsJsonObject();
            JsonObject e2 = JsonParser.parseString(error2).getAsJsonObject();

            // Compare exception type and message
            String type1 = e1.has("type") ? e1.get("type").getAsString() : "";
            String type2 = e2.has("type") ? e2.get("type").getAsString() : "";

            // Types must match
            return type1.equals(type2);
        } catch (Exception e) {
            return error1.equals(error2);
        }
    }

    // Data classes

    private static class Invocation {
        final long callId;
        final String methodId;
        final String argsJson;
        final String resultJson;
        final String errorJson;

        Invocation(long callId, String methodId, String argsJson, String resultJson, String errorJson) {
            this.callId = callId;
            this.methodId = methodId;
            this.argsJson = argsJson;
            this.resultJson = resultJson;
            this.errorJson = errorJson;
        }
    }

    public enum DiffType {
        RETURN_VALUE,
        EXCEPTION,
        MISSING_IN_CANDIDATE,
        EXTRA_IN_CANDIDATE
    }

    public static class Diff {
        private final long callId;
        private final String methodId;
        private final DiffType type;
        private final String message;
        private final String originalValue;
        private final String candidateValue;

        public Diff(long callId, String methodId, DiffType type, String message,
                   String originalValue, String candidateValue) {
            this.callId = callId;
            this.methodId = methodId;
            this.type = type;
            this.message = message;
            this.originalValue = originalValue;
            this.candidateValue = candidateValue;
        }

        // Getters
        public long getCallId() { return callId; }
        public String getMethodId() { return methodId; }
        public DiffType getType() { return type; }
        public String getMessage() { return message; }
        public String getOriginalValue() { return originalValue; }
        public String getCandidateValue() { return candidateValue; }
    }

    public static class ComparisonResult {
        private final boolean equivalent;
        private final List<Diff> diffs;

        public ComparisonResult(boolean equivalent, List<Diff> diffs) {
            this.equivalent = equivalent;
            this.diffs = diffs;
        }

        public boolean isEquivalent() { return equivalent; }
        public List<Diff> getDiffs() { return diffs; }
    }
}
