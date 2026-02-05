package com.codeflash;

import java.io.Serializable;
import java.util.Objects;

/**
 * Placeholder for objects that could not be serialized.
 *
 * When KryoSerializer encounters an object that cannot be serialized
 * (e.g., Socket, Connection, Stream), it replaces it with a KryoPlaceholder
 * that stores metadata about the original object.
 *
 * This allows the rest of the object graph to be serialized while preserving
 * information about what was lost. If code attempts to use the placeholder
 * during replay tests, an error can be detected.
 */
public final class KryoPlaceholder implements Serializable {

    private static final long serialVersionUID = 1L;
    private static final int MAX_STR_LENGTH = 100;

    private final String objType;
    private final String objStr;
    private final String errorMsg;
    private final String path;

    /**
     * Create a placeholder for an unserializable object.
     *
     * @param objType   The fully qualified class name of the original object
     * @param objStr    String representation of the object (may be truncated)
     * @param errorMsg  The error message explaining why serialization failed
     * @param path      The path in the object graph (e.g., "data.nested[0].socket")
     */
    public KryoPlaceholder(String objType, String objStr, String errorMsg, String path) {
        this.objType = objType;
        this.objStr = truncate(objStr, MAX_STR_LENGTH);
        this.errorMsg = errorMsg;
        this.path = path;
    }

    /**
     * Create a placeholder from an object and error.
     */
    public static KryoPlaceholder create(Object obj, String errorMsg, String path) {
        String objType = obj != null ? obj.getClass().getName() : "null";
        String objStr = safeToString(obj);
        return new KryoPlaceholder(objType, objStr, errorMsg, path);
    }

    private static String safeToString(Object obj) {
        if (obj == null) {
            return "null";
        }
        try {
            return obj.toString();
        } catch (Exception e) {
            return "<toString failed: " + e.getMessage() + ">";
        }
    }

    private static String truncate(String s, int maxLength) {
        if (s == null) {
            return null;
        }
        if (s.length() <= maxLength) {
            return s;
        }
        return s.substring(0, maxLength) + "...";
    }

    /**
     * Get the original type name of the unserializable object.
     */
    public String getObjType() {
        return objType;
    }

    /**
     * Get the string representation of the original object (may be truncated).
     */
    public String getObjStr() {
        return objStr;
    }

    /**
     * Get the error message explaining why serialization failed.
     */
    public String getErrorMsg() {
        return errorMsg;
    }

    /**
     * Get the path in the object graph where this placeholder was created.
     */
    public String getPath() {
        return path;
    }

    @Override
    public String toString() {
        return String.format("<KryoPlaceholder[%s] at '%s': %s>", objType, path, objStr);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        KryoPlaceholder that = (KryoPlaceholder) o;
        return Objects.equals(objType, that.objType) &&
               Objects.equals(path, that.path);
    }

    @Override
    public int hashCode() {
        return Objects.hash(objType, path);
    }
}
