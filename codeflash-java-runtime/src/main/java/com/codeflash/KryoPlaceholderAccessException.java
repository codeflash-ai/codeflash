package com.codeflash;

/**
 * Exception thrown when attempting to access or use a KryoPlaceholder.
 *
 * This exception indicates that code attempted to interact with an object
 * that could not be serialized and was replaced with a placeholder. This
 * typically means the test behavior cannot be verified for this code path.
 */
public class KryoPlaceholderAccessException extends RuntimeException {

    private final String objType;
    private final String path;

    public KryoPlaceholderAccessException(String message, String objType, String path) {
        super(message);
        this.objType = objType;
        this.path = path;
    }

    /**
     * Get the original type name of the unserializable object.
     */
    public String getObjType() {
        return objType;
    }

    /**
     * Get the path in the object graph where the placeholder was created.
     */
    public String getPath() {
        return path;
    }

    @Override
    public String toString() {
        return String.format("KryoPlaceholderAccessException[type=%s, path=%s]: %s",
            objType, path, getMessage());
    }
}
