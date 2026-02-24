package com.codeflash.agent;

import java.util.Objects;

/**
 * Immutable key identifying a method, matching the Python tracer's 4-tuple:
 * (filename, firstlineno, funcname, classname).
 */
public final class MethodKey {

    private final String fileName;
    private final int lineNumber;
    private final String methodName;
    private final String className;
    private final int hashCode;

    public MethodKey(String fileName, int lineNumber, String methodName, String className) {
        this.fileName = fileName;
        this.lineNumber = lineNumber;
        this.methodName = methodName;
        this.className = className;
        this.hashCode = Objects.hash(fileName, lineNumber, methodName, className);
    }

    public String getFileName() {
        return fileName;
    }

    public int getLineNumber() {
        return lineNumber;
    }

    public String getMethodName() {
        return methodName;
    }

    public String getClassName() {
        return className;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof MethodKey)) return false;
        MethodKey that = (MethodKey) o;
        return lineNumber == that.lineNumber
            && Objects.equals(fileName, that.fileName)
            && Objects.equals(methodName, that.methodName)
            && Objects.equals(className, that.className);
    }

    @Override
    public int hashCode() {
        return hashCode;
    }

    @Override
    public String toString() {
        return className + "." + methodName + "(" + fileName + ":" + lineNumber + ")";
    }
}
