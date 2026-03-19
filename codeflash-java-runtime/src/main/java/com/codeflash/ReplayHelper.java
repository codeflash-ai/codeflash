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

    private final Connection db;

    public ReplayHelper(String traceDbPath) {
        try {
            this.db = DriverManager.getConnection("jdbc:sqlite:" + traceDbPath);
        } catch (SQLException e) {
            throw new RuntimeException("Failed to open trace database: " + traceDbPath, e);
        }
    }

    public void replay(String className, String methodName, String descriptor, int invocationIndex) throws Exception {
        // Query the function_calls table for this method at the given index
        byte[] argsBlob;
        try (PreparedStatement stmt = db.prepareStatement(
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

        // Deserialize args
        Object deserialized = Serializer.deserialize(argsBlob);
        if (!(deserialized instanceof Object[])) {
            throw new RuntimeException("Deserialized args is not Object[], got: "
                    + (deserialized == null ? "null" : deserialized.getClass().getName()));
        }
        Object[] allArgs = (Object[]) deserialized;

        // Load the target class
        Class<?> targetClass = Class.forName(className);

        // Parse descriptor to find parameter types
        Type[] paramTypes = Type.getArgumentTypes(descriptor);
        Class<?>[] paramClasses = new Class<?>[paramTypes.length];
        for (int i = 0; i < paramTypes.length; i++) {
            paramClasses[i] = typeToClass(paramTypes[i]);
        }

        // Find the method
        Method method = targetClass.getDeclaredMethod(methodName, paramClasses);
        method.setAccessible(true);

        boolean isStatic = Modifier.isStatic(method.getModifiers());

        if (isStatic) {
            method.invoke(null, allArgs);
        } else {
            // Args contain only explicit parameters (no 'this').
            // Create a default instance via no-arg constructor or Kryo.
            Object instance;
            try {
                java.lang.reflect.Constructor<?> ctor = targetClass.getDeclaredConstructor();
                ctor.setAccessible(true);
                instance = ctor.newInstance();
            } catch (NoSuchMethodException e) {
                // Fall back to Objenesis instantiation (no constructor needed)
                instance = new org.objenesis.ObjenesisStd().newInstance(targetClass);
            }
            method.invoke(instance, allArgs);
        }
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

    public void close() {
        try {
            if (db != null) db.close();
        } catch (SQLException e) {
            System.err.println("Error closing ReplayHelper: " + e.getMessage());
        }
    }
}
