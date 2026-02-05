package com.codeflash;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.esotericsoftware.kryo.util.DefaultInstantiatorStrategy;
import org.objenesis.strategy.StdInstantiatorStrategy;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.net.ServerSocket;
import java.net.Socket;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Binary serializer using Kryo with graceful handling of unserializable objects.
 *
 * This class provides Python-like dill behavior:
 * 1. Attempts direct Kryo serialization first
 * 2. On failure, recursively processes containers (Map, Collection, Array)
 * 3. Replaces truly unserializable objects with KryoPlaceholder
 *
 * Thread-safe via ThreadLocal Kryo instances.
 */
public final class KryoSerializer {

    private static final int MAX_DEPTH = 10;
    private static final int MAX_COLLECTION_SIZE = 1000;
    private static final int BUFFER_SIZE = 4096;

    // Thread-local Kryo instances (Kryo is not thread-safe)
    private static final ThreadLocal<Kryo> KRYO = ThreadLocal.withInitial(() -> {
        Kryo kryo = new Kryo();
        kryo.setRegistrationRequired(false);
        kryo.setReferences(true);
        kryo.setInstantiatorStrategy(new DefaultInstantiatorStrategy(
            new StdInstantiatorStrategy()));

        // Register common types for efficiency
        kryo.register(ArrayList.class);
        kryo.register(LinkedList.class);
        kryo.register(HashMap.class);
        kryo.register(LinkedHashMap.class);
        kryo.register(HashSet.class);
        kryo.register(LinkedHashSet.class);
        kryo.register(TreeMap.class);
        kryo.register(TreeSet.class);
        kryo.register(KryoPlaceholder.class);

        return kryo;
    });

    // Cache of known unserializable types
    private static final Set<Class<?>> UNSERIALIZABLE_TYPES = ConcurrentHashMap.newKeySet();

    static {
        // Pre-populate with known unserializable types
        UNSERIALIZABLE_TYPES.add(Socket.class);
        UNSERIALIZABLE_TYPES.add(ServerSocket.class);
        UNSERIALIZABLE_TYPES.add(InputStream.class);
        UNSERIALIZABLE_TYPES.add(OutputStream.class);
        UNSERIALIZABLE_TYPES.add(Connection.class);
        UNSERIALIZABLE_TYPES.add(Statement.class);
        UNSERIALIZABLE_TYPES.add(ResultSet.class);
        UNSERIALIZABLE_TYPES.add(Thread.class);
        UNSERIALIZABLE_TYPES.add(ThreadGroup.class);
        UNSERIALIZABLE_TYPES.add(ClassLoader.class);
    }

    private KryoSerializer() {
        // Utility class
    }

    /**
     * Serialize an object to bytes with graceful handling of unserializable parts.
     *
     * @param obj The object to serialize
     * @return Serialized bytes (may contain KryoPlaceholder for unserializable parts)
     */
    public static byte[] serialize(Object obj) {
        Object processed = recursiveProcess(obj, new IdentityHashMap<>(), 0, "");
        return directSerialize(processed);
    }

    /**
     * Deserialize bytes back to an object.
     * The returned object may contain KryoPlaceholder instances for parts
     * that could not be serialized originally.
     *
     * @param data Serialized bytes
     * @return Deserialized object
     */
    public static Object deserialize(byte[] data) {
        if (data == null || data.length == 0) {
            return null;
        }
        Kryo kryo = KRYO.get();
        try (Input input = new Input(data)) {
            return kryo.readClassAndObject(input);
        }
    }

    /**
     * Serialize an exception with its metadata.
     *
     * @param error The exception to serialize
     * @return Serialized bytes containing exception information
     */
    public static byte[] serializeException(Throwable error) {
        Map<String, Object> exceptionData = new LinkedHashMap<>();
        exceptionData.put("__exception__", true);
        exceptionData.put("type", error.getClass().getName());
        exceptionData.put("message", error.getMessage());

        // Capture stack trace as strings
        List<String> stackTrace = new ArrayList<>();
        for (StackTraceElement element : error.getStackTrace()) {
            stackTrace.add(element.toString());
        }
        exceptionData.put("stackTrace", stackTrace);

        // Capture cause if present
        if (error.getCause() != null) {
            exceptionData.put("causeType", error.getCause().getClass().getName());
            exceptionData.put("causeMessage", error.getCause().getMessage());
        }

        return serialize(exceptionData);
    }

    /**
     * Direct serialization without recursive processing.
     */
    private static byte[] directSerialize(Object obj) {
        Kryo kryo = KRYO.get();
        ByteArrayOutputStream baos = new ByteArrayOutputStream(BUFFER_SIZE);
        try (Output output = new Output(baos)) {
            kryo.writeClassAndObject(output, obj);
        }
        return baos.toByteArray();
    }

    /**
     * Try to serialize directly; returns null on failure.
     */
    private static byte[] tryDirectSerialize(Object obj) {
        try {
            return directSerialize(obj);
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * Recursively process an object, replacing unserializable parts with placeholders.
     */
    private static Object recursiveProcess(Object obj, IdentityHashMap<Object, Object> seen,
                                           int depth, String path) {
        // Handle null
        if (obj == null) {
            return null;
        }

        Class<?> clazz = obj.getClass();

        // Check if known unserializable type
        if (isKnownUnserializable(clazz)) {
            return KryoPlaceholder.create(obj, "Known unserializable type: " + clazz.getName(), path);
        }

        // Check max depth
        if (depth > MAX_DEPTH) {
            return KryoPlaceholder.create(obj, "Max recursion depth exceeded", path);
        }

        // Primitives and common immutable types - try direct serialization
        if (isPrimitiveOrWrapper(clazz) || obj instanceof String || obj instanceof Enum) {
            return obj;
        }

        // Try direct serialization first
        byte[] serialized = tryDirectSerialize(obj);
        if (serialized != null) {
            // Verify it can be deserialized
            try {
                deserialize(serialized);
                return obj; // Success - return original
            } catch (Exception e) {
                // Fall through to recursive handling
            }
        }

        // Check for circular reference
        if (seen.containsKey(obj)) {
            return KryoPlaceholder.create(obj, "Circular reference detected", path);
        }
        seen.put(obj, Boolean.TRUE);

        try {
            // Handle containers recursively
            if (obj instanceof Map) {
                return handleMap((Map<?, ?>) obj, seen, depth, path);
            }
            if (obj instanceof Collection) {
                return handleCollection((Collection<?>) obj, seen, depth, path);
            }
            if (clazz.isArray()) {
                return handleArray(obj, seen, depth, path);
            }

            // Handle objects with fields
            return handleObject(obj, seen, depth, path);

        } finally {
            seen.remove(obj);
        }
    }

    /**
     * Check if a class is known to be unserializable.
     */
    private static boolean isKnownUnserializable(Class<?> clazz) {
        if (UNSERIALIZABLE_TYPES.contains(clazz)) {
            return true;
        }
        // Check superclasses and interfaces
        for (Class<?> unserializable : UNSERIALIZABLE_TYPES) {
            if (unserializable.isAssignableFrom(clazz)) {
                UNSERIALIZABLE_TYPES.add(clazz); // Cache for future
                return true;
            }
        }
        return false;
    }

    /**
     * Check if a class is a primitive or wrapper type.
     */
    private static boolean isPrimitiveOrWrapper(Class<?> clazz) {
        return clazz.isPrimitive() ||
               clazz == Boolean.class ||
               clazz == Byte.class ||
               clazz == Character.class ||
               clazz == Short.class ||
               clazz == Integer.class ||
               clazz == Long.class ||
               clazz == Float.class ||
               clazz == Double.class;
    }

    /**
     * Handle Map serialization with recursive processing of values.
     */
    private static Object handleMap(Map<?, ?> map, IdentityHashMap<Object, Object> seen,
                                    int depth, String path) {
        Map<Object, Object> result = new LinkedHashMap<>();
        int count = 0;

        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (count >= MAX_COLLECTION_SIZE) {
                result.put("__truncated__", map.size() - count + " more entries");
                break;
            }

            Object key = entry.getKey();
            Object value = entry.getValue();

            // Process key
            String keyStr = key != null ? key.toString() : "null";
            String keyPath = path.isEmpty() ? "[" + keyStr + "]" : path + "[" + keyStr + "]";

            Object processedKey;
            try {
                processedKey = recursiveProcess(key, seen, depth + 1, keyPath + ".key");
            } catch (Exception e) {
                processedKey = KryoPlaceholder.create(key, e.getMessage(), keyPath + ".key");
            }

            // Process value
            Object processedValue;
            try {
                processedValue = recursiveProcess(value, seen, depth + 1, keyPath);
            } catch (Exception e) {
                processedValue = KryoPlaceholder.create(value, e.getMessage(), keyPath);
            }

            result.put(processedKey, processedValue);
            count++;
        }

        return result;
    }

    /**
     * Handle Collection serialization with recursive processing of elements.
     */
    private static Object handleCollection(Collection<?> collection, IdentityHashMap<Object, Object> seen,
                                           int depth, String path) {
        List<Object> result = new ArrayList<>();
        int count = 0;

        for (Object item : collection) {
            if (count >= MAX_COLLECTION_SIZE) {
                result.add(KryoPlaceholder.create(null,
                    collection.size() - count + " more elements truncated", path + "[truncated]"));
                break;
            }

            String itemPath = path.isEmpty() ? "[" + count + "]" : path + "[" + count + "]";

            try {
                result.add(recursiveProcess(item, seen, depth + 1, itemPath));
            } catch (Exception e) {
                result.add(KryoPlaceholder.create(item, e.getMessage(), itemPath));
            }
            count++;
        }

        // Try to preserve original collection type
        if (collection instanceof Set) {
            return new LinkedHashSet<>(result);
        }
        return result;
    }

    /**
     * Handle Array serialization with recursive processing of elements.
     */
    private static Object handleArray(Object array, IdentityHashMap<Object, Object> seen,
                                      int depth, String path) {
        int length = java.lang.reflect.Array.getLength(array);
        int limit = Math.min(length, MAX_COLLECTION_SIZE);

        List<Object> result = new ArrayList<>();
        for (int i = 0; i < limit; i++) {
            String itemPath = path.isEmpty() ? "[" + i + "]" : path + "[" + i + "]";
            Object element = java.lang.reflect.Array.get(array, i);

            try {
                result.add(recursiveProcess(element, seen, depth + 1, itemPath));
            } catch (Exception e) {
                result.add(KryoPlaceholder.create(element, e.getMessage(), itemPath));
            }
        }

        if (length > limit) {
            result.add(KryoPlaceholder.create(null,
                length - limit + " more elements truncated", path + "[truncated]"));
        }

        return result;
    }

    /**
     * Handle custom object serialization with recursive processing of fields.
     */
    private static Object handleObject(Object obj, IdentityHashMap<Object, Object> seen,
                                       int depth, String path) {
        Class<?> clazz = obj.getClass();

        // Try to create a copy with processed fields
        try {
            Object newObj = createInstance(clazz);
            if (newObj == null) {
                return KryoPlaceholder.create(obj, "Cannot instantiate class: " + clazz.getName(), path);
            }

            // Copy and process all fields
            Class<?> currentClass = clazz;
            while (currentClass != null && currentClass != Object.class) {
                for (Field field : currentClass.getDeclaredFields()) {
                    if (Modifier.isStatic(field.getModifiers()) ||
                        Modifier.isTransient(field.getModifiers())) {
                        continue;
                    }

                    try {
                        field.setAccessible(true);
                        Object value = field.get(obj);
                        String fieldPath = path.isEmpty() ? field.getName() : path + "." + field.getName();

                        Object processedValue = recursiveProcess(value, seen, depth + 1, fieldPath);
                        field.set(newObj, processedValue);
                    } catch (Exception e) {
                        // Field couldn't be processed - leave as default
                    }
                }
                currentClass = currentClass.getSuperclass();
            }

            // Verify the new object can be serialized
            byte[] testSerialize = tryDirectSerialize(newObj);
            if (testSerialize != null) {
                return newObj;
            }

            // Still can't serialize - return as map representation
            return objectToMap(obj, seen, depth, path);

        } catch (Exception e) {
            // Fall back to map representation
            return objectToMap(obj, seen, depth, path);
        }
    }

    /**
     * Convert an object to a Map representation for serialization.
     */
    private static Map<String, Object> objectToMap(Object obj, IdentityHashMap<Object, Object> seen,
                                                   int depth, String path) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("__type__", obj.getClass().getName());

        Class<?> currentClass = obj.getClass();
        while (currentClass != null && currentClass != Object.class) {
            for (Field field : currentClass.getDeclaredFields()) {
                if (Modifier.isStatic(field.getModifiers()) ||
                    Modifier.isTransient(field.getModifiers())) {
                    continue;
                }

                try {
                    field.setAccessible(true);
                    Object value = field.get(obj);
                    String fieldPath = path.isEmpty() ? field.getName() : path + "." + field.getName();

                    Object processedValue = recursiveProcess(value, seen, depth + 1, fieldPath);
                    result.put(field.getName(), processedValue);
                } catch (Exception e) {
                    result.put(field.getName(),
                        KryoPlaceholder.create(null, "Field access error: " + e.getMessage(),
                            path + "." + field.getName()));
                }
            }
            currentClass = currentClass.getSuperclass();
        }

        return result;
    }

    /**
     * Try to create an instance of a class.
     */
    private static Object createInstance(Class<?> clazz) {
        try {
            return clazz.getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            // Try Objenesis via Kryo's instantiator
            try {
                Kryo kryo = KRYO.get();
                return kryo.newInstance(clazz);
            } catch (Exception e2) {
                return null;
            }
        }
    }

    /**
     * Add a type to the known unserializable types cache.
     */
    public static void registerUnserializableType(Class<?> clazz) {
        UNSERIALIZABLE_TYPES.add(clazz);
    }

    /**
     * Reset the unserializable types cache to default state.
     * Clears any dynamically discovered types but keeps the built-in defaults.
     */
    public static void clearUnserializableTypesCache() {
        UNSERIALIZABLE_TYPES.clear();
        // Re-add default unserializable types
        UNSERIALIZABLE_TYPES.add(Socket.class);
        UNSERIALIZABLE_TYPES.add(ServerSocket.class);
        UNSERIALIZABLE_TYPES.add(InputStream.class);
        UNSERIALIZABLE_TYPES.add(OutputStream.class);
        UNSERIALIZABLE_TYPES.add(Connection.class);
        UNSERIALIZABLE_TYPES.add(Statement.class);
        UNSERIALIZABLE_TYPES.add(ResultSet.class);
        UNSERIALIZABLE_TYPES.add(Thread.class);
        UNSERIALIZABLE_TYPES.add(ThreadGroup.class);
        UNSERIALIZABLE_TYPES.add(ClassLoader.class);
    }
}
