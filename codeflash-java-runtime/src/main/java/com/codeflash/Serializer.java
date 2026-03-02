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
import java.util.AbstractMap;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Binary serializer using Kryo with graceful handling of unserializable objects.
 *
 * This class provides:
 * 1. Attempts direct Kryo serialization first
 * 2. On failure, recursively processes containers (Map, Collection, Array)
 * 3. Replaces truly unserializable objects with Placeholder
 *
 * Thread-safe via ThreadLocal Kryo instances.
 */
public final class Serializer {

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
        kryo.register(java.util.UUID.class);
        kryo.register(java.math.BigDecimal.class);
        kryo.register(java.math.BigInteger.class);

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

    private Serializer() {
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

        // Primitives and common immutable types - return directly (Kryo handles these well)
        if (isPrimitiveOrWrapper(clazz) || obj instanceof String || obj instanceof Enum) {
            return obj;
        }

        // Check for circular reference
        if (seen.containsKey(obj)) {
            return KryoPlaceholder.create(obj, "Circular reference detected", path);
        }
        seen.put(obj, Boolean.TRUE);

        try {
            // Handle containers: for simple containers (only primitives, wrappers, strings, enums),
            // try direct serialization to preserve full size. For containers with complex/potentially
            // unserializable types, recursively process to catch and replace unserializable objects.
            if (obj instanceof Map) {
                Map<?, ?> map = (Map<?, ?>) obj;
                if (containsOnlySimpleTypes(map)) {
                    // Simple map - try direct serialization to preserve full size
                    byte[] serialized = tryDirectSerialize(obj);
                    if (serialized != null) {
                        try {
                            deserialize(serialized);
                            return obj; // Success - return original
                        } catch (Exception e) {
                            // Fall through to recursive handling
                        }
                    }
                }
                return handleMap(map, seen, depth, path);
            }
            if (obj instanceof Collection) {
                Collection<?> collection = (Collection<?>) obj;
                if (containsOnlySimpleTypes(collection)) {
                    // Simple collection - try direct serialization to preserve full size
                    byte[] serialized = tryDirectSerialize(obj);
                    if (serialized != null) {
                        try {
                            deserialize(serialized);
                            return obj; // Success - return original
                        } catch (Exception e) {
                            // Fall through to recursive handling
                        }
                    }
                }
                return handleCollection(collection, seen, depth, path);
            }
            if (clazz.isArray()) {
                return handleArray(obj, seen, depth, path);
            }

            // For non-container objects, try direct serialization first
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
     * Check if an object is a "simple" type that Kryo can serialize directly without issues.
     * Simple types include primitives, wrappers, strings, enums, and common date/time types.
     */
    private static boolean isSimpleType(Object obj) {
        if (obj == null) {
            return true;
        }
        Class<?> clazz = obj.getClass();
        return isPrimitiveOrWrapper(clazz) ||
               obj instanceof String ||
               obj instanceof Enum ||
               obj instanceof java.util.UUID ||
               obj instanceof java.math.BigDecimal ||
               obj instanceof java.math.BigInteger ||
               obj instanceof java.util.Date ||
               obj instanceof java.time.temporal.Temporal;
    }

    /**
     * Check if a collection contains only simple types that don't need recursive processing
     * to check for unserializable nested objects.
     */
    private static boolean containsOnlySimpleTypes(Collection<?> collection) {
        for (Object item : collection) {
            if (!isSimpleType(item)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Check if a map contains only simple types (both keys and values).
     */
    private static boolean containsOnlySimpleTypes(Map<?, ?> map) {
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (!isSimpleType(entry.getKey()) || !isSimpleType(entry.getValue())) {
                return false;
            }
        }
        return true;
    }

    /**
     * Handle Map serialization with recursive processing of values.
     * Preserves map type (TreeMap, LinkedHashMap, etc.) where possible.
     */
    private static Object handleMap(Map<?, ?> map, IdentityHashMap<Object, Object> seen,
                                    int depth, String path) {
        List<Map.Entry<Object, Object>> processed = new ArrayList<>();
        int count = 0;

        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (count >= MAX_COLLECTION_SIZE) {
                processed.add(new AbstractMap.SimpleEntry<>("__truncated__",
                    map.size() - count + " more entries"));
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

            processed.add(new AbstractMap.SimpleEntry<>(processedKey, processedValue));
            count++;
        }

        return createMapOfSameType(map, processed);
    }

    /**
     * Create a map of the same type as the original, populated with processed entries.
     */
    @SuppressWarnings("unchecked")
    private static Map<Object, Object> createMapOfSameType(Map<?, ?> original,
                                                            List<Map.Entry<Object, Object>> entries) {
        try {
            // Handle specific map types
            if (original instanceof TreeMap) {
                // TreeMap - try to preserve with serializable comparator
                try {
                    TreeMap<Object, Object> result = new TreeMap<>(new SerializableComparator());
                    for (Map.Entry<Object, Object> entry : entries) {
                        result.put(entry.getKey(), entry.getValue());
                    }
                    return result;
                } catch (Exception e) {
                    // Fall back to LinkedHashMap if keys aren't comparable
                    LinkedHashMap<Object, Object> result = new LinkedHashMap<>();
                    for (Map.Entry<Object, Object> entry : entries) {
                        result.put(entry.getKey(), entry.getValue());
                    }
                    return result;
                }
            }

            if (original instanceof LinkedHashMap) {
                LinkedHashMap<Object, Object> result = new LinkedHashMap<>();
                for (Map.Entry<Object, Object> entry : entries) {
                    result.put(entry.getKey(), entry.getValue());
                }
                return result;
            }

            if (original instanceof HashMap) {
                HashMap<Object, Object> result = new HashMap<>();
                for (Map.Entry<Object, Object> entry : entries) {
                    result.put(entry.getKey(), entry.getValue());
                }
                return result;
            }

            // Try to instantiate the same type
            try {
                Map<Object, Object> result = (Map<Object, Object>) original.getClass().getDeclaredConstructor().newInstance();
                for (Map.Entry<Object, Object> entry : entries) {
                    result.put(entry.getKey(), entry.getValue());
                }
                return result;
            } catch (Exception e) {
                // Fallback
            }

            // Default fallback - LinkedHashMap preserves insertion order
            LinkedHashMap<Object, Object> result = new LinkedHashMap<>();
            for (Map.Entry<Object, Object> entry : entries) {
                result.put(entry.getKey(), entry.getValue());
            }
            return result;

        } catch (Exception e) {
            // Final fallback
            LinkedHashMap<Object, Object> result = new LinkedHashMap<>();
            for (Map.Entry<Object, Object> entry : entries) {
                result.put(entry.getKey(), entry.getValue());
            }
            return result;
        }
    }

    /**
     * Serializable comparator for TreeSet/TreeMap that handles mixed types.
     */
    private static class SerializableComparator implements java.util.Comparator<Object>, java.io.Serializable {
        private static final long serialVersionUID = 1L;

        @Override
        @SuppressWarnings("unchecked")
        public int compare(Object a, Object b) {
            if (a == null && b == null) return 0;
            if (a == null) return -1;
            if (b == null) return 1;
            if (a instanceof Comparable && b instanceof Comparable && a.getClass().equals(b.getClass())) {
                return ((Comparable<Object>) a).compareTo(b);
            }
            return a.toString().compareTo(b.toString());
        }
    }

    /**
     * Handle Collection serialization with recursive processing of elements.
     * Preserves collection type (LinkedList, TreeSet, etc.) where possible.
     */
    private static Object handleCollection(Collection<?> collection, IdentityHashMap<Object, Object> seen,
                                           int depth, String path) {
        List<Object> processed = new ArrayList<>();
        int count = 0;

        for (Object item : collection) {
            if (count >= MAX_COLLECTION_SIZE) {
                processed.add(KryoPlaceholder.create(null,
                    collection.size() - count + " more elements truncated", path + "[truncated]"));
                break;
            }

            String itemPath = path.isEmpty() ? "[" + count + "]" : path + "[" + count + "]";

            try {
                processed.add(recursiveProcess(item, seen, depth + 1, itemPath));
            } catch (Exception e) {
                processed.add(KryoPlaceholder.create(item, e.getMessage(), itemPath));
            }
            count++;
        }

        // Try to preserve original collection type
        return createCollectionOfSameType(collection, processed);
    }

    /**
     * Create a collection of the same type as the original, populated with processed elements.
     */
    @SuppressWarnings("unchecked")
    private static Collection<Object> createCollectionOfSameType(Collection<?> original, List<Object> elements) {
        try {
            // Handle specific collection types
            if (original instanceof TreeSet) {
                // TreeSet - try to preserve with natural ordering using serializable comparator
                try {
                    TreeSet<Object> result = new TreeSet<>(new SerializableComparator());
                    result.addAll(elements);
                    return result;
                } catch (Exception e) {
                    // Fall back to LinkedHashSet if elements aren't comparable
                    return new LinkedHashSet<>(elements);
                }
            }

            if (original instanceof LinkedHashSet) {
                return new LinkedHashSet<>(elements);
            }

            if (original instanceof HashSet) {
                return new HashSet<>(elements);
            }

            if (original instanceof Set) {
                return new LinkedHashSet<>(elements);
            }

            // List types
            if (original instanceof LinkedList) {
                return new LinkedList<>(elements);
            }

            if (original instanceof ArrayList) {
                return new ArrayList<>(elements);
            }

            // Try to instantiate the same type
            try {
                Collection<Object> result = (Collection<Object>) original.getClass().getDeclaredConstructor().newInstance();
                result.addAll(elements);
                return result;
            } catch (Exception e) {
                // Fallback
            }

            // Default fallbacks
            if (original instanceof Set) {
                return new LinkedHashSet<>(elements);
            }
            return new ArrayList<>(elements);

        } catch (Exception e) {
            // Final fallback
            if (original instanceof Set) {
                return new LinkedHashSet<>(elements);
            }
            return new ArrayList<>(elements);
        }
    }

    /**
     * Handle Array serialization with recursive processing of elements.
     * Preserves array type instead of converting to List.
     */
    private static Object handleArray(Object array, IdentityHashMap<Object, Object> seen,
                                      int depth, String path) {
        int length = java.lang.reflect.Array.getLength(array);
        int limit = Math.min(length, MAX_COLLECTION_SIZE);
        Class<?> componentType = array.getClass().getComponentType();

        // Process elements into a temporary list first
        List<Object> processed = new ArrayList<>();
        boolean hasPlaceholder = false;

        for (int i = 0; i < limit; i++) {
            String itemPath = path.isEmpty() ? "[" + i + "]" : path + "[" + i + "]";
            Object element = java.lang.reflect.Array.get(array, i);

            try {
                Object processedElement = recursiveProcess(element, seen, depth + 1, itemPath);
                processed.add(processedElement);
                if (processedElement instanceof KryoPlaceholder) {
                    hasPlaceholder = true;
                }
            } catch (Exception e) {
                processed.add(KryoPlaceholder.create(element, e.getMessage(), itemPath));
                hasPlaceholder = true;
            }
        }

        // If truncated or has placeholders with primitive array, return as Object[]
        if (length > limit || (hasPlaceholder && componentType.isPrimitive())) {
            Object[] result = new Object[processed.size() + (length > limit ? 1 : 0)];
            for (int i = 0; i < processed.size(); i++) {
                result[i] = processed.get(i);
            }
            if (length > limit) {
                result[processed.size()] = KryoPlaceholder.create(null,
                    length - limit + " more elements truncated", path + "[truncated]");
            }
            return result;
        }

        // Try to preserve the original array type
        try {
            // For object arrays, use Object[] if there are placeholders (type mismatch)
            Class<?> resultComponentType = hasPlaceholder ? Object.class : componentType;
            Object result = java.lang.reflect.Array.newInstance(resultComponentType, processed.size());

            for (int i = 0; i < processed.size(); i++) {
                java.lang.reflect.Array.set(result, i, processed.get(i));
            }
            return result;
        } catch (Exception e) {
            // Fallback to Object array if we can't create the specific type
            return processed.toArray();
        }
    }

    /**
     * Handle custom object serialization with recursive processing of fields.
     * Falls back to Map representation if field types can't accept placeholders.
     */
    private static Object handleObject(Object obj, IdentityHashMap<Object, Object> seen,
                                       int depth, String path) {
        Class<?> clazz = obj.getClass();

        // Try to create a copy with processed fields
        try {
            Object newObj = createInstance(clazz);
            if (newObj == null) {
                return objectToMap(obj, seen, depth, path);
            }

            boolean hasTypeMismatch = false;

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

                        // Check if we can assign the processed value to this field
                        if (processedValue != null) {
                            Class<?> fieldType = field.getType();
                            Class<?> valueType = processedValue.getClass();

                            // If processed value is a placeholder but field type can't hold it
                            if (processedValue instanceof KryoPlaceholder && !fieldType.isAssignableFrom(KryoPlaceholder.class)) {
                                // Type mismatch - can't assign placeholder to typed field
                                hasTypeMismatch = true;
                            } else if (!isAssignable(fieldType, valueType)) {
                                // Other type mismatch (e.g., array became list)
                                hasTypeMismatch = true;
                            } else {
                                field.set(newObj, processedValue);
                            }
                        } else {
                            field.set(newObj, null);
                        }
                    } catch (Exception e) {
                        // Field couldn't be processed - mark as type mismatch
                        hasTypeMismatch = true;
                    }
                }
                currentClass = currentClass.getSuperclass();
            }

            // If there's a type mismatch, use Map representation to preserve placeholders
            if (hasTypeMismatch) {
                return objectToMap(obj, seen, depth, path);
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
     * Check if a value type can be assigned to a field type.
     */
    private static boolean isAssignable(Class<?> fieldType, Class<?> valueType) {
        if (fieldType.isAssignableFrom(valueType)) {
            return true;
        }
        // Handle primitive/wrapper conversion
        if (fieldType.isPrimitive()) {
            if (fieldType == int.class && valueType == Integer.class) return true;
            if (fieldType == long.class && valueType == Long.class) return true;
            if (fieldType == double.class && valueType == Double.class) return true;
            if (fieldType == float.class && valueType == Float.class) return true;
            if (fieldType == boolean.class && valueType == Boolean.class) return true;
            if (fieldType == byte.class && valueType == Byte.class) return true;
            if (fieldType == char.class && valueType == Character.class) return true;
            if (fieldType == short.class && valueType == Short.class) return true;
        }
        return false;
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
