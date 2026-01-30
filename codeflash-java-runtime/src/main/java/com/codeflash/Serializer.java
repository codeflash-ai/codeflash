package com.codeflash;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.lang.reflect.Proxy;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.Collection;
import java.util.Date;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Serializer for Java objects to JSON format.
 *
 * Handles:
 * - Primitives and their wrappers
 * - Strings
 * - Arrays (primitive and object)
 * - Collections (List, Set, etc.)
 * - Maps
 * - Date/Time types
 * - Custom objects via reflection
 * - Circular references (detected and marked)
 */
public final class Serializer {

    private static final Gson GSON = new GsonBuilder()
            .serializeNulls()
            .create();

    private static final int MAX_DEPTH = 10;
    private static final int MAX_COLLECTION_SIZE = 1000;

    private Serializer() {
        // Utility class
    }

    /**
     * Serialize an object to JSON string.
     *
     * @param obj Object to serialize
     * @return JSON string representation
     */
    public static String toJson(Object obj) {
        try {
            JsonElement element = serialize(obj, new IdentityHashMap<>(), 0);
            return GSON.toJson(element);
        } catch (Exception e) {
            // Fallback for serialization errors
            JsonObject error = new JsonObject();
            error.addProperty("__serialization_error__", e.getMessage());
            error.addProperty("__type__", obj != null ? obj.getClass().getName() : "null");
            return GSON.toJson(error);
        }
    }

    /**
     * Serialize varargs (for capturing multiple arguments).
     *
     * @param args Arguments to serialize
     * @return JSON array string
     */
    public static String toJson(Object... args) {
        JsonArray array = new JsonArray();
        IdentityHashMap<Object, Boolean> seen = new IdentityHashMap<>();
        for (Object arg : args) {
            array.add(serialize(arg, seen, 0));
        }
        return GSON.toJson(array);
    }

    /**
     * Serialize an exception to JSON.
     *
     * @param error Exception to serialize
     * @return JSON string with exception details
     */
    public static String exceptionToJson(Throwable error) {
        JsonObject obj = new JsonObject();
        obj.addProperty("__exception__", true);
        obj.addProperty("type", error.getClass().getName());
        obj.addProperty("message", error.getMessage());

        // Capture stack trace
        JsonArray stackTrace = new JsonArray();
        for (StackTraceElement element : error.getStackTrace()) {
            stackTrace.add(element.toString());
        }
        obj.add("stackTrace", stackTrace);

        // Capture cause if present
        if (error.getCause() != null) {
            obj.addProperty("causeType", error.getCause().getClass().getName());
            obj.addProperty("causeMessage", error.getCause().getMessage());
        }

        return GSON.toJson(obj);
    }

    private static JsonElement serialize(Object obj, IdentityHashMap<Object, Boolean> seen, int depth) {
        if (obj == null) {
            return JsonNull.INSTANCE;
        }

        // Depth limit to prevent infinite recursion
        if (depth > MAX_DEPTH) {
            JsonObject truncated = new JsonObject();
            truncated.addProperty("__truncated__", "max depth exceeded");
            return truncated;
        }

        Class<?> clazz = obj.getClass();

        // Primitives and wrappers
        if (obj instanceof Boolean) {
            return new JsonPrimitive((Boolean) obj);
        }
        if (obj instanceof Number) {
            return new JsonPrimitive((Number) obj);
        }
        if (obj instanceof Character) {
            return new JsonPrimitive(String.valueOf(obj));
        }
        if (obj instanceof String) {
            return new JsonPrimitive((String) obj);
        }

        // Class objects - serialize as class name string
        if (obj instanceof Class) {
            return new JsonPrimitive(getClassName((Class<?>) obj));
        }

        // Dynamic proxies - serialize cleanly without reflection
        if (Proxy.isProxyClass(clazz)) {
            JsonObject proxyObj = new JsonObject();
            proxyObj.addProperty("__proxy__", true);
            Class<?>[] interfaces = clazz.getInterfaces();
            if (interfaces.length > 0) {
                JsonArray interfaceNames = new JsonArray();
                for (Class<?> iface : interfaces) {
                    interfaceNames.add(iface.getName());
                }
                proxyObj.add("interfaces", interfaceNames);
            }
            return proxyObj;
        }

        // Check for circular reference (only for reference types)
        if (seen.containsKey(obj)) {
            JsonObject circular = new JsonObject();
            circular.addProperty("__circular_ref__", clazz.getName());
            return circular;
        }
        seen.put(obj, Boolean.TRUE);

        try {
            // Date/Time types
            if (obj instanceof Date) {
                return new JsonPrimitive(((Date) obj).toInstant().toString());
            }
            if (obj instanceof LocalDateTime) {
                return new JsonPrimitive(obj.toString());
            }
            if (obj instanceof LocalDate) {
                return new JsonPrimitive(obj.toString());
            }
            if (obj instanceof LocalTime) {
                return new JsonPrimitive(obj.toString());
            }

            // Optional
            if (obj instanceof Optional) {
                Optional<?> opt = (Optional<?>) obj;
                if (opt.isPresent()) {
                    return serialize(opt.get(), seen, depth + 1);
                } else {
                    return JsonNull.INSTANCE;
                }
            }

            // Arrays
            if (clazz.isArray()) {
                return serializeArray(obj, seen, depth);
            }

            // Collections
            if (obj instanceof Collection) {
                return serializeCollection((Collection<?>) obj, seen, depth);
            }

            // Maps
            if (obj instanceof Map) {
                return serializeMap((Map<?, ?>) obj, seen, depth);
            }

            // Enums
            if (clazz.isEnum()) {
                return new JsonPrimitive(((Enum<?>) obj).name());
            }

            // Custom objects - serialize via reflection
            return serializeObject(obj, seen, depth);

        } finally {
            seen.remove(obj);
        }
    }

    private static JsonElement serializeArray(Object array, IdentityHashMap<Object, Boolean> seen, int depth) {
        JsonArray jsonArray = new JsonArray();
        int length = java.lang.reflect.Array.getLength(array);
        int limit = Math.min(length, MAX_COLLECTION_SIZE);

        for (int i = 0; i < limit; i++) {
            Object element = java.lang.reflect.Array.get(array, i);
            jsonArray.add(serialize(element, seen, depth + 1));
        }

        if (length > limit) {
            JsonObject truncated = new JsonObject();
            truncated.addProperty("__truncated__", length - limit + " more elements");
            jsonArray.add(truncated);
        }

        return jsonArray;
    }

    private static JsonElement serializeCollection(Collection<?> collection, IdentityHashMap<Object, Boolean> seen, int depth) {
        JsonArray jsonArray = new JsonArray();
        int count = 0;

        for (Object element : collection) {
            if (count >= MAX_COLLECTION_SIZE) {
                JsonObject truncated = new JsonObject();
                truncated.addProperty("__truncated__", collection.size() - count + " more elements");
                jsonArray.add(truncated);
                break;
            }
            jsonArray.add(serialize(element, seen, depth + 1));
            count++;
        }

        return jsonArray;
    }

    private static JsonElement serializeMap(Map<?, ?> map, IdentityHashMap<Object, Boolean> seen, int depth) {
        JsonObject jsonObject = new JsonObject();
        int count = 0;

        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (count >= MAX_COLLECTION_SIZE) {
                jsonObject.addProperty("__truncated__", map.size() - count + " more entries");
                break;
            }
            String key = entry.getKey() != null ? entry.getKey().toString() : "null";
            jsonObject.add(key, serialize(entry.getValue(), seen, depth + 1));
            count++;
        }

        return jsonObject;
    }

    private static JsonElement serializeObject(Object obj, IdentityHashMap<Object, Boolean> seen, int depth) {
        JsonObject jsonObject = new JsonObject();
        Class<?> clazz = obj.getClass();

        // Add type information
        jsonObject.addProperty("__type__", clazz.getName());

        // Serialize all fields (including inherited)
        while (clazz != null && clazz != Object.class) {
            for (Field field : clazz.getDeclaredFields()) {
                // Skip static and transient fields
                if (Modifier.isStatic(field.getModifiers()) ||
                    Modifier.isTransient(field.getModifiers())) {
                    continue;
                }

                try {
                    field.setAccessible(true);
                    Object value = field.get(obj);
                    jsonObject.add(field.getName(), serialize(value, seen, depth + 1));
                } catch (IllegalAccessException e) {
                    jsonObject.addProperty(field.getName(), "__access_denied__");
                }
            }
            clazz = clazz.getSuperclass();
        }

        return jsonObject;
    }

    /**
     * Get a readable class name, handling arrays and primitives.
     */
    private static String getClassName(Class<?> clazz) {
        if (clazz.isArray()) {
            return getClassName(clazz.getComponentType()) + "[]";
        }
        return clazz.getName();
    }

}
