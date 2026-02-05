package com.codeflash;

import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.*;

/**
 * Deep object comparison for verifying serialization/deserialization correctness.
 *
 * This comparator is used to verify that objects survive the serialize-deserialize
 * cycle correctly. It handles:
 * - Primitives and wrappers with epsilon tolerance for floats
 * - Collections, Maps, and Arrays
 * - Custom objects via reflection
 * - NaN and Infinity special cases
 * - Exception comparison
 * - KryoPlaceholder rejection
 */
public final class ObjectComparator {

    private static final double EPSILON = 1e-9;

    private ObjectComparator() {
        // Utility class
    }

    /**
     * Compare two objects for deep equality.
     *
     * @param orig   The original object
     * @param newObj The object to compare against
     * @return true if objects are equivalent
     * @throws KryoPlaceholderAccessException if comparison involves a placeholder
     */
    public static boolean compare(Object orig, Object newObj) {
        return compareInternal(orig, newObj, new IdentityHashMap<>());
    }

    /**
     * Compare two objects, returning a detailed result.
     *
     * @param orig   The original object
     * @param newObj The object to compare against
     * @return ComparisonResult with details about the comparison
     */
    public static ComparisonResult compareWithDetails(Object orig, Object newObj) {
        try {
            boolean equal = compareInternal(orig, newObj, new IdentityHashMap<>());
            return new ComparisonResult(equal, null);
        } catch (KryoPlaceholderAccessException e) {
            return new ComparisonResult(false, e.getMessage());
        }
    }

    private static boolean compareInternal(Object orig, Object newObj,
                                           IdentityHashMap<Object, Object> seen) {
        // Handle nulls
        if (orig == null && newObj == null) {
            return true;
        }
        if (orig == null || newObj == null) {
            return false;
        }

        // Detect and reject KryoPlaceholder
        if (orig instanceof KryoPlaceholder) {
            KryoPlaceholder p = (KryoPlaceholder) orig;
            throw new KryoPlaceholderAccessException(
                "Cannot compare: original contains placeholder for unserializable object",
                p.getObjType(), p.getPath());
        }
        if (newObj instanceof KryoPlaceholder) {
            KryoPlaceholder p = (KryoPlaceholder) newObj;
            throw new KryoPlaceholderAccessException(
                "Cannot compare: new object contains placeholder for unserializable object",
                p.getObjType(), p.getPath());
        }

        // Handle exceptions specially
        if (orig instanceof Throwable && newObj instanceof Throwable) {
            return compareExceptions((Throwable) orig, (Throwable) newObj);
        }

        Class<?> origClass = orig.getClass();
        Class<?> newClass = newObj.getClass();

        // Check type compatibility
        if (!origClass.equals(newClass)) {
            if (!areTypesCompatible(origClass, newClass)) {
                return false;
            }
        }

        // Handle primitives and wrappers
        if (orig instanceof Boolean) {
            return orig.equals(newObj);
        }
        if (orig instanceof Character) {
            return orig.equals(newObj);
        }
        if (orig instanceof String) {
            return orig.equals(newObj);
        }
        if (orig instanceof Number) {
            return compareNumbers((Number) orig, (Number) newObj);
        }

        // Handle enums
        if (origClass.isEnum()) {
            return orig.equals(newObj);
        }

        // Handle Class objects
        if (orig instanceof Class) {
            return orig.equals(newObj);
        }

        // Handle date/time types
        if (orig instanceof Date || orig instanceof LocalDateTime ||
            orig instanceof LocalDate || orig instanceof LocalTime) {
            return orig.equals(newObj);
        }

        // Handle Optional
        if (orig instanceof Optional && newObj instanceof Optional) {
            return compareOptionals((Optional<?>) orig, (Optional<?>) newObj, seen);
        }

        // Check for circular reference to prevent infinite recursion
        if (seen.containsKey(orig)) {
            // If we've seen this object before, just check identity
            return seen.get(orig) == newObj;
        }
        seen.put(orig, newObj);

        try {
            // Handle arrays
            if (origClass.isArray()) {
                return compareArrays(orig, newObj, seen);
            }

            // Handle collections
            if (orig instanceof Collection && newObj instanceof Collection) {
                return compareCollections((Collection<?>) orig, (Collection<?>) newObj, seen);
            }

            // Handle maps
            if (orig instanceof Map && newObj instanceof Map) {
                return compareMaps((Map<?, ?>) orig, (Map<?, ?>) newObj, seen);
            }

            // Handle general objects via reflection
            return compareObjects(orig, newObj, seen);

        } finally {
            seen.remove(orig);
        }
    }

    /**
     * Check if two types are compatible for comparison.
     */
    private static boolean areTypesCompatible(Class<?> type1, Class<?> type2) {
        // Allow comparing different Collection implementations
        if (Collection.class.isAssignableFrom(type1) && Collection.class.isAssignableFrom(type2)) {
            return true;
        }
        // Allow comparing different Map implementations
        if (Map.class.isAssignableFrom(type1) && Map.class.isAssignableFrom(type2)) {
            return true;
        }
        // Allow comparing different Number types
        if (Number.class.isAssignableFrom(type1) && Number.class.isAssignableFrom(type2)) {
            return true;
        }
        return false;
    }

    /**
     * Compare two numbers with epsilon tolerance for floating point.
     */
    private static boolean compareNumbers(Number n1, Number n2) {
        // Handle floating point with epsilon
        if (n1 instanceof Double || n1 instanceof Float ||
            n2 instanceof Double || n2 instanceof Float) {

            double d1 = n1.doubleValue();
            double d2 = n2.doubleValue();

            // Handle NaN
            if (Double.isNaN(d1) && Double.isNaN(d2)) {
                return true;
            }
            if (Double.isNaN(d1) || Double.isNaN(d2)) {
                return false;
            }

            // Handle Infinity
            if (Double.isInfinite(d1) && Double.isInfinite(d2)) {
                return (d1 > 0) == (d2 > 0); // Same sign
            }
            if (Double.isInfinite(d1) || Double.isInfinite(d2)) {
                return false;
            }

            // Compare with epsilon
            return Math.abs(d1 - d2) < EPSILON;
        }

        // Integer types - exact comparison
        return n1.longValue() == n2.longValue();
    }

    /**
     * Compare two exceptions.
     */
    private static boolean compareExceptions(Throwable orig, Throwable newEx) {
        // Must be same type
        if (!orig.getClass().equals(newEx.getClass())) {
            return false;
        }
        // Compare message (both may be null)
        return Objects.equals(orig.getMessage(), newEx.getMessage());
    }

    /**
     * Compare two Optional values.
     */
    private static boolean compareOptionals(Optional<?> orig, Optional<?> newOpt,
                                            IdentityHashMap<Object, Object> seen) {
        if (orig.isPresent() != newOpt.isPresent()) {
            return false;
        }
        if (!orig.isPresent()) {
            return true; // Both empty
        }
        return compareInternal(orig.get(), newOpt.get(), seen);
    }

    /**
     * Compare two arrays.
     */
    private static boolean compareArrays(Object orig, Object newObj,
                                         IdentityHashMap<Object, Object> seen) {
        int length1 = Array.getLength(orig);
        int length2 = Array.getLength(newObj);

        if (length1 != length2) {
            return false;
        }

        for (int i = 0; i < length1; i++) {
            Object elem1 = Array.get(orig, i);
            Object elem2 = Array.get(newObj, i);
            if (!compareInternal(elem1, elem2, seen)) {
                return false;
            }
        }

        return true;
    }

    /**
     * Compare two collections.
     */
    private static boolean compareCollections(Collection<?> orig, Collection<?> newColl,
                                              IdentityHashMap<Object, Object> seen) {
        if (orig.size() != newColl.size()) {
            return false;
        }

        // For Sets, compare element-by-element (order doesn't matter)
        if (orig instanceof Set && newColl instanceof Set) {
            return compareSets((Set<?>) orig, (Set<?>) newColl, seen);
        }

        // For ordered collections (List, etc.), compare in order
        Iterator<?> iter1 = orig.iterator();
        Iterator<?> iter2 = newColl.iterator();

        while (iter1.hasNext() && iter2.hasNext()) {
            if (!compareInternal(iter1.next(), iter2.next(), seen)) {
                return false;
            }
        }

        return !iter1.hasNext() && !iter2.hasNext();
    }

    /**
     * Compare two sets (order-independent).
     */
    private static boolean compareSets(Set<?> orig, Set<?> newSet,
                                       IdentityHashMap<Object, Object> seen) {
        if (orig.size() != newSet.size()) {
            return false;
        }

        // For each element in orig, find a matching element in newSet
        for (Object elem1 : orig) {
            boolean found = false;
            for (Object elem2 : newSet) {
                try {
                    if (compareInternal(elem1, elem2, new IdentityHashMap<>(seen))) {
                        found = true;
                        break;
                    }
                } catch (KryoPlaceholderAccessException e) {
                    // Propagate placeholder exceptions
                    throw e;
                }
            }
            if (!found) {
                return false;
            }
        }
        return true;
    }

    /**
     * Compare two maps.
     */
    private static boolean compareMaps(Map<?, ?> orig, Map<?, ?> newMap,
                                       IdentityHashMap<Object, Object> seen) {
        if (orig.size() != newMap.size()) {
            return false;
        }

        for (Map.Entry<?, ?> entry : orig.entrySet()) {
            Object key = entry.getKey();
            Object value1 = entry.getValue();

            if (!newMap.containsKey(key)) {
                return false;
            }

            Object value2 = newMap.get(key);
            if (!compareInternal(value1, value2, seen)) {
                return false;
            }
        }

        return true;
    }

    /**
     * Compare two objects via reflection.
     */
    private static boolean compareObjects(Object orig, Object newObj,
                                          IdentityHashMap<Object, Object> seen) {
        Class<?> clazz = orig.getClass();

        // If class has a custom equals method, use it
        try {
            if (hasCustomEquals(clazz)) {
                return orig.equals(newObj);
            }
        } catch (Exception e) {
            // Fall through to field comparison
        }

        // Compare all fields via reflection
        Class<?> currentClass = clazz;
        while (currentClass != null && currentClass != Object.class) {
            for (Field field : currentClass.getDeclaredFields()) {
                if (Modifier.isStatic(field.getModifiers()) ||
                    Modifier.isTransient(field.getModifiers())) {
                    continue;
                }

                try {
                    field.setAccessible(true);
                    Object value1 = field.get(orig);
                    Object value2 = field.get(newObj);

                    if (!compareInternal(value1, value2, seen)) {
                        return false;
                    }
                } catch (IllegalAccessException e) {
                    // Can't access field - assume not equal
                    return false;
                }
            }
            currentClass = currentClass.getSuperclass();
        }

        return true;
    }

    /**
     * Check if a class has a custom equals method (not from Object).
     */
    private static boolean hasCustomEquals(Class<?> clazz) {
        try {
            java.lang.reflect.Method equalsMethod = clazz.getMethod("equals", Object.class);
            return equalsMethod.getDeclaringClass() != Object.class;
        } catch (NoSuchMethodException e) {
            return false;
        }
    }

    /**
     * Result of a comparison with optional error details.
     */
    public static class ComparisonResult {
        private final boolean equal;
        private final String errorMessage;

        public ComparisonResult(boolean equal, String errorMessage) {
            this.equal = equal;
            this.errorMessage = errorMessage;
        }

        public boolean isEqual() {
            return equal;
        }

        public String getErrorMessage() {
            return errorMessage;
        }

        public boolean hasError() {
            return errorMessage != null;
        }
    }
}
