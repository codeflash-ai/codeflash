# Conditional Execution

Rich set of built-in conditions for controlling test execution based on runtime environment, system properties, and custom logic. Tests can be enabled or disabled dynamically based on various criteria.

## Imports

```java
import org.junit.jupiter.api.condition.*;
import static org.junit.jupiter.api.Assertions.*;
```

## Capabilities

### Operating System Conditions

Enable or disable tests based on the operating system.

```java { .api }
/**
 * Enable test on specific operating systems
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ExtendWith(EnabledOnOsCondition.class)
@interface EnabledOnOs {
    OS[] value();
    String disabledReason() default "";
}

/**
 * Disable test on specific operating systems
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ExtendWith(DisabledOnOsCondition.class)
@interface DisabledOnOs {
    OS[] value();
    String disabledReason() default "";
}

/**
 * Operating system enumeration
 */
enum OS {
    LINUX,
    MAC,
    WINDOWS,
    AIX,
    SOLARIS,
    OTHER
}
```

**Usage Examples:**

```java
class OsSpecificTest {
    
    @Test
    @EnabledOnOs(OS.LINUX)
    void testOnLinuxOnly() {
        assertEquals("/", File.separator);
    }
    
    @Test
    @EnabledOnOs({OS.WINDOWS, OS.MAC})
    void testOnWindowsOrMac() {
        assertNotEquals("/", File.separator);
    }
    
    @Test
    @DisabledOnOs(value = OS.WINDOWS, disabledReason = "Windows path handling differs")
    void testUnixPaths() {
        assertTrue(Paths.get("/usr/local").isAbsolute());
    }
}
```

### Java Runtime Environment Conditions

Control execution based on JRE version.

```java { .api }
/**
 * Enable test on specific JRE versions
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ExtendWith(EnabledOnJreCondition.class)
@interface EnabledOnJre {
    JRE[] value();
    String disabledReason() default "";
}

/**
 * Disable test on specific JRE versions
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ExtendWith(DisabledOnJreCondition.class)
@interface DisabledOnJre {
    JRE[] value();
    String disabledReason() default "";
}

/**
 * Enable test for JRE version ranges
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ExtendWith(EnabledForJreRangeCondition.class)
@interface EnabledForJreRange {
    JRE min() default JRE.JAVA_8;
    JRE max() default JRE.OTHER;
    String disabledReason() default "";
}

/**
 * Disable test for JRE version ranges
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ExtendWith(DisabledForJreRangeCondition.class)
@interface DisabledForJreRange {
    JRE min() default JRE.JAVA_8;
    JRE max() default JRE.OTHER;
    String disabledReason() default "";
}
```

**Usage Examples:**

```java
class JreSpecificTest {
    
    @Test
    @EnabledOnJre(JRE.JAVA_8)
    void testOnJava8Only() {
        // Java 8 specific functionality
    }
    
    @Test
    @EnabledForJreRange(min = JRE.JAVA_11, max = JRE.JAVA_17)
    void testOnJava11To17() {
        // Features available in Java 11-17
    }
    
    @Test
    @DisabledOnJre(value = JRE.JAVA_8, disabledReason = "Lambda syntax not supported")
    void testWithModernJavaFeatures() {
        // Modern Java features
        var list = List.of("item1", "item2");
        assertFalse(list.isEmpty());
    }
}
```

### System Property Conditions

Execute tests conditionally based on system properties.

```java { .api }
/**
 * Enable test if system property matches
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@Repeatable(EnabledIfSystemProperties.class)
@ExtendWith(EnabledIfSystemPropertyCondition.class)
@interface EnabledIfSystemProperty {
    String named();
    String matches();
    String disabledReason() default "";
}

/**
 * Container for multiple system property conditions
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ExtendWith(EnabledIfSystemPropertyCondition.class)
@interface EnabledIfSystemProperties {
    EnabledIfSystemProperty[] value();
}

/**
 * Disable test if system property matches
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@Repeatable(DisabledIfSystemProperties.class)
@ExtendWith(DisabledIfSystemPropertyCondition.class)
@interface DisabledIfSystemProperty {
    String named();
    String matches();
    String disabledReason() default "";
}

/**
 * Container for multiple system property conditions
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ExtendWith(DisabledIfSystemPropertyCondition.class)
@interface DisabledIfSystemProperties {
    DisabledIfSystemProperty[] value();
}
```

**Usage Examples:**

```java
class SystemPropertyTest {
    
    @Test
    @EnabledIfSystemProperty(named = "env", matches = "dev")
    void testInDevelopmentOnly() {
        // Development environment specific test
    }
    
    @Test
    @EnabledIfSystemProperty(named = "debug", matches = "true")
    void testWithDebugEnabled() {
        // Debug mode specific test
    }
    
    @Test
    @DisabledIfSystemProperty(named = "ci", matches = "true", 
                              disabledReason = "Flaky in CI environment")
    void testDisabledInCI() {
        // Test that's unreliable in CI
    }
    
    @Test
    @EnabledIfSystemProperties({
        @EnabledIfSystemProperty(named = "env", matches = "test"),
        @EnabledIfSystemProperty(named = "db.enabled", matches = "true")
    })
    void testWithMultipleProperties() {
        // Test requiring multiple system properties
    }
}
```

### Environment Variable Conditions

Execute tests conditionally based on environment variables.

```java { .api }
/**
 * Enable test if environment variable matches
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@Repeatable(EnabledIfEnvironmentVariables.class)
@ExtendWith(EnabledIfEnvironmentVariableCondition.class)
@interface EnabledIfEnvironmentVariable {
    String named();
    String matches();
    String disabledReason() default "";
}

/**
 * Container for multiple environment variable conditions
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ExtendWith(EnabledIfEnvironmentVariableCondition.class)  
@interface EnabledIfEnvironmentVariables {
    EnabledIfEnvironmentVariable[] value();
}

/**
 * Disable test if environment variable matches
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@Repeatable(DisabledIfEnvironmentVariables.class)
@ExtendWith(DisabledIfEnvironmentVariableCondition.class)
@interface DisabledIfEnvironmentVariable {
    String named();
    String matches();
    String disabledReason() default "";
}

/**
 * Container for multiple environment variable conditions
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ExtendWith(DisabledIfEnvironmentVariableCondition.class)
@interface DisabledIfEnvironmentVariables {
    DisabledIfEnvironmentVariable[] value();
}
```

**Usage Examples:**

```java
class EnvironmentVariableTest {
    
    @Test
    @EnabledIfEnvironmentVariable(named = "ENV", matches = "production")
    void testInProductionOnly() {
        // Production-specific test
    }
    
    @Test
    @EnabledIfEnvironmentVariable(named = "DATABASE_URL", matches = ".*localhost.*")
    void testWithLocalDatabase() {
        // Test requiring local database
    }
    
    @Test
    @DisabledIfEnvironmentVariable(named = "SKIP_SLOW_TESTS", matches = "true")
    void slowTest() throws InterruptedException {
        Thread.sleep(5000);
        assertTrue(true);
    }
}
```

### Custom Condition Methods

Execute tests based on custom boolean methods.

```java { .api }
/**
 * Enable test if custom condition method returns true
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ExtendWith(EnabledIfCondition.class)
@interface EnabledIf {
    /**
     * Method name that returns boolean
     */
    String value();
    String disabledReason() default "";
}

/**
 * Disable test if custom condition method returns true
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ExtendWith(DisabledIfCondition.class)
@interface DisabledIf {
    /**
     * Method name that returns boolean
     */
    String value();
    String disabledReason() default "";
}
```

**Usage Examples:**

```java
class CustomConditionTest {
    
    @Test
    @EnabledIf("isExternalServiceAvailable")
    void testWithExternalService() {
        // Test that requires external service
    }
    
    @Test
    @DisabledIf("isWeekend")
    void testDisabledOnWeekends() {
        // Test that shouldn't run on weekends
    }
    
    static boolean isExternalServiceAvailable() {
        try {
            // Check if external service is reachable
            URL url = new URL("http://api.example.com/health");
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            connection.setConnectTimeout(1000);
            return connection.getResponseCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }
    
    static boolean isWeekend() {
        DayOfWeek today = LocalDate.now().getDayOfWeek();
        return today == DayOfWeek.SATURDAY || today == DayOfWeek.SUNDAY;
    }
}
```

### GraalVM Native Image Conditions

Control execution in GraalVM native image contexts.

```java { .api }
/**
 * Enable test only in GraalVM native image
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@interface EnabledInNativeImage {
    String disabledReason() default "";
}

/**
 * Disable test in GraalVM native image
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@interface DisabledInNativeImage {
    String disabledReason() default "";
}
```

**Usage Examples:**

```java
class NativeImageTest {
    
    @Test
    @EnabledInNativeImage
    void testNativeImageSpecificBehavior() {
        // Test behavior specific to native image
    }
    
    @Test
    @DisabledInNativeImage(disabledReason = "Reflection not available in native image")
    void testWithReflection() {
        // Test using reflection APIs
        Class<?> clazz = String.class;
        Method[] methods = clazz.getDeclaredMethods();
        assertTrue(methods.length > 0);
    }
}
```