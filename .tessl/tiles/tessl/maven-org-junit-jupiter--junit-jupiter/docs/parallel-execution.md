# Parallel Execution and Resource Management

Configuration for parallel test execution, resource locking, and temporary file management. JUnit Jupiter provides fine-grained control over test concurrency and resource access.

## Imports

```java
import org.junit.jupiter.api.parallel.*;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Path;
import static org.junit.jupiter.api.Assertions.*;
```

## Capabilities

### Parallel Execution Configuration

Control concurrent execution of tests and test classes.

```java { .api }
/**
 * Configure parallel execution mode for tests
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@interface Execution {
    /**
     * Execution mode for this test or test class
     */
    ExecutionMode value();
}

/**
 * Execution mode enumeration
 */
enum ExecutionMode {
    /**
     * Execute in same thread as parent
     */
    SAME_THREAD,
    
    /**
     * Execute concurrently with other tests (if parallel execution enabled)
     */
    CONCURRENT
}
```

**Usage Examples:**

```java
// Enable concurrent execution for entire test class
@Execution(ExecutionMode.CONCURRENT)
class ParallelTest {
    
    @Test
    void test1() {
        // Runs concurrently with other tests
        performIndependentOperation();
    }
    
    @Test
    void test2() {
        // Runs concurrently with other tests
        performAnotherIndependentOperation();
    }
    
    @Test
    @Execution(ExecutionMode.SAME_THREAD)
    void sequentialTest() {
        // Runs sequentially despite class-level concurrent setting
        performSequentialOperation();
    }
}

// Mixed execution modes
class MixedExecutionTest {
    
    @Test
    @Execution(ExecutionMode.CONCURRENT)
    void concurrentTest1() {
        // Runs concurrently
    }
    
    @Test
    @Execution(ExecutionMode.CONCURRENT)
    void concurrentTest2() {
        // Runs concurrently
    }
    
    @Test
    void defaultTest() {
        // Uses default execution mode
    }
}
```

### Test Isolation

Force sequential execution for tests that require isolation.

```java { .api }
/**
 * Force sequential execution in separate classloader
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@interface Isolated {
}
```

**Usage Example:**

```java
@Isolated
class IsolatedTest {
    
    @Test
    void testThatModifiesGlobalState() {
        System.setProperty("test.mode", "isolated");
        // Test runs in isolation
    }
    
    @Test
    void anotherIsolatedTest() {
        // Also runs in isolation
    }
}

class RegularTest {
    
    @Test
    @Isolated
    void isolatedMethod() {
        // Only this method runs in isolation
    }
    
    @Test
    void regularMethod() {
        // Regular execution
    }
}
```

### Resource Locking

Coordinate access to shared resources across concurrent tests.

```java { .api }
/**
 * Lock access to a shared resource
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@Repeatable(ResourceLocks.class)
@interface ResourceLock {
    /**
     * Resource identifier
     */
    String value();
    
    /**
     * Access mode for the resource
     */
    ResourceAccessMode mode() default ResourceAccessMode.READ_WRITE;
    
    /**
     * Target level for the lock
     */
    ResourceLockTarget target() default ResourceLockTarget.METHOD;
}

/**
 * Container for multiple resource locks
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@interface ResourceLocks {
    ResourceLock[] value();
}

/**
 * Resource access mode
 */
enum ResourceAccessMode {
    /**
     * Exclusive read-write access
     */
    READ_WRITE,
    
    /**
     * Shared read-only access
     */
    READ
}

/**
 * Resource lock target level
 */
enum ResourceLockTarget {
    /**
     * Lock applies to individual method
     */
    METHOD,
    
    /**
     * Lock applies to entire class
     */
    CLASS
}
```

**Usage Examples:**

```java
class ResourceLockTest {
    
    @Test
    @ResourceLock("database")
    void testDatabaseWrite() {
        // Exclusive access to database resource
        database.insert("test data");
    }
    
    @Test
    @ResourceLock(value = "database", mode = ResourceAccessMode.READ)
    void testDatabaseRead1() {
        // Shared read access - can run concurrently with other read tests
        String data = database.select("test data");
        assertNotNull(data);
    }
    
    @Test
    @ResourceLock(value = "database", mode = ResourceAccessMode.READ)
    void testDatabaseRead2() {
        // Shared read access - can run concurrently with testDatabaseRead1
        int count = database.count();
        assertTrue(count >= 0);
    }
    
    @Test
    @ResourceLocks({
        @ResourceLock("database"),
        @ResourceLock("filesystem")
    })
    void testMultipleResources() {
        // Requires exclusive access to both database and filesystem
        database.backup("/tmp/backup");
    }
}

@ResourceLock(value = "system-properties", target = ResourceLockTarget.CLASS)
class SystemPropertiesTest {
    
    @Test
    void testSystemProperty1() {
        System.setProperty("test.prop", "value1");
        // Entire class has exclusive access to system properties
    }
    
    @Test
    void testSystemProperty2() {
        System.setProperty("test.prop", "value2");
        // Sequential execution guaranteed
    }
}
```

### Standard Resources

Pre-defined resource identifiers for common shared resources.

```java { .api }
/**
 * Standard resource constants
 */
class Resources {
    /**
     * Global resource lock
     */
    public static final String GLOBAL = "GLOBAL";
    
    /**
     * Java system properties
     */
    public static final String SYSTEM_PROPERTIES = "SYSTEM_PROPERTIES";
    
    /**
     * Java system environment
     */
    public static final String SYSTEM_ENVIRONMENT = "SYSTEM_ENVIRONMENT";
    
    /**
     * Standard input/output streams
     */
    public static final String SYSTEM_OUT = "SYSTEM_OUT";
    public static final String SYSTEM_ERR = "SYSTEM_ERR";
    public static final String SYSTEM_IN = "SYSTEM_IN";
    
    /**
     * Java locale settings
     */
    public static final String LOCALE = "LOCALE";
    
    /**
     * Java time zone settings
     */
    public static final String TIME_ZONE = "TIME_ZONE";
}
```

**Usage Examples:**

```java
class StandardResourcesTest {
    
    @Test
    @ResourceLock(Resources.SYSTEM_PROPERTIES)
    void testWithSystemProperties() {
        String original = System.getProperty("user.dir");
        System.setProperty("user.dir", "/tmp");
        
        // Test with modified system property
        assertEquals("/tmp", System.getProperty("user.dir"));
        
        // Restore
        System.setProperty("user.dir", original);
    }
    
    @Test
    @ResourceLock(Resources.SYSTEM_OUT)
    void testWithSystemOut() {
        PrintStream originalOut = System.out;
        ByteArrayOutputStream capturedOut = new ByteArrayOutputStream();
        System.setOut(new PrintStream(capturedOut));
        
        System.out.println("Test output");
        assertEquals("Test output\n", capturedOut.toString());
        
        System.setOut(originalOut);
    }
    
    @Test
    @ResourceLock(Resources.LOCALE)
    void testWithLocale() {
        Locale original = Locale.getDefault();
        Locale.setDefault(Locale.FRENCH);
        
        // Test with French locale
        assertEquals(Locale.FRENCH, Locale.getDefault());
        
        Locale.setDefault(original);
    }
}
```

### Custom Resource Locks Provider

Programmatically provide resource locks based on test context.

```java { .api }
/**
 * Provides resource locks programmatically
 */
interface ResourceLocksProvider {
    /**
     * Provide resource locks for the given extension context
     */
    Set<Lock> provideForClass(ExtensionContext context);
    Set<Lock> provideForNestedClass(ExtensionContext context);
    Set<Lock> provideForMethod(ExtensionContext context);
    
    /**
     * Resource lock representation
     */
    interface Lock {
        String getKey();
        ResourceAccessMode getAccessMode();
    }
}
```

### Temporary Directory Support

Automatic temporary directory creation and cleanup for tests.

```java { .api }
/**
 * Inject temporary directory into test method or field
 */
@Target({ElementType.FIELD, ElementType.PARAMETER})
@Retention(RetentionPolicy.RUNTIME)
@interface TempDir {
    /**
     * Cleanup mode for temporary directory
     */
    CleanupMode cleanup() default CleanupMode.DEFAULT;
    
    /**
     * Factory for creating temporary directories
     */
    Class<? extends TempDirFactory> factory() default TempDirFactory.Standard.class;
}

/**
 * Cleanup mode for temporary directories
 */
enum CleanupMode {
    /**
     * Use default cleanup behavior
     */
    DEFAULT,
    
    /**
     * Never clean up temporary directories
     */
    NEVER,
    
    /**
     * Always clean up temporary directories
     */
    ALWAYS,
    
    /**
     * Clean up on success, keep on failure
     */
    ON_SUCCESS
}

/**
 * Factory for creating temporary directories
 */
interface TempDirFactory {
    /**
     * Create temporary directory
     */
    Path createTempDirectory(AnnotatedElement annotatedElement, ExtensionContext extensionContext) throws IOException;
    
    /**
     * Standard temporary directory factory
     */
    class Standard implements TempDirFactory {
        @Override
        public Path createTempDirectory(AnnotatedElement annotatedElement, ExtensionContext extensionContext) throws IOException {
            return Files.createTempDirectory("junit");
        }
    }
}
```

**Usage Examples:**

```java
class TempDirTest {
    
    @TempDir
    Path sharedTempDir;
    
    @Test
    void testWithSharedTempDir() throws IOException {
        Path file = sharedTempDir.resolve("test.txt");
        Files.write(file, "test content".getBytes());
        
        assertTrue(Files.exists(file));
        assertEquals("test content", Files.readString(file));
    }
    
    @Test
    void testWithMethodTempDir(@TempDir Path tempDir) throws IOException {
        // Each test method gets its own temp directory
        assertNotEquals(sharedTempDir, tempDir);
        
        Path file = tempDir.resolve("method-test.txt");
        Files.createFile(file);
        assertTrue(Files.exists(file));
    }
    
    @Test
    void testWithCustomCleanup(@TempDir(cleanup = CleanupMode.NEVER) Path persistentDir) throws IOException {
        // This directory won't be cleaned up automatically
        Path file = persistentDir.resolve("persistent.txt");
        Files.write(file, "This file will persist".getBytes());
        
        System.out.println("Persistent dir: " + persistentDir);
    }
    
    @Test
    void testWithCustomFactory(@TempDir(factory = CustomTempDirFactory.class) Path customDir) {
        // Directory created by custom factory
        assertTrue(customDir.toString().contains("custom"));
    }
}

class CustomTempDirFactory implements TempDirFactory {
    @Override
    public Path createTempDirectory(AnnotatedElement annotatedElement, ExtensionContext extensionContext) throws IOException {
        return Files.createTempDirectory("custom-junit-" + extensionContext.getDisplayName());
    }
}
```

### Configuration Properties

Configure parallel execution behavior through system properties or configuration files.

**Key Configuration Properties:**

```properties
# Enable parallel execution
junit.jupiter.execution.parallel.enabled=true

# Default execution mode
junit.jupiter.execution.parallel.mode.default=concurrent

# Class-level execution mode  
junit.jupiter.execution.parallel.mode.classes.default=concurrent

# Parallelism strategy
junit.jupiter.execution.parallel.config.strategy=dynamic
# or fixed with custom thread count
junit.jupiter.execution.parallel.config.strategy=fixed
junit.jupiter.execution.parallel.config.fixed.parallelism=4

# Dynamic parallelism factor
junit.jupiter.execution.parallel.config.dynamic.factor=2.0
```

**Usage in junit-platform.properties:**

```properties
junit.jupiter.execution.parallel.enabled=true
junit.jupiter.execution.parallel.mode.default=concurrent
junit.jupiter.execution.parallel.mode.classes.default=same_thread
junit.jupiter.execution.parallel.config.strategy=dynamic
junit.jupiter.execution.parallel.config.dynamic.factor=1.5
```