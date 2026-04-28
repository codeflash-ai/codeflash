# Dynamic Tests

Runtime test generation capabilities that allow creating tests programmatically during execution. Dynamic tests enable flexible test scenarios based on runtime data and conditions.

## Imports

```java
import org.junit.jupiter.api.*;
import java.util.stream.Stream;
import java.util.Collection;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;
import static org.junit.jupiter.api.DynamicContainer.dynamicContainer;
```

## Capabilities

### Test Factory Annotation

Mark methods that generate dynamic tests at runtime.

```java { .api }
/**
 * Marks a method as a factory for dynamic tests
 * Method must return Stream, Collection, Iterable, or Iterator of DynamicNode
 */
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@interface TestFactory {
}
```

### Dynamic Test Creation

Create individual dynamic test instances.

```java { .api }
/**
 * A dynamically generated test
 */
class DynamicTest extends DynamicNode {
    /**
     * Create a dynamic test with display name and executable
     */
    static DynamicTest dynamicTest(String displayName, Executable executable);
    
    /**
     * Create a dynamic test with display name, URI, and executable
     */
    static DynamicTest dynamicTest(String displayName, URI testSourceUri, Executable executable);
    
    /**
     * Create a stream of dynamic tests from input data
     */
    static <T> Stream<DynamicTest> stream(Iterator<T> inputGenerator, 
                                          Function<? super T, String> displayNameGenerator,
                                          ThrowingConsumer<? super T> testExecutor);
    
    /**
     * Create a stream of dynamic tests from input stream
     */
    static <T> Stream<DynamicTest> stream(Stream<T> inputStream,
                                          Function<? super T, String> displayNameGenerator, 
                                          ThrowingConsumer<? super T> testExecutor);
}
```

**Usage Examples:**

```java
class DynamicTestExample {
    
    @TestFactory
    Stream<DynamicTest> simpleTests() {
        return Stream.of("apple", "banana", "cherry")
            .map(fruit -> DynamicTest.dynamicTest(
                "test " + fruit,
                () -> {
                    assertNotNull(fruit);
                    assertTrue(fruit.length() > 3);
                }
            ));
    }
    
    @TestFactory
    Collection<DynamicTest> numbersTest() {
        return Arrays.asList(
            DynamicTest.dynamicTest("Test 1", () -> assertEquals(2, 1 + 1)),
            DynamicTest.dynamicTest("Test 2", () -> assertEquals(4, 2 * 2)),
            DynamicTest.dynamicTest("Test 3", () -> assertEquals(9, 3 * 3))
        );
    }
    
    @TestFactory
    Stream<DynamicTest> dataStreamTests() {
        List<String> data = Arrays.asList("alpha", "beta", "gamma");
        
        return DynamicTest.stream(
            data.stream(),
            name -> "Processing " + name,
            value -> {
                assertNotNull(value);
                assertTrue(value.length() > 3);
                assertFalse(value.isEmpty());
            }
        );
    }
}
```

### Dynamic Container Creation

Group related dynamic tests in containers.

```java { .api }
/**
 * A container for dynamic tests or other containers
 */
class DynamicContainer extends DynamicNode {
    /**
     * Create a dynamic container with display name and children
     */
    static DynamicContainer dynamicContainer(String displayName, Stream<DynamicNode> children);
    
    /**
     * Create a dynamic container with display name, URI, and children
     */
    static DynamicContainer dynamicContainer(String displayName, URI testSourceUri, Stream<DynamicNode> children);
    
    /**
     * Create a dynamic container with display name and iterable children
     */
    static DynamicContainer dynamicContainer(String displayName, Iterable<DynamicNode> children);
    
    /**
     * Create a dynamic container with display name, URI, and iterable children
     */
    static DynamicContainer dynamicContainer(String displayName, URI testSourceUri, Iterable<DynamicNode> children);
}
```

**Usage Examples:**

```java
class DynamicContainerExample {
    
    @TestFactory
    Stream<DynamicNode> nestedTests() {
        return Stream.of(
            DynamicContainer.dynamicContainer("String Tests", Stream.of(
                DynamicTest.dynamicTest("Test empty string", () -> assertTrue("".isEmpty())),
                DynamicTest.dynamicTest("Test non-empty string", () -> assertFalse("hello".isEmpty()))
            )),
            
            DynamicContainer.dynamicContainer("Number Tests", Stream.of(
                DynamicTest.dynamicTest("Test positive", () -> assertTrue(5 > 0)),
                DynamicTest.dynamicTest("Test negative", () -> assertTrue(-5 < 0))
            ))
        );
    }
    
    @TestFactory
    Stream<DynamicNode> hierarchicalTests() {
        return Stream.of("Category A", "Category B")
            .map(category -> DynamicContainer.dynamicContainer(
                category,
                IntStream.range(1, 4)
                    .mapToObj(i -> DynamicTest.dynamicTest(
                        category + " Test " + i,
                        () -> {
                            assertNotNull(category);
                            assertTrue(i > 0);
                        }
                    ))
            ));
    }
}
```

### Dynamic Node Base

Base class for all dynamic test elements.

```java { .api }
/**
 * Base class for dynamic tests and containers
 */
abstract class DynamicNode {
    /**
     * Get display name
     */
    String getDisplayName();
    
    /**
     * Get test source URI
     */
    Optional<URI> getTestSourceUri();
}
```

### Named Interface

Interface for providing names to test components.

```java { .api }
/**
 * Interface for named test components
 */
interface Named {
    /**
     * Get the name
     */
    String getName();
    
    /**
     * Create Named instance with given name and payload
     */
    static <T> Named<T> of(String name, T payload);
}

/**
 * Named executable for dynamic test creation
 */
interface NamedExecutable extends Named {
    /**
     * Get the executable
     */
    Executable getExecutable(); 
    
    /**
     * Create NamedExecutable with name and executable
     */
    static NamedExecutable of(String name, Executable executable);
}
```

**Usage Examples:**

```java
class NamedTestExample {
    
    @TestFactory
    Stream<DynamicTest> namedTests() {
        List<Named<String>> testData = Arrays.asList(
            Named.of("First test", "alpha"),
            Named.of("Second test", "beta"),
            Named.of("Third test", "gamma")
        );
        
        return testData.stream()
            .map(namedData -> DynamicTest.dynamicTest(
                namedData.getName(),
                () -> {
                    String value = namedData.getPayload();
                    assertNotNull(value);
                    assertTrue(value.length() > 3);
                }
            ));
    }
    
    @TestFactory
    Stream<DynamicTest> namedExecutableTests() {
        List<NamedExecutable> executables = Arrays.asList(
            NamedExecutable.of("Test addition", () -> assertEquals(4, 2 + 2)),
            NamedExecutable.of("Test subtraction", () -> assertEquals(0, 2 - 2)),
            NamedExecutable.of("Test multiplication", () -> assertEquals(4, 2 * 2))
        );
        
        return executables.stream()
            .map(namedExec -> DynamicTest.dynamicTest(
                namedExec.getName(),
                namedExec.getExecutable()
            ));
    }
}
```

### Complex Dynamic Test Scenarios

Advanced patterns for dynamic test generation.

**Database-driven Tests:**

```java
class DatabaseDrivenTests {
    
    @TestFactory
    Stream<DynamicTest> testAllUsers() {
        UserRepository repository = new UserRepository();
        
        return repository.findAll().stream()
            .map(user -> DynamicTest.dynamicTest(
                "Validate user: " + user.getUsername(),
                () -> {
                    assertNotNull(user.getEmail());
                    assertTrue(user.getAge() >= 0);
                    assertFalse(user.getUsername().isEmpty());
                }
            ));
    }
    
    @TestFactory
    Stream<DynamicNode> testUsersByRole() {
        UserRepository repository = new UserRepository();
        Map<String, List<User>> usersByRole = repository.findAll().stream()
            .collect(Collectors.groupingBy(User::getRole));
        
        return usersByRole.entrySet().stream()
            .map(entry -> DynamicContainer.dynamicContainer(
                "Role: " + entry.getKey(),
                entry.getValue().stream()
                    .map(user -> DynamicTest.dynamicTest(
                        "Test " + user.getUsername(),
                        () -> validateUserInRole(user, entry.getKey())
                    ))
            ));
    }
    
    private void validateUserInRole(User user, String expectedRole) {
        assertEquals(expectedRole, user.getRole());
        assertNotNull(user.getPermissions());
        assertFalse(user.getPermissions().isEmpty());
    }
}
```

**Configuration-based Tests:**

```java
class ConfigurationBasedTests {
    
    @TestFactory
    Stream<DynamicTest> testConfigurations() throws IOException {
        Properties configs = loadTestConfigurations();
        
        return configs.entrySet().stream()
            .map(entry -> DynamicTest.dynamicTest(
                "Test config: " + entry.getKey(),
                () -> {
                    String key = (String) entry.getKey();
                    String value = (String) entry.getValue();
                    
                    assertNotNull(value, "Configuration value should not be null");
                    validateConfigurationValue(key, value);
                }
            ));
    }
    
    @TestFactory
    Stream<DynamicNode> testConfigurationGroups() throws IOException {
        Map<String, Properties> configGroups = loadConfigurationGroups();
        
        return configGroups.entrySet().stream()
            .map(group -> DynamicContainer.dynamicContainer(
                "Config Group: " + group.getKey(),
                group.getValue().entrySet().stream()
                    .map(config -> DynamicTest.dynamicTest(
                        "Test " + config.getKey(),
                        () -> validateConfigurationValue(
                            (String) config.getKey(), 
                            (String) config.getValue()
                        )
                    ))
            ));
    }
    
    private Properties loadTestConfigurations() throws IOException {
        Properties props = new Properties();
        props.load(getClass().getResourceAsStream("/test-config.properties"));
        return props;
    }
    
    private Map<String, Properties> loadConfigurationGroups() {
        // Load different configuration groups
        return Map.of(
            "database", loadDatabaseConfigs(),
            "security", loadSecurityConfigs(),
            "performance", loadPerformanceConfigs()
        );
    }
    
    private void validateConfigurationValue(String key, String value) {
        switch (key) {
            case "timeout":
                assertTrue(Integer.parseInt(value) > 0);
                break;
            case "url":
                assertTrue(value.startsWith("http"));
                break;
            default:
                assertNotNull(value);
        }
    }
}
```