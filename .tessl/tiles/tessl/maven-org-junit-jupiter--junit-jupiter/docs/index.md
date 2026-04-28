# JUnit Jupiter

JUnit Jupiter is the new programming and extension model for JUnit 5, providing a comprehensive testing framework for Java applications. As an aggregator module, it combines the core JUnit Jupiter API, parameterized test support, and the Jupiter test engine to deliver a unified, modern testing experience with advanced features like nested tests, dynamic tests, custom extensions, and parallel execution.

## Package Information

- **Package Name**: org.junit.jupiter:junit-jupiter
- **Package Type**: Maven
- **Language**: Java
- **Installation**: Add to Maven `pom.xml`:

```xml
<dependency>
    <groupId>org.junit.jupiter</groupId>
    <artifactId>junit-jupiter</artifactId>
    <version>5.12.2</version>
    <scope>test</scope>
</dependency>
```

Or Gradle `build.gradle`:

```groovy
testImplementation 'org.junit.jupiter:junit-jupiter:5.12.2'
```

## Core Imports

```java
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.*;
```

Common static imports for assertions:

```java
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;
```

## Basic Usage

```java
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

class CalculatorTest {

    @Test
    @DisplayName("Addition should work correctly")
    void testAddition() {
        Calculator calc = new Calculator();
        assertEquals(5, calc.add(2, 3));
        assertNotNull(calc);
    }

    @BeforeEach
    void setUp() {
        // Setup before each test
        System.out.println("Setting up test");
    }

    @AfterEach
    void tearDown() {
        // Cleanup after each test
        System.out.println("Cleaning up test");
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 3, 4, 5})
    void testMultipleValues(int value) {
        assertTrue(value > 0);
    }
}
```

## Architecture

JUnit Jupiter is built around several key components:

- **Test API**: Core annotations and interfaces for writing tests (`@Test`, `@BeforeEach`, etc.)
- **Assertion Engine**: Comprehensive assertion methods with descriptive failure messages
- **Extension Model**: Powerful extension system for custom behavior and integrations
- **Test Engine**: Runtime execution engine that discovers and runs tests
- **Parameter Resolution**: Dependency injection system for test methods and constructors
- **Conditional Execution**: Rich set of conditions for enabling/disabling tests based on environment

## Capabilities

### Core Testing API

Essential testing annotations, lifecycle methods, and basic test structure. Provides the foundation for writing JUnit 5 tests with modern Java features.

```java { .api }
@Test
@BeforeAll
@BeforeEach
@AfterEach
@AfterAll
@DisplayName(String value)
@Nested
@Disabled(String reason)
@Timeout(long value, TimeUnit unit)
```

[Core Testing](./core-testing.md)

### Assertions and Assumptions

Comprehensive assertion methods for verifying test conditions and conditional test execution based on assumptions.

```java { .api }
// Core assertions
static void assertEquals(Object expected, Object actual);
static void assertTrue(boolean condition);
static void assertThrows(Class<T> expectedType, Executable executable);
static void assertAll(Executable... executables);

// Assumptions
static void assumeTrue(boolean assumption);
static void assumingThat(boolean assumption, Executable executable);
```

[Assertions and Assumptions](./assertions.md)

### Parameterized Tests

Advanced parameterized testing with multiple data sources, argument conversion, and aggregation for data-driven test scenarios.

```java { .api }
@ParameterizedTest
@ValueSource(ints = {1, 2, 3})
@CsvSource({"1,John", "2,Jane"})
@MethodSource("argumentProvider")
void parameterizedTest(int value, String name);
```

[Parameterized Tests](./parameterized-tests.md)

### Extensions and Lifecycle

Powerful extension model for customizing test behavior, dependency injection, and integrating with external frameworks.

```java { .api }
@ExtendWith(MyExtension.class)
@RegisterExtension
static MyExtension extension = new MyExtension();

interface Extension { }
interface BeforeAllCallback extends Extension;
interface ParameterResolver extends Extension;
```

[Extensions](./extensions.md)

### Conditional Execution

Rich set of conditions for controlling test execution based on operating system, JRE version, system properties, and custom conditions.

```java { .api }
@EnabledOnOs(OS.LINUX)
@DisabledOnJre(JRE.JAVA_8)
@EnabledIfSystemProperty(named = "env", matches = "prod")
@EnabledIf("customCondition")
```

[Conditional Execution](./conditional-execution.md)

### Dynamic Tests

Runtime test generation and nested test organization for complex test scenarios and hierarchical test structure.

```java { .api }
@TestFactory
Stream<DynamicTest> dynamicTests();

static DynamicTest dynamicTest(String displayName, Executable executable);
static DynamicContainer dynamicContainer(String displayName, Stream<DynamicNode> children);
```

[Dynamic Tests](./dynamic-tests.md)

### Parallel Execution and Resource Management

Configuration for parallel test execution, resource locking, and temporary file management for performance optimization.

```java { .api }
@Execution(ExecutionMode.CONCURRENT)
@ResourceLock("database")
@TempDir
Path tempDirectory;
```

[Parallel Execution](./parallel-execution.md)

## Types

### Core Test Interfaces

```java { .api }
interface TestInfo {
    String getDisplayName();
    Set<String> getTags();
    Optional<Class<?>> getTestClass();
    Optional<Method> getTestMethod();
}

interface TestReporter {
    void publishEntry(Map<String, String> map);
    void publishEntry(String key, String value);
}

interface RepetitionInfo {
    int getCurrentRepetition();
    int getTotalRepetitions();
}
```

### Assertion Utilities

```java { .api }
class AssertionFailureBuilder {
    static AssertionFailureBuilder assertionFailure();
    AssertionFailureBuilder message(String message);
    AssertionFailureBuilder expected(Object expected);
    AssertionFailureBuilder actual(Object actual);
    AssertionFailedError build();
}
```

### Functional Interfaces

```java { .api }
@FunctionalInterface
interface Executable {
    void execute() throws Throwable;
}

@FunctionalInterface  
interface ThrowingSupplier<T> {
    T get() throws Throwable;
}

@FunctionalInterface
interface ThrowingConsumer<T> {
    void accept(T t) throws Throwable;
}
```