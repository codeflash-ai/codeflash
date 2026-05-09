# Core Testing API

Essential testing annotations and lifecycle methods that form the foundation of JUnit Jupiter tests. These provide the basic structure for organizing and executing tests with modern Java features.

## Imports

```java
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;
```

## Capabilities

### Test Annotation

Marks a method as a test method that should be executed by the test engine.

```java { .api }
/**
 * Marks a method as a test method
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@interface Test {
}
```

**Usage Example:**

```java
class MyTest {
    @Test
    void shouldCalculateCorrectly() {
        // Test implementation
        assertEquals(4, 2 + 2);
    }
}
```

### Lifecycle Annotations

Control test execution lifecycle with setup and teardown methods.

```java { .api }
/**
 * Executed once before all test methods in the class
 * Method must be static
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@interface BeforeAll {
}

/**
 * Executed before each individual test method
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@interface BeforeEach {
}

/**
 * Executed after each individual test method
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@interface AfterEach {
}

/**
 * Executed once after all test methods in the class
 * Method must be static
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@interface AfterAll {
}
```

**Usage Example:**

```java
class DatabaseTest {
    static Database database;
    Connection connection;

    @BeforeAll
    static void initDatabase() {
        database = new Database();
        database.start();
    }

    @AfterAll
    static void cleanupDatabase() {
        database.stop();
    }

    @BeforeEach
    void openConnection() {
        connection = database.openConnection();
    }

    @AfterEach
    void closeConnection() {
        if (connection != null) {
            connection.close();
        }
    }

    @Test
    void testQuery() {
        // Test with connection
    }
}
```

### Display Names

Customize test names for better readability in test reports.

```java { .api }
/**
 * Custom display name for tests and test classes
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@interface DisplayName {
    String value();
}

/**
 * Generate display names using a specific strategy
 */
@Target({ElementType.TYPE})
@Retention(RetentionPolicy.RUNTIME)
@interface DisplayNameGeneration {
    Class<? extends DisplayNameGenerator> value();
}
```

**Usage Example:**

```java
@DisplayName("Calculator Tests")
class CalculatorTest {

    @Test
    @DisplayName("Addition should work for positive numbers")
    void testAddition() {
        assertEquals(5, 2 + 3);
    }

    @Test
    @DisplayName("Division by zero should throw exception")
    void testDivisionByZero() {
        assertThrows(ArithmeticException.class, () -> 10 / 0);
    }
}
```

### Display Name Generators

Built-in strategies for generating display names automatically.

```java { .api }
interface DisplayNameGenerator {
    String generateDisplayNameForClass(Class<?> testClass);
    String generateDisplayNameForNestedClass(Class<?> nestedClass);
    String generateDisplayNameForMethod(Class<?> testClass, Method testMethod);

    class Standard implements DisplayNameGenerator { }
    class Simple implements DisplayNameGenerator { }
    class ReplaceUnderscores implements DisplayNameGenerator { }
    class IndicativeSentences implements DisplayNameGenerator { }
}
```

### Nested Tests

Organize related tests in hierarchical structure using nested classes.

```java { .api }
/**
 * Marks a nested class as a test class
 */
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@interface Nested {
}
```

**Usage Example:**

```java
class AccountTest {

    @Test
    void testCreateAccount() {
        // Test account creation
    }

    @Nested
    @DisplayName("When account has balance")
    class WhenAccountHasBalance {

        Account account;

        @BeforeEach
        void createAccountWithBalance() {
            account = new Account(100);
        }

        @Test
        @DisplayName("withdraw should decrease balance")
        void withdrawShouldDecreaseBalance() {
            account.withdraw(20);
            assertEquals(80, account.getBalance());
        }

        @Nested
        @DisplayName("And withdrawal amount exceeds balance")
        class AndWithdrawalExceedsBalance {

            @Test
            @DisplayName("should throw InsufficientFundsException")
            void shouldThrowException() {
                assertThrows(InsufficientFundsException.class, 
                    () -> account.withdraw(150));
            }
        }
    }
}
```

### Test Disabling

Disable tests temporarily or conditionally.

```java { .api }
/**
 * Disable test execution with optional reason
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@interface Disabled {
    String value() default "";
}
```

**Usage Example:**

```java
class FeatureTest {

    @Test
    @Disabled("Feature not yet implemented")
    void testNewFeature() {
        // This test won't run
    }

    @Test
    @Disabled
    void temporarilyDisabled() {
        // This test won't run either
    }
}
```

### Test Tagging

Tag tests for filtering and selective execution.

```java { .api }
/**
 * Tag a test for filtering
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@Repeatable(Tags.class)
@interface Tag {
    String value();
}

/**
 * Container for multiple tags
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@interface Tags {
    Tag[] value();
}
```

**Usage Example:**

```java
class IntegrationTest {

    @Test
    @Tag("fast")
    void testQuickOperation() {
        // Fast test
    }

    @Test
    @Tag("slow")
    @Tag("integration")
    void testDatabaseIntegration() {
        // Slow integration test
    }

    @Test
    @Tags({@Tag("smoke"), @Tag("critical")})
    void testCriticalPath() {
        // Critical smoke test
    }
}
```

### Repeated Tests

Execute the same test multiple times with repetition information.

```java { .api }
/**
 * Repeat test execution multiple times
 */
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@interface RepeatedTest {
    int value();
    String name() default "";
}
```

**Usage Example:**

```java
class RandomTest {

    @RepeatedTest(10)
    void testRandomBehavior() {
        int random = (int) (Math.random() * 100);
        assertTrue(random >= 0 && random < 100);
    }

    @RepeatedTest(value = 5, name = "Run {currentRepetition} of {totalRepetitions}")
    void testWithCustomName(RepetitionInfo repetitionInfo) {
        System.out.println("Repetition: " + repetitionInfo.getCurrentRepetition());
    }
}
```

### Test Information

Access test metadata at runtime.

```java { .api }
interface TestInfo {
    String getDisplayName();
    Set<String> getTags();
    Optional<Class<?>> getTestClass();
    Optional<Method> getTestMethod();
}

interface RepetitionInfo {
    int getCurrentRepetition();
    int getTotalRepetitions();
}
```

**Usage Example:**

```java
class InfoTest {

    @Test
    void testWithInfo(TestInfo testInfo) {
        System.out.println("Test name: " + testInfo.getDisplayName());
        System.out.println("Tags: " + testInfo.getTags());
    }

    @RepeatedTest(3)
    void repeatedTestWithInfo(RepetitionInfo repetitionInfo) {
        System.out.println("Repetition " + repetitionInfo.getCurrentRepetition() 
            + " of " + repetitionInfo.getTotalRepetitions());
    }
}
```

### Test Ordering

Control the order of test execution.

```java { .api }
/**
 * Configure test method execution order
 */
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@interface TestMethodOrder {
    Class<? extends MethodOrderer> value();
}

/**
 * Configure test class execution order
 */
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@interface TestClassOrder {
    Class<? extends ClassOrderer> value();
}

/**
 * Specify execution order
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@interface Order {
    int value();
}
```

**Usage Example:**

```java
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class OrderedTest {

    @Test
    @Order(3)
    void testThird() {
        // Runs third
    }

    @Test
    @Order(1)
    void testFirst() {
        // Runs first
    }

    @Test
    @Order(2)
    void testSecond() {
        // Runs second
    }
}
```

### Test Instance Lifecycle

Control how test instances are created and managed.

```java { .api }
/**
 * Configure test instance lifecycle
 */
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@interface TestInstance {
    Lifecycle value();

    enum Lifecycle {
        PER_METHOD,  // Default: new instance per test method
        PER_CLASS    // One instance per test class
    }
}
```

**Usage Example:**

```java
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class SharedStateTest {
    int counter = 0;

    @Test
    void firstTest() {
        counter++;
        assertEquals(1, counter);
    }

    @Test
    void secondTest() {
        counter++;
        assertEquals(2, counter); // Works because same instance
    }
}
```

### Test Timeouts

Configure execution timeouts for individual tests or entire test classes.

```java { .api }
/**
 * Configure test execution timeout
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@interface Timeout {
    /**
     * Timeout value
     */
    long value();
    
    /**
     * Time unit for timeout value
     */
    TimeUnit unit() default TimeUnit.SECONDS;
    
    /**
     * Thread mode for timeout enforcement
     */
    ThreadMode threadMode() default ThreadMode.SAME_THREAD;
    
    enum ThreadMode {
        /**
         * Execute in same thread with timeout monitoring
         */
        SAME_THREAD,
        
        /**
         * Execute in separate thread and interrupt on timeout
         */
        SEPARATE_THREAD
    }
}
```

**Usage Examples:**

```java
class TimeoutTest {
    
    @Test
    @Timeout(5) // 5 seconds
    void testWithTimeout() throws InterruptedException {
        Thread.sleep(1000); // Will pass
    }
    
    @Test
    @Timeout(value = 500, unit = TimeUnit.MILLISECONDS)
    void testWithMillisecondTimeout() {
        // Test must complete within 500ms
        performQuickOperation();
    }
    
    @Test
    @Timeout(value = 10, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testWithSeparateThread() throws InterruptedException {
        // Will be interrupted after 10 seconds if still running
        Thread.sleep(5000);
    }
}

@Timeout(30) // Default 30 second timeout for all tests in class
class SlowTestsWithTimeout {
    
    @Test
    void slowTest1() throws InterruptedException {
        Thread.sleep(10000); // 10 seconds - within class timeout
    }
    
    @Test
    @Timeout(60) // Override class timeout for this test
    void verySlowTest() throws InterruptedException {
        Thread.sleep(45000); // 45 seconds - within method timeout
    }
}
```