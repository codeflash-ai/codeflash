# Assertions and Assumptions

Comprehensive assertion methods for verifying test conditions and assumptions for conditional test execution. JUnit Jupiter provides a rich set of assertion methods with clear failure messages and support for custom error messages.

## Imports

```java
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;
```

## Capabilities

### Basic Assertions

Core assertion methods for common verification scenarios.

```java { .api }
/**
 * Assert that two objects are equal
 */
static void assertEquals(Object expected, Object actual);
static void assertEquals(Object expected, Object actual, String message);
static void assertEquals(Object expected, Object actual, Supplier<String> messageSupplier);

/**
 * Assert that two objects are not equal
 */
static void assertNotEquals(Object unexpected, Object actual);
static void assertNotEquals(Object unexpected, Object actual, String message);
static void assertNotEquals(Object unexpected, Object actual, Supplier<String> messageSupplier);

/**
 * Assert that a condition is true
 */
static void assertTrue(boolean condition);
static void assertTrue(boolean condition, String message);
static void assertTrue(boolean condition, Supplier<String> messageSupplier);

/**
 * Assert that a condition is false
 */
static void assertFalse(boolean condition);
static void assertFalse(boolean condition, String message);
static void assertFalse(boolean condition, Supplier<String> messageSupplier);

/**
 * Assert that an object is null
 */
static void assertNull(Object actual);
static void assertNull(Object actual, String message);
static void assertNull(Object actual, Supplier<String> messageSupplier);

/**
 * Assert that an object is not null
 */
static void assertNotNull(Object actual);
static void assertNotNull(Object actual, String message);
static void assertNotNull(Object actual, Supplier<String> messageSupplier);
```

**Usage Examples:**

```java
@Test
void testBasicAssertions() {
    assertEquals(4, 2 + 2, "Simple addition should work");
    assertNotEquals(3, 2 + 2);
    assertTrue(5 > 3, "5 should be greater than 3");
    assertFalse(5 < 3);
    
    String nullString = null;
    String nonNullString = "hello";
    assertNull(nullString);
    assertNotNull(nonNullString, "String should not be null");
}
```

### Reference Assertions

Assert object identity and reference equality.

```java { .api }
/**
 * Assert that two objects refer to the same object
 */
static void assertSame(Object expected, Object actual);
static void assertSame(Object expected, Object actual, String message);
static void assertSame(Object expected, Object actual, Supplier<String> messageSupplier);

/**
 * Assert that two objects do not refer to the same object
 */
static void assertNotSame(Object unexpected, Object actual);
static void assertNotSame(Object unexpected, Object actual, String message);
static void assertNotSame(Object unexpected, Object actual, Supplier<String> messageSupplier);
```

**Usage Example:**

```java
@Test
void testReferenceAssertions() {
    String str1 = new String("hello");
    String str2 = new String("hello");
    String str3 = str1;
    
    assertEquals(str1, str2); // Content equality
    assertNotSame(str1, str2, "Different objects should not be same");
    assertSame(str1, str3, "Same reference should be same");
}
```

### Array and Collection Assertions

Specialized assertions for arrays and collections.

```java { .api }
/**
 * Assert that two arrays are equal
 */
static void assertArrayEquals(boolean[] expected, boolean[] actual);
static void assertArrayEquals(byte[] expected, byte[] actual);
static void assertArrayEquals(char[] expected, char[] actual);
static void assertArrayEquals(double[] expected, double[] actual);
static void assertArrayEquals(double[] expected, double[] actual, double delta);
static void assertArrayEquals(float[] expected, float[] actual);
static void assertArrayEquals(float[] expected, float[] actual, double delta);
static void assertArrayEquals(int[] expected, int[] actual);
static void assertArrayEquals(long[] expected, long[] actual);
static void assertArrayEquals(Object[] expected, Object[] actual);
static void assertArrayEquals(short[] expected, short[] actual);

/**
 * Assert that two iterables are equal
 */
static void assertIterableEquals(Iterable<?> expected, Iterable<?> actual);
static void assertIterableEquals(Iterable<?> expected, Iterable<?> actual, String message);
static void assertIterableEquals(Iterable<?> expected, Iterable<?> actual, Supplier<String> messageSupplier);

/**
 * Assert that lines match with pattern support
 */
static void assertLinesMatch(List<String> expectedLines, List<String> actualLines);
static void assertLinesMatch(List<String> expectedLines, List<String> actualLines, String message);
static void assertLinesMatch(List<String> expectedLines, List<String> actualLines, Supplier<String> messageSupplier);
static void assertLinesMatch(Stream<String> expectedLines, Stream<String> actualLines);
static void assertLinesMatch(Stream<String> expectedLines, Stream<String> actualLines, String message);
static void assertLinesMatch(Stream<String> expectedLines, Stream<String> actualLines, Supplier<String> messageSupplier);
```

**Usage Examples:**

```java
@Test
void testArrayAssertions() {
    int[] expected = {1, 2, 3};
    int[] actual = {1, 2, 3};
    assertArrayEquals(expected, actual);
    
    double[] expectedDoubles = {1.1, 2.2, 3.3};
    double[] actualDoubles = {1.1, 2.2, 3.3};
    assertArrayEquals(expectedDoubles, actualDoubles, 0.01);
}

@Test
void testIterableAssertions() {
    List<String> expected = Arrays.asList("a", "b", "c");
    List<String> actual = Arrays.asList("a", "b", "c");
    assertIterableEquals(expected, actual);
}

@Test
void testLinesMatch() {
    List<String> expected = Arrays.asList("Hello.*", "\\d+", "End");
    List<String> actual = Arrays.asList("Hello World", "123", "End");
    assertLinesMatch(expected, actual);
}
```

### Exception Assertions

Assert that specific exceptions are thrown or not thrown.

```java { .api }
/**
 * Assert that executable throws expected exception type
 */
static <T extends Throwable> T assertThrows(Class<T> expectedType, Executable executable);
static <T extends Throwable> T assertThrows(Class<T> expectedType, Executable executable, String message);
static <T extends Throwable> T assertThrows(Class<T> expectedType, Executable executable, Supplier<String> messageSupplier);

/**
 * Assert that executable throws exactly the expected exception type
 */
static <T extends Throwable> T assertThrowsExactly(Class<T> expectedType, Executable executable);
static <T extends Throwable> T assertThrowsExactly(Class<T> expectedType, Executable executable, String message);
static <T extends Throwable> T assertThrowsExactly(Class<T> expectedType, Executable executable, Supplier<String> messageSupplier);

/**
 * Assert that executable does not throw any exception
 */
static void assertDoesNotThrow(Executable executable);
static void assertDoesNotThrow(Executable executable, String message);
static void assertDoesNotThrow(Executable executable, Supplier<String> messageSupplier);
static <T> T assertDoesNotThrow(ThrowingSupplier<T> supplier);
static <T> T assertDoesNotThrow(ThrowingSupplier<T> supplier, String message);
static <T> T assertDoesNotThrow(ThrowingSupplier<T> supplier, Supplier<String> messageSupplier);
```

**Usage Examples:**

```java
@Test
void testExceptionAssertions() {
    // Assert specific exception is thrown
    IllegalArgumentException exception = assertThrows(
        IllegalArgumentException.class, 
        () -> { throw new IllegalArgumentException("Invalid argument"); },
        "Should throw IllegalArgumentException"
    );
    assertEquals("Invalid argument", exception.getMessage());
    
    // Assert exact exception type
    RuntimeException exactException = assertThrowsExactly(
        RuntimeException.class,
        () -> { throw new RuntimeException("Runtime error"); }
    );
    
    // Assert no exception is thrown
    assertDoesNotThrow(() -> {
        String result = "safe operation";
        return result;
    });
    
    // Assert no exception and return value
    String result = assertDoesNotThrow(() -> "safe operation");
    assertEquals("safe operation", result);
}
```

### Timeout Assertions

Assert that operations complete within specified time limits.

```java { .api }
/**
 * Assert that executable completes within timeout
 */
static void assertTimeout(Duration timeout, Executable executable);
static void assertTimeout(Duration timeout, Executable executable, String message);
static void assertTimeout(Duration timeout, Executable executable, Supplier<String> messageSupplier);
static <T> T assertTimeout(Duration timeout, ThrowingSupplier<T> supplier);
static <T> T assertTimeout(Duration timeout, ThrowingSupplier<T> supplier, String message);
static <T> T assertTimeout(Duration timeout, ThrowingSupplier<T> supplier, Supplier<String> messageSupplier);

/**
 * Assert that executable completes within timeout, preemptively aborting if it takes too long
 */
static void assertTimeoutPreemptively(Duration timeout, Executable executable);
static void assertTimeoutPreemptively(Duration timeout, Executable executable, String message);
static void assertTimeoutPreemptively(Duration timeout, Executable executable, Supplier<String> messageSupplier);
static <T> T assertTimeoutPreemptively(Duration timeout, ThrowingSupplier<T> supplier);
static <T> T assertTimeoutPreemptively(Duration timeout, ThrowingSupplier<T> supplier, String message);
static <T> T assertTimeoutPreemptively(Duration timeout, ThrowingSupplier<T> supplier, Supplier<String> messageSupplier);
```

**Usage Examples:**

```java
@Test
void testTimeoutAssertions() {
    // Assert operation completes within timeout
    assertTimeout(Duration.ofSeconds(2), () -> {
        Thread.sleep(1000); // 1 second delay
    });
    
    // Assert with return value
    String result = assertTimeout(Duration.ofSeconds(1), () -> {
        return "Quick operation";
    });
    assertEquals("Quick operation", result);
    
    // Preemptive timeout (interrupts if takes too long)
    assertTimeoutPreemptively(Duration.ofMillis(500), () -> {
        Thread.sleep(100); // Short delay
    });
}
```

### Instance Type Assertions

Assert object types and inheritance relationships.

```java { .api }
/**
 * Assert that object is instance of expected type
 */
static void assertInstanceOf(Class<?> expectedType, Object actualValue);
static void assertInstanceOf(Class<?> expectedType, Object actualValue, String message);
static void assertInstanceOf(Class<?> expectedType, Object actualValue, Supplier<String> messageSupplier);
static <T> T assertInstanceOf(Class<T> expectedType, Object actualValue);
static <T> T assertInstanceOf(Class<T> expectedType, Object actualValue, String message);
static <T> T assertInstanceOf(Class<T> expectedType, Object actualValue, Supplier<String> messageSupplier);
```

**Usage Example:**

```java
@Test
void testInstanceAssertions() {
    Object obj = "Hello World";
    
    // Simple instance check
    assertInstanceOf(String.class, obj);
    
    // With type casting
    String str = assertInstanceOf(String.class, obj, "Object should be String");
    assertEquals(11, str.length());
    
    // Check inheritance
    Number num = 42;
    assertInstanceOf(Integer.class, num);
}
```

### Grouped Assertions

Execute multiple assertions together and report all failures.

```java { .api }
/**
 * Group multiple assertions and execute all, reporting all failures
 */
static void assertAll(Executable... executables);
static void assertAll(String heading, Executable... executables);
static void assertAll(Collection<Executable> executables);
static void assertAll(String heading, Collection<Executable> executables);
static void assertAll(Stream<Executable> executables);
static void assertAll(String heading, Stream<Executable> executables);
```

**Usage Example:**

```java
@Test
void testGroupedAssertions() {
    Person person = new Person("John", "Doe", 30);
    
    assertAll("Person properties",
        () -> assertEquals("John", person.getFirstName()),
        () -> assertEquals("Doe", person.getLastName()),
        () -> assertTrue(person.getAge() > 0),
        () -> assertNotNull(person.getFullName())
    );
    
    // Using collections
    List<Executable> assertions = Arrays.asList(
        () -> assertEquals(4, 2 + 2),
        () -> assertTrue(5 > 3),
        () -> assertNotNull("test")
    );
    assertAll("Math assertions", assertions);
}
```

### Failure Methods

Explicitly fail tests with custom messages.

```java { .api }
/**
 * Explicitly fail test
 */
static void fail();
static void fail(String message);
static void fail(String message, Throwable cause);
static void fail(Throwable cause);
static void fail(Supplier<String> messageSupplier);
static <T> T fail();
static <T> T fail(String message);
static <T> T fail(String message, Throwable cause);
static <T> T fail(Throwable cause);
static <T> T fail(Supplier<String> messageSupplier);
```

**Usage Example:**

```java
@Test
void testFailMethods() {
    boolean condition = false;
    
    if (!condition) {
        fail("Condition was not met");
    }
    
    // With supplier for expensive message creation
    if (!condition) {
        fail(() -> "Complex message: " + generateComplexMessage());
    }
    
    // In switch statement
    switch (value) {
        case 1: /* handle */ break;
        case 2: /* handle */ break;
        default: fail("Unexpected value: " + value);
    }
}
```

### Assumptions

Conditional test execution based on assumptions about the test environment.

```java { .api }
/**
 * Assume that a condition is true, abort test if false
 */
static void assumeTrue(boolean assumption);
static void assumeTrue(boolean assumption, String message);
static void assumeTrue(boolean assumption, Supplier<String> messageSupplier);
static void assumeTrue(BooleanSupplier assumptionSupplier);
static void assumeTrue(BooleanSupplier assumptionSupplier, String message);
static void assumeTrue(BooleanSupplier assumptionSupplier, Supplier<String> messageSupplier);

/**
 * Assume that a condition is false, abort test if true
 */
static void assumeFalse(boolean assumption);
static void assumeFalse(boolean assumption, String message);
static void assumeFalse(boolean assumption, Supplier<String> messageSupplier);
static void assumeFalse(BooleanSupplier assumptionSupplier);
static void assumeFalse(BooleanSupplier assumptionSupplier, String message);
static void assumeFalse(BooleanSupplier assumptionSupplier, Supplier<String> messageSupplier);

/**
 * Execute test code only if assumption is true
 */
static void assumingThat(boolean assumption, Executable executable);
static void assumingThat(BooleanSupplier assumptionSupplier, Executable executable);
```

**Usage Examples:**

```java
@Test
void testOnlyOnLinux() {
    assumeTrue(System.getProperty("os.name").toLowerCase().contains("linux"));
    
    // This test will only run on Linux
    // Will be skipped (not failed) on other operating systems
    assertEquals("/", File.separator);
}

@Test
void testWithPartialAssumption() {
    // This part always runs
    assertEquals(4, 2 + 2);
    
    // This part only runs if assumption is true
    assumingThat(System.getProperty("env").equals("dev"), () -> {
        // Development-only test code
        assertEquals("localhost", getServerHost());
    });
    
    // This part always runs
    assertTrue(true);
}

@Test
void testWithEnvironmentCheck() {
    String env = System.getProperty("test.env");
    assumeFalse("prod".equals(env), "Not running destructive test in production");
    
    // Destructive test that should not run in production
    database.deleteAllData();
}
```

### Assertion Failure Builder

Build custom assertion failures with detailed information.

```java { .api }
class AssertionFailureBuilder {
    /**
     * Create new assertion failure builder
     */
    static AssertionFailureBuilder assertionFailure();
    
    /**
     * Set failure message
     */
    AssertionFailureBuilder message(String message);
    
    /**
     * Set expected value
     */
    AssertionFailureBuilder expected(Object expected);
    
    /**
     * Set actual value  
     */
    AssertionFailureBuilder actual(Object actual);
    
    /**
     * Set cause exception
     */
    AssertionFailureBuilder cause(Throwable cause);
    
    /**
     * Build the assertion failure
     */
    AssertionFailedError build();
}
```

**Usage Example:**

```java
@Test
void testCustomAssertion() {
    String expected = "hello";
    String actual = "world";
    
    if (!expected.equals(actual)) {
        throw AssertionFailureBuilder.assertionFailure()
            .message("Strings should match")
            .expected(expected)
            .actual(actual)
            .build();
    }
}
```