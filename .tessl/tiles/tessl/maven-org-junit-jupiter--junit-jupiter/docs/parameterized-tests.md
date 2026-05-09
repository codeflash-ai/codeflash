# Parameterized Tests

Advanced parameterized testing capabilities that allow running the same test logic with different sets of arguments. JUnit Jupiter provides multiple ways to supply test arguments with support for custom conversion and aggregation.

## Imports

```java
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.*;
import org.junit.jupiter.params.aggregator.*;
import org.junit.jupiter.params.converter.*;
import static org.junit.jupiter.api.Assertions.*;
```

## Capabilities

### Parameterized Test Annotation

Core annotation for defining parameterized tests.

```java { .api }
/**
 * Marks a method as a parameterized test with multiple argument sources
 */
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@interface ParameterizedTest {
    /**
     * Custom name pattern for parameterized test invocations
     */
    String name() default "[{index}] {arguments}";
    
    /**
     * How to handle argument count mismatches
     */
    ArgumentCountValidationMode argumentCountValidationMode() default ArgumentCountValidationMode.STRICT;
}

enum ArgumentCountValidationMode {
    STRICT,      // Fail if parameter count doesn't match
    LENIENT,     // Allow missing parameters (null values)
    IGNORE       // Ignore extra arguments
}
```

**Basic Usage:**

```java
@ParameterizedTest
@ValueSource(ints = {1, 2, 3})
void testWithValueSource(int argument) {
    assertTrue(argument > 0);
}

@ParameterizedTest(name = "Run {index}: testing with value {0}")
@ValueSource(strings = {"apple", "banana", "cherry"})
void testWithCustomName(String fruit) {
    assertNotNull(fruit);
    assertTrue(fruit.length() > 3);
}
```

### Value Sources

Simple argument sources for primitive types and strings.

```java { .api }
/**
 * Array of literal values as arguments
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ArgumentsSource(ValueArgumentsProvider.class)
@interface ValueSource {
    short[] shorts() default {};
    byte[] bytes() default {};
    int[] ints() default {};
    long[] longs() default {};
    float[] floats() default {};
    double[] doubles() default {};
    char[] chars() default {};
    boolean[] booleans() default {};
    String[] strings() default {};
    Class<?>[] classes() default {};
}

/**
 * Container for multiple value sources
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@interface ValueSources {
    ValueSource[] value();
}
```

**Usage Examples:**

```java
@ParameterizedTest
@ValueSource(ints = {1, 2, 3, 4, 5})
void testNumbers(int number) {
    assertTrue(number > 0 && number < 6);
}

@ParameterizedTest
@ValueSource(strings = {"", "  "})
void testBlankStrings(String input) {
    assertTrue(input.isBlank());
}

@ParameterizedTest
@ValueSource(booleans = {true, false})
void testBooleans(boolean value) {
    // Test both true and false cases
    assertNotNull(Boolean.valueOf(value));
}

@ParameterizedTest
@ValueSource(classes = {String.class, Integer.class, List.class})
void testClasses(Class<?> clazz) {
    assertNotNull(clazz);
    assertNotNull(clazz.getName());
}
```

### Null and Empty Sources

Special argument sources for null and empty values.

```java { .api }
/**
 * Provides a single null argument
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ArgumentsSource(NullArgumentsProvider.class)
@interface NullSource {
}

/**
 * Provides empty values for strings, lists, sets, maps, and primitive arrays
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ArgumentsSource(EmptyArgumentsProvider.class)
@interface EmptySource {
}

/**
 * Combines null and empty sources
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@NullSource
@EmptySource
@interface NullAndEmptySource {
}
```

**Usage Examples:**

```java
@ParameterizedTest
@NullSource
@ValueSource(strings = {"", "  ", "valid"})
void testStringValidation(String input) {
    // Test with null, empty, blank, and valid strings
    String result = StringUtils.clean(input);
    // Assert based on input type
}

@ParameterizedTest
@NullAndEmptySource
@ValueSource(strings = {"apple", "banana"})
void testStringProcessing(String input) {
    // Test null, empty, and actual values
    String processed = processString(input);
    if (input == null || input.isEmpty()) {
        assertEquals("default", processed);
    } else {
        assertNotEquals("default", processed);
    }
}

@ParameterizedTest
@EmptySource
@ValueSource(ints = {1, 2, 3})
void testIntArrays(int[] array) {
    // Test with empty array and arrays with values
    assertNotNull(array);
}
```

### Enum Sources

Arguments from enum values with filtering options.

```java { .api }
/**
 * Provides enum values as arguments
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ArgumentsSource(EnumArgumentsProvider.class)
@interface EnumSource {
    /**
     * Enum class to get values from
     */
    Class<? extends Enum<?>> value();
    
    /**
     * Enum constant names to include/exclude
     */
    String[] names() default {};
    
    /**
     * Whether to include or exclude specified names
     */
    Mode mode() default Mode.INCLUDE;
    
    enum Mode {
        INCLUDE,          // Include only specified names
        EXCLUDE,          // Exclude specified names  
        MATCH_ALL,        // Include names matching all patterns
        MATCH_ANY         // Include names matching any pattern
    }
}

/**
 * Container for multiple enum sources
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@interface EnumSources {
    EnumSource[] value();
}
```

**Usage Examples:**

```java
enum Color {
    RED, GREEN, BLUE, YELLOW, PURPLE
}

@ParameterizedTest
@EnumSource(Color.class)
void testAllColors(Color color) {
    assertNotNull(color);
    assertTrue(color.name().length() > 2);
}

@ParameterizedTest
@EnumSource(value = Color.class, names = {"RED", "BLUE"})
void testSpecificColors(Color color) {
    assertTrue(color == Color.RED || color == Color.BLUE);
}

@ParameterizedTest
@EnumSource(value = Color.class, names = {"YELLOW"}, mode = EnumSource.Mode.EXCLUDE)
void testAllColorsExceptYellow(Color color) {
    assertNotEquals(Color.YELLOW, color);
}

@ParameterizedTest
@EnumSource(value = Color.class, names = {"^B.*"}, mode = EnumSource.Mode.MATCH_ALL)
void testColorsStartingWithB(Color color) {
    assertTrue(color.name().startsWith("B"));
}
```

### CSV Sources

Arguments from CSV data, either inline or from files.

```java { .api }
/**
 * Provides CSV data as arguments
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ArgumentsSource(CsvArgumentsProvider.class)
@interface CsvSource {
    /**
     * CSV records as string array
     */
    String[] value();
    
    /**
     * Column delimiter character
     */
    char delimiter() default ',';
    
    /**
     * String to represent null values
     */
    String nullValues() default "";
    
    /**
     * Quote character for escaping
     */
    char quoteCharacter() default '"';
    
    /**
     * How to handle empty values
     */
    EmptyValue emptyValue() default EmptyValue.EMPTY_STRING;
    
    /**
     * Whether to ignore leading/trailing whitespace
     */
    boolean ignoreLeadingAndTrailingWhitespace() default true;
    
    enum EmptyValue {
        EMPTY_STRING,     // Empty string ""
        NULL_REFERENCE    // null
    }
}

/**
 * Container for multiple CSV sources
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@interface CsvSources {
    CsvSource[] value();
}
```

**Usage Examples:**

```java
@ParameterizedTest
@CsvSource({
    "apple,         1",
    "banana,        2", 
    "'lemon, lime', 3"
})
void testWithCsvSource(String fruit, int rank) {
    assertNotNull(fruit);
    assertTrue(rank > 0);
}

@ParameterizedTest
@CsvSource(value = {
    "John:25:Engineer",
    "Jane:30:Manager",
    "Bob:35:Developer"
}, delimiter = ':')
void testPersonData(String name, int age, String role) {
    assertNotNull(name);
    assertTrue(age > 0);
    assertNotNull(role);
}

@ParameterizedTest
@CsvSource(value = {
    "test,    NULL,   42",
    "example, ,      0"
}, nullValues = "NULL")
void testWithNullValues(String str, String nullableStr, int number) {
    assertNotNull(str);
    // nullableStr might be null
    assertTrue(number >= 0);
}
```

### CSV File Sources

Arguments from external CSV files.

```java { .api }
/**
 * Provides CSV data from files as arguments  
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ArgumentsSource(CsvFileArgumentsProvider.class)
@interface CsvFileSource {
    /**
     * CSV file resources (classpath relative)
     */
    String[] resources() default {};
    
    /**
     * CSV files (file system paths)
     */
    String[] files() default {};
    
    /**
     * Character encoding for files
     */
    String encoding() default "UTF-8";
    
    /**
     * Line separator for files
     */
    String lineSeparator() default "\n";
    
    /**
     * Column delimiter character
     */
    char delimiter() default ',';
    
    /**
     * String to represent null values
     */
    String nullValues() default "";
    
    /**
     * Quote character for escaping
     */
    char quoteCharacter() default '"';
    
    /**
     * How to handle empty values
     */
    CsvSource.EmptyValue emptyValue() default CsvSource.EmptyValue.EMPTY_STRING;
    
    /**
     * Whether to ignore leading/trailing whitespace
     */
    boolean ignoreLeadingAndTrailingWhitespace() default true;
    
    /**
     * Number of header lines to skip
     */
    int numLinesToSkip() default 0;
}

/**
 * Container for multiple CSV file sources
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@interface CsvFileSources {
    CsvFileSource[] value();
}
```

**Usage Examples:**

```java
@ParameterizedTest
@CsvFileSource(resources = "/test-data.csv", numLinesToSkip = 1)
void testWithCsvFileSource(String name, int age, String city) {
    assertNotNull(name);
    assertTrue(age > 0);
    assertNotNull(city);
}

@ParameterizedTest  
@CsvFileSource(files = "src/test/resources/users.csv", delimiter = ';')
void testUserData(String username, String email, boolean active) {
    assertNotNull(username);
    assertTrue(email.contains("@"));
    // active can be true or false
}
```

### Method Sources

Arguments from static methods.

```java { .api }
/**
 * Provides arguments from static methods
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ArgumentsSource(MethodArgumentsProvider.class)
@interface MethodSource {
    /**
     * Method names that provide arguments
     * If empty, uses test method name
     */
    String[] value() default {};
}

/**
 * Container for multiple method sources
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@interface MethodSources {
    MethodSource[] value();
}
```

**Usage Examples:**

```java
@ParameterizedTest
@MethodSource("stringProvider")
void testWithMethodSource(String argument) {
    assertNotNull(argument);
}

static Stream<String> stringProvider() {
    return Stream.of("apple", "banana", "cherry");
}

@ParameterizedTest
@MethodSource("personProvider")
void testPersons(Person person) {
    assertNotNull(person.getName());
    assertTrue(person.getAge() > 0);
}

static Stream<Person> personProvider() {
    return Stream.of(
        new Person("John", 25),
        new Person("Jane", 30),
        new Person("Bob", 35)
    );
}

@ParameterizedTest
@MethodSource("argumentProvider")
void testWithMultipleArguments(int number, String text, boolean flag) {
    assertTrue(number > 0);
    assertNotNull(text);
    // flag can be any boolean value
}

static Stream<Arguments> argumentProvider() {
    return Stream.of(
        Arguments.of(1, "first", true),
        Arguments.of(2, "second", false),
        Arguments.of(3, "third", true)
    );
}
```

### Field Sources

Arguments from static fields.

```java { .api }
/**
 * Provides arguments from static fields
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ArgumentsSource(FieldArgumentsProvider.class)
@interface FieldSource {
    /**
     * Field names that provide arguments
     * If empty, uses test method name
     */
    String[] value() default {};
}

/**
 * Container for multiple field sources  
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@interface FieldSources {
    FieldSource[] value();
}
```

**Usage Examples:**

```java
static List<String> fruits = Arrays.asList("apple", "banana", "cherry");

@ParameterizedTest
@FieldSource("fruits")
void testWithFieldSource(String fruit) {
    assertNotNull(fruit);
    assertTrue(fruit.length() > 3);
}

static Stream<Arguments> testData = Stream.of(
    Arguments.of(1, "one"),
    Arguments.of(2, "two"), 
    Arguments.of(3, "three")
);

@ParameterizedTest
@FieldSource("testData")
void testWithArgumentsField(int number, String word) {
    assertTrue(number > 0);
    assertNotNull(word);
}
```

### Custom Argument Sources

Create custom argument providers for complex scenarios.

```java { .api }
/**
 * Custom arguments source annotation
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ArgumentsSource(CustomArgumentsProvider.class)
@interface ArgumentsSource {
    /**
     * ArgumentsProvider implementation class
     */
    Class<? extends ArgumentsProvider> value();
}

/**
 * Container for multiple custom sources
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@interface ArgumentsSources {
    ArgumentsSource[] value();
}

/**
 * Arguments provider interface
 */
interface ArgumentsProvider {
    /**
     * Provide arguments for parameterized test
     */
    Stream<? extends Arguments> provideArguments(ExtensionContext context) throws Exception;
}

/**
 * Base class for annotation-based providers
 */
abstract class AnnotationBasedArgumentsProvider<T extends Annotation> implements ArgumentsProvider {
    /**
     * Accept annotation for configuration
     */
    protected abstract void accept(T annotation);
}
```

**Usage Example:**

```java
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@ArgumentsSource(RandomIntegerProvider.class)
@interface RandomIntegers {
    int count() default 10;
    int min() default 0;
    int max() default 100;
}

class RandomIntegerProvider extends AnnotationBasedArgumentsProvider<RandomIntegers> {
    private int count;
    private int min; 
    private int max;

    @Override
    protected void accept(RandomIntegers annotation) {
        this.count = annotation.count();
        this.min = annotation.min();
        this.max = annotation.max();
    }

    @Override
    public Stream<Arguments> provideArguments(ExtensionContext context) {
        Random random = new Random();
        return random.ints(count, min, max)
                    .mapToObj(Arguments::of);
    }
}

@ParameterizedTest
@RandomIntegers(count = 5, min = 1, max = 10)
void testWithRandomIntegers(int value) {
    assertTrue(value >= 1 && value <= 10);
}
```

### Argument Conversion

Convert string arguments to other types automatically or with custom converters.

```java { .api }
/**
 * Custom argument converter annotation
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.PARAMETER})
@Retention(RetentionPolicy.RUNTIME)
@interface ConvertWith {
    /**
     * ArgumentConverter implementation class
     */
    Class<? extends ArgumentConverter> value();
}

/**
 * Java time conversion pattern
 */
@Target({ElementType.ANNOTATION_TYPE, ElementType.PARAMETER})
@Retention(RetentionPolicy.RUNTIME)
@ConvertWith(JavaTimeArgumentConverter.class)
@interface JavaTimeConversionPattern {
    /**
     * Pattern for parsing date/time
     */
    String value();
}

/**
 * Argument converter interface
 */
interface ArgumentConverter<S, T> {
    /**
     * Convert source argument to target type
     */
    T convert(S source, ParameterContext context) throws ArgumentConversionException;
}

/**
 * Simple converter for single argument types
 */
abstract class SimpleArgumentConverter<S, T> implements ArgumentConverter<S, T> {
    @Override
    public final T convert(S source, ParameterContext context) throws ArgumentConversionException {
        return convert(source, context.getParameter().getType());
    }
    
    /**
     * Convert source to target type
     */
    protected abstract T convert(S source, Class<?> targetType) throws ArgumentConversionException;
}

/**
 * Typed converter with type safety
 */
abstract class TypedArgumentConverter<S, T> extends SimpleArgumentConverter<S, T> {
    private final Class<S> sourceType;
    private final Class<T> targetType;
    
    protected TypedArgumentConverter(Class<S> sourceType, Class<T> targetType) {
        this.sourceType = sourceType;
        this.targetType = targetType; 
    }
}
```

**Usage Examples:**

```java
@ParameterizedTest
@ValueSource(strings = {"2023-01-01", "2023-12-31"})
void testDates(@JavaTimeConversionPattern("yyyy-MM-dd") LocalDate date) {
    assertNotNull(date);
    assertEquals(2023, date.getYear());
}

class StringToPersonConverter extends TypedArgumentConverter<String, Person> {
    protected StringToPersonConverter() {
        super(String.class, Person.class);
    }

    @Override
    protected Person convert(String source, Class<?> targetType) {
        String[] parts = source.split(",");
        return new Person(parts[0], Integer.parseInt(parts[1]));
    }
}

@ParameterizedTest  
@ValueSource(strings = {"John,25", "Jane,30", "Bob,35"})
void testPersonConversion(@ConvertWith(StringToPersonConverter.class) Person person) {
    assertNotNull(person.getName());
    assertTrue(person.getAge() > 0);
}
```

### Argument Aggregation

Aggregate multiple arguments into complex objects.

```java { .api }
/**
 * Custom argument aggregator annotation
 */
@Target(ElementType.PARAMETER)
@Retention(RetentionPolicy.RUNTIME)
@interface AggregateWith {
    /**
     * ArgumentsAggregator implementation class
     */
    Class<? extends ArgumentsAggregator> value();
}

/**
 * Arguments aggregator interface
 */
interface ArgumentsAggregator {
    /**
     * Aggregate arguments into single object
     */
    Object aggregateArguments(ArgumentsAccessor accessor, ParameterContext context) 
        throws ArgumentsAggregationException;
}

/**
 * Arguments accessor for retrieving individual arguments
 */
interface ArgumentsAccessor {
    Object get(int index);
    <T> T get(int index, Class<T> requiredType);
    Character getCharacter(int index);
    Boolean getBoolean(int index);
    Byte getByte(int index);
    Short getShort(int index);
    Integer getInteger(int index);
    Long getLong(int index);
    Float getFloat(int index);
    Double getDouble(int index);
    String getString(int index);
    int size();
    Object[] toArray();
    List<Object> toList();
}
```

**Usage Examples:**

```java
class PersonAggregator implements ArgumentsAggregator {
    @Override
    public Object aggregateArguments(ArgumentsAccessor accessor, ParameterContext context) {
        return new Person(accessor.getString(0), accessor.getInteger(1));
    }
}

@ParameterizedTest
@CsvSource({
    "John, 25",
    "Jane, 30", 
    "Bob, 35"
})
void testPersonAggregation(@AggregateWith(PersonAggregator.class) Person person) {
    assertNotNull(person.getName());
    assertTrue(person.getAge() > 0);
}

@ParameterizedTest
@CsvSource({
    "John, 25, Engineer", 
    "Jane, 30, Manager",
    "Bob, 35, Developer"
})
void testWithArgumentsAccessor(ArgumentsAccessor arguments) {
    String name = arguments.getString(0);
    int age = arguments.getInteger(1);
    String role = arguments.getString(2);
    
    Person person = new Person(name, age, role);
    assertNotNull(person);
}
```

### Arguments Utility

Utility class for creating argument sets programmatically.

```java { .api }
/**
 * Factory for creating Arguments instances
 */
interface Arguments {
    /**
     * Create Arguments from array of objects
     */
    static Arguments of(Object... arguments);
    
    /**
     * Get arguments as object array
     */
    Object[] get();
}
```

**Usage Example:**

```java
static Stream<Arguments> complexArgumentProvider() {
    return Stream.of(
        Arguments.of(1, "apple", true, new Person("John", 25)),
        Arguments.of(2, "banana", false, new Person("Jane", 30)),
        Arguments.of(3, "cherry", true, new Person("Bob", 35))
    );
}

@ParameterizedTest
@MethodSource("complexArgumentProvider")
void testComplexArguments(int id, String fruit, boolean active, Person person) {
    assertTrue(id > 0);
    assertNotNull(fruit);
    assertNotNull(person);
}
```