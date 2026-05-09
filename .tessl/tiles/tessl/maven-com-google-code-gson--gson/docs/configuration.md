# Configuration

Gson provides extensive configuration options through the GsonBuilder class for customizing serialization and deserialization behavior.

## GsonBuilder

Builder pattern class for configuring Gson instances.

```java { .api }
public final class GsonBuilder {
    public GsonBuilder();
    
    // JSON formatting
    public GsonBuilder setPrettyPrinting();
    public GsonBuilder setFormattingStyle(FormattingStyle formattingStyle);
    public GsonBuilder disableHtmlEscaping();
    
    // Null handling
    public GsonBuilder serializeNulls();
    
    // Field naming
    public GsonBuilder setFieldNamingPolicy(FieldNamingPolicy namingConvention);
    public GsonBuilder setFieldNamingStrategy(FieldNamingStrategy fieldNamingStrategy);
    
    // Version control
    public GsonBuilder setVersion(double version);
    
    // Field exposure
    public GsonBuilder excludeFieldsWithoutExposeAnnotation();
    public GsonBuilder excludeFieldsWithModifiers(int... modifiers);
    
    // Exclusion strategies
    public GsonBuilder setExclusionStrategies(ExclusionStrategy... strategies);
    public GsonBuilder addSerializationExclusionStrategy(ExclusionStrategy strategy);
    public GsonBuilder addDeserializationExclusionStrategy(ExclusionStrategy strategy);
    
    // Type adapters
    public GsonBuilder registerTypeAdapter(Type type, Object typeAdapter);
    public GsonBuilder registerTypeAdapterFactory(TypeAdapterFactory factory);
    public GsonBuilder registerTypeHierarchyAdapter(Class<?> baseType, Object typeAdapter);
    
    // Number handling
    public GsonBuilder setLongSerializationPolicy(LongSerializationPolicy serializationPolicy);
    public GsonBuilder setObjectToNumberStrategy(ToNumberStrategy objectToNumberStrategy);
    public GsonBuilder setNumberToNumberStrategy(ToNumberStrategy numberToNumberStrategy);
    public GsonBuilder serializeSpecialFloatingPointValues();
    
    // Date formatting
    public GsonBuilder setDateFormat(String pattern);
    public GsonBuilder setDateFormat(int dateStyle);
    public GsonBuilder setDateFormat(int dateStyle, int timeStyle);
    
    // JSON parsing strictness
    public GsonBuilder setLenient();
    public GsonBuilder setStrictness(Strictness strictness);
    
    // Advanced options
    public GsonBuilder enableComplexMapKeySerialization();
    public GsonBuilder disableInnerClassSerialization();
    public GsonBuilder generateNonExecutableJson();
    public GsonBuilder disableJdkUnsafe();
    public GsonBuilder addReflectionAccessFilter(ReflectionAccessFilter filter);
    
    // Build final instance
    public Gson create();
}
```

## Basic Configuration

**Creating a configured Gson instance:**
```java
Gson gson = new GsonBuilder()
    .setPrettyPrinting()
    .serializeNulls()
    .create();
```

## Field Naming Policies

Control how Java field names are converted to JSON property names.

```java { .api }
public enum FieldNamingPolicy implements FieldNamingStrategy {
    IDENTITY,
    UPPER_CAMEL_CASE,
    UPPER_CAMEL_CASE_WITH_SPACES,
    UPPER_CASE_WITH_UNDERSCORES,
    LOWER_CASE_WITH_UNDERSCORES,
    LOWER_CASE_WITH_DASHES,
    LOWER_CASE_WITH_DOTS
}
```

**Usage:**
```java
Gson gson = new GsonBuilder()
    .setFieldNamingPolicy(FieldNamingPolicy.LOWER_CASE_WITH_UNDERSCORES)
    .create();

class Person {
    String firstName; // becomes "first_name" in JSON
    String lastName;  // becomes "last_name" in JSON
}
```

## Custom Field Naming Strategy

```java { .api }
public interface FieldNamingStrategy {
    public String translateName(Field f);
}
```

**Usage:**
```java
FieldNamingStrategy customStrategy = field -> {
    return "prefix_" + field.getName().toLowerCase();
};

Gson gson = new GsonBuilder()
    .setFieldNamingStrategy(customStrategy)
    .create();
```

## Exclusion Strategies

Control which fields and classes are included in serialization/deserialization.

```java { .api }
public interface ExclusionStrategy {
    public boolean shouldSkipField(FieldAttributes f);
    public boolean shouldSkipClass(Class<?> clazz);
}
```

**Usage:**
```java
ExclusionStrategy strategy = new ExclusionStrategy() {
    @Override
    public boolean shouldSkipField(FieldAttributes f) {
        return f.getName().startsWith("internal");
    }
    
    @Override
    public boolean shouldSkipClass(Class<?> clazz) {
        return clazz.getName().contains("Internal");
    }
};

Gson gson = new GsonBuilder()
    .setExclusionStrategies(strategy)
    .create();
```

## Field Attributes

Information about fields for exclusion strategies.

```java { .api }
public final class FieldAttributes {
    public Class<?> getDeclaringClass();
    public String getName();
    public Type getDeclaredType();
    public Class<?> getDeclaredClass();
    public <T extends Annotation> T getAnnotation(Class<T> annotation);
    public Collection<Annotation> getAnnotations();
    public boolean hasModifier(int modifier);
}
```

## Number Handling

### Long Serialization Policy

```java { .api }
public enum LongSerializationPolicy {
    DEFAULT,  // Serialize as numbers
    STRING    // Serialize as strings
}
```

**Usage:**
```java
Gson gson = new GsonBuilder()
    .setLongSerializationPolicy(LongSerializationPolicy.STRING)
    .create();
```

### Number Strategies

```java { .api }
public interface ToNumberStrategy {
    public Number readNumber(JsonReader in) throws IOException;
}

public enum ToNumberPolicy implements ToNumberStrategy {
    DOUBLE,
    LAZILY_PARSED_NUMBER,
    LONG_OR_DOUBLE,
    BIG_DECIMAL
}
```

**Usage:**
```java
Gson gson = new GsonBuilder()
    .setObjectToNumberStrategy(ToNumberPolicy.BIG_DECIMAL)
    .setNumberToNumberStrategy(ToNumberPolicy.BIG_DECIMAL)
    .create();
```

## Date Formatting

**Pattern-based formatting:**
```java
Gson gson = new GsonBuilder()
    .setDateFormat("yyyy-MM-dd HH:mm:ss")
    .create();
```

**Style-based formatting:**
```java
import java.text.DateFormat;

Gson gson = new GsonBuilder()
    .setDateFormat(DateFormat.LONG)
    .create();

// Or with both date and time styles
Gson gson = new GsonBuilder()
    .setDateFormat(DateFormat.MEDIUM, DateFormat.SHORT)
    .create();
```

## JSON Formatting

### Pretty Printing

```java
Gson gson = new GsonBuilder()
    .setPrettyPrinting()
    .create();
```

### Custom Formatting Style

```java { .api }
public class FormattingStyle {
    public static FormattingStyle COMPACT;
    public static FormattingStyle PRETTY;
    
    public FormattingStyle withNewline(String newline);
    public FormattingStyle withIndent(String indent);
    public FormattingStyle withSpaceAfterSeparators(boolean spaceAfterSeparators);
}
```

**Usage:**
```java
FormattingStyle style = FormattingStyle.PRETTY
    .withIndent("    ")  // 4 spaces
    .withNewline("\n");

Gson gson = new GsonBuilder()
    .setFormattingStyle(style)
    .create();
```

## Strictness Control

```java { .api }
public enum Strictness {
    LENIENT,  // Allows malformed JSON
    STRICT    // Requires well-formed JSON
}
```

**Usage:**
```java
// Lenient parsing (allows single quotes, unquoted names, etc.)
Gson gson = new GsonBuilder()
    .setLenient()
    .create();

// Or explicitly set strictness
Gson gson = new GsonBuilder()
    .setStrictness(Strictness.LENIENT)
    .create();
```

## Advanced Configuration

### Complex Map Keys

Enable serialization of complex objects as map keys:
```java
Gson gson = new GsonBuilder()
    .enableComplexMapKeySerialization()
    .create();
```

### HTML Escaping

Disable HTML character escaping:
```java
Gson gson = new GsonBuilder()
    .disableHtmlEscaping()
    .create();
```

### Reflection Access Control

```java { .api }
public interface ReflectionAccessFilter {
    enum FilterResult { ALLOW, BLOCK_INACCESSIBLE, BLOCK_ALL }
    
    FilterResult check(Class<?> rawClass);
}
```

**Usage:**
```java
ReflectionAccessFilter filter = new ReflectionAccessFilter() {
    @Override
    public FilterResult check(Class<?> rawClass) {
        if (rawClass.getPackage().getName().startsWith("com.example.internal")) {
            return FilterResult.BLOCK_ALL;
        }
        return FilterResult.ALLOW;
    }
};

Gson gson = new GsonBuilder()
    .addReflectionAccessFilter(filter)
    .create();
```

### Version Control

Use `@Since` and `@Until` annotations with version setting:
```java
Gson gson = new GsonBuilder()
    .setVersion(2.0)
    .create();
```

### Modifier-based Exclusion

Exclude fields based on Java modifiers:
```java
import java.lang.reflect.Modifier;

Gson gson = new GsonBuilder()
    .excludeFieldsWithModifiers(Modifier.STATIC, Modifier.TRANSIENT)
    .create();
```