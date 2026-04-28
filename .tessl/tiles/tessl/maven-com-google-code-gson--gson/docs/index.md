# Gson

Gson is a Java library that can be used to convert Java Objects into their JSON representation. It can also be used to convert a JSON string to an equivalent Java object. Gson can work with arbitrary Java objects including pre-existing objects that you do not have source-code access to.

## Package Information

- **Package Name**: com.google.code.gson:gson
- **Package Type**: Maven
- **Language**: Java
- **Installation**: 
  ```xml
  <dependency>
    <groupId>com.google.code.gson</groupId>
    <artifactId>gson</artifactId>
    <version>2.13.1</version>
  </dependency>
  ```

## Core Imports

```java
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
```

For JSON tree model:
```java
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonArray;
import com.google.gson.JsonPrimitive;
```

For streaming:
```java
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
```

## Basic Usage

```java
import com.google.gson.Gson;

// Simple object serialization
class Person {
    private String name;
    private int age;
    
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
}

// Create Gson instance
Gson gson = new Gson();

// Convert Java object to JSON
Person person = new Person("John", 30);
String json = gson.toJson(person);
// Result: {"name":"John","age":30}

// Convert JSON to Java object
String jsonString = "{\"name\":\"Jane\",\"age\":25}";
Person deserializedPerson = gson.fromJson(jsonString, Person.class);
```

## Architecture

Gson is built around several key components:

- **Gson Class**: Main entry point providing `toJson()` and `fromJson()` methods
- **GsonBuilder**: Builder pattern for configuring Gson instances with custom settings
- **JSON Tree Model**: Object representation (JsonElement hierarchy) for manipulating JSON as trees
- **Streaming API**: Memory-efficient reading/writing for large JSON documents
- **Type Adapters**: Pluggable serialization/deserialization for custom types
- **Reflection System**: Automatic object mapping using Java reflection

## Capabilities

### Object Serialization and Deserialization

Core functionality for converting between Java objects and JSON strings. Handles arbitrarily complex objects with deep inheritance hierarchies and generic types.

```java { .api }
public String toJson(Object src);
public String toJson(Object src, Type typeOfSrc);
public <T> T fromJson(String json, Class<T> classOfT);
public <T> T fromJson(String json, Type typeOfT);
public <T> T fromJson(String json, TypeToken<T> typeOfT);
```

[Object Serialization](./object-serialization.md)

### JSON Tree Model

Tree-based API for building and manipulating JSON structures programmatically. Useful when you need to modify JSON before serialization or after deserialization.

```java { .api }
public JsonElement toJsonTree(Object src);
public <T> T fromJson(JsonElement json, Class<T> classOfT);

abstract class JsonElement {
    public boolean isJsonObject();
    public boolean isJsonArray();
    public boolean isJsonPrimitive();
    public boolean isJsonNull();
}
```

[JSON Tree Model](./json-tree-model.md)

### Configuration and Customization

Extensive configuration options for controlling serialization behavior, field naming, date formatting, and custom type adapters.

```java { .api }
public final class GsonBuilder {
    public GsonBuilder setPrettyPrinting();
    public GsonBuilder serializeNulls();
    public GsonBuilder setFieldNamingPolicy(FieldNamingPolicy namingConvention);
    public GsonBuilder registerTypeAdapter(Type type, Object typeAdapter);
    public Gson create();
}
```

[Configuration](./configuration.md)

### Streaming API

Memory-efficient streaming API for processing large JSON documents without loading them entirely into memory.

```java { .api }
public class JsonReader implements Closeable {
    public JsonToken peek() throws IOException;
    public String nextString() throws IOException;
    public int nextInt() throws IOException;
    public void beginObject() throws IOException;
    public void endObject() throws IOException;
}

public class JsonWriter implements Closeable, Flushable {
    public JsonWriter name(String name) throws IOException;
    public JsonWriter value(String value) throws IOException;
    public JsonWriter beginObject() throws IOException;
    public JsonWriter endObject() throws IOException;
}
```

[Streaming API](./streaming-api.md)

### Type Adapters

Pluggable system for custom serialization and deserialization logic. Allows fine-grained control over how specific types are converted.

```java { .api }
public abstract class TypeAdapter<T> {
    public abstract void write(JsonWriter out, T value) throws IOException;
    public abstract T read(JsonReader in) throws IOException;
}

public interface TypeAdapterFactory {
    public <T> TypeAdapter<T> create(Gson gson, TypeToken<T> type);
}
```

[Type Adapters](./type-adapters.md)

### Annotations

Annotations for controlling serialization behavior on fields and classes without requiring configuration changes.

```java { .api }
@SerializedName("custom_name")
@Expose(serialize = true, deserialize = true)
@Since(1.0)
@Until(2.0)
@JsonAdapter(CustomAdapter.class)
```

[Annotations](./annotations.md)

## Types

```java { .api }
// Core exception types
class JsonParseException extends RuntimeException {}
class JsonSyntaxException extends JsonParseException {}
class JsonIOException extends JsonParseException {}

// Enums for configuration
enum FieldNamingPolicy {
    IDENTITY, UPPER_CAMEL_CASE, UPPER_CAMEL_CASE_WITH_SPACES,
    UPPER_CASE_WITH_UNDERSCORES, LOWER_CASE_WITH_UNDERSCORES,
    LOWER_CASE_WITH_DASHES, LOWER_CASE_WITH_DOTS
}

enum LongSerializationPolicy { DEFAULT, STRING }

enum ToNumberPolicy implements ToNumberStrategy {
    DOUBLE, LAZILY_PARSED_NUMBER, LONG_OR_DOUBLE, BIG_DECIMAL
}

// Type handling
class TypeToken<T> {
    public static <T> TypeToken<T> get(Class<T> type);
    public static TypeToken<?> get(Type type);
    public Type getType();
}
```