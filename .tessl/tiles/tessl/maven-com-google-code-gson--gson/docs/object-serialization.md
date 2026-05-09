# Object Serialization

The core functionality of Gson for converting between Java objects and JSON strings.

## Serialization (Java to JSON)

### Basic Serialization

```java { .api }
public String toJson(Object src);
```

Converts any Java object to its JSON string representation using default settings.

**Usage:**
```java
Gson gson = new Gson();
Person person = new Person("Alice", 30);
String json = gson.toJson(person);
// Result: {"name":"Alice","age":30}
```

### Serialization with Type Information

```java { .api }
public String toJson(Object src, Type typeOfSrc);
```

Explicitly specifies the type for serialization, useful for generic collections and inheritance hierarchies.

**Usage:**
```java
List<String> names = Arrays.asList("Alice", "Bob", "Charlie");
Type listType = new TypeToken<List<String>>(){}.getType();
String json = gson.toJson(names, listType);
// Result: ["Alice","Bob","Charlie"]
```

### Serialization to Writer

```java { .api }
public void toJson(Object src, Appendable writer) throws JsonIOException;
public void toJson(Object src, Type typeOfSrc, Appendable writer) throws JsonIOException;
public void toJson(Object src, Type typeOfSrc, JsonWriter writer) throws JsonIOException;
```

Serialize directly to a Writer or JsonWriter for memory efficiency or streaming output.

**Usage:**
```java
StringWriter writer = new StringWriter();
gson.toJson(person, writer);
String json = writer.toString();
```

## Deserialization (JSON to Java)

### Basic Deserialization

```java { .api }
public <T> T fromJson(String json, Class<T> classOfT) throws JsonSyntaxException;
```

Converts JSON string to Java object of the specified class.

**Usage:**
```java
String json = "{\"name\":\"Bob\",\"age\":25}";
Person person = gson.fromJson(json, Person.class);
```

### Deserialization with Generic Types

```java { .api }
public <T> T fromJson(String json, Type typeOfT) throws JsonSyntaxException;
public <T> T fromJson(String json, TypeToken<T> typeOfT) throws JsonSyntaxException;
```

Handles generic types like collections and maps using Type or TypeToken.

**Usage:**
```java
String json = "[\"Alice\",\"Bob\",\"Charlie\"]";
Type listType = new TypeToken<List<String>>(){}.getType();
List<String> names = gson.fromJson(json, listType);

// Or using TypeToken directly
TypeToken<List<String>> token = new TypeToken<List<String>>(){};
List<String> names = gson.fromJson(json, token);
```

### Deserialization from Reader

```java { .api }
public <T> T fromJson(Reader json, Class<T> classOfT) throws JsonIOException, JsonSyntaxException;
public <T> T fromJson(Reader json, Type typeOfT) throws JsonIOException, JsonSyntaxException;
public <T> T fromJson(Reader json, TypeToken<T> typeOfT) throws JsonIOException, JsonSyntaxException;
```

Deserialize from a Reader for memory-efficient processing of large files.

**Usage:**
```java
FileReader reader = new FileReader("data.json");
Person person = gson.fromJson(reader, Person.class);
reader.close();
```

### Deserialization from JsonReader

```java { .api }
public <T> T fromJson(JsonReader reader, Type typeOfT) throws JsonIOException, JsonSyntaxException;
public <T> T fromJson(JsonReader reader, TypeToken<T> typeOfT) throws JsonIOException, JsonSyntaxException;
```

Deserialize from JsonReader for streaming API integration.

**Usage:**
```java
JsonReader reader = new JsonReader(new StringReader(json));
Person person = gson.fromJson(reader, Person.class);
reader.close();
```

## Complex Object Handling

### Collections

Gson automatically handles standard Java collections:

```java
// Lists
List<String> list = Arrays.asList("a", "b", "c");
String json = gson.toJson(list);
List<String> restored = gson.fromJson(json, new TypeToken<List<String>>(){}.getType());

// Maps
Map<String, Integer> map = new HashMap<>();
map.put("one", 1);
map.put("two", 2);
String json = gson.toJson(map);
Map<String, Integer> restored = gson.fromJson(json, new TypeToken<Map<String, Integer>>(){}.getType());
```

### Arrays

```java
int[] numbers = {1, 2, 3, 4, 5};
String json = gson.toJson(numbers);
int[] restored = gson.fromJson(json, int[].class);
```

### Nested Objects

```java
class Address {
    String street;
    String city;
}

class Person {
    String name;
    Address address;
}

Person person = new Person();
person.name = "John";
person.address = new Address();
person.address.street = "123 Main St";
person.address.city = "Springfield";

String json = gson.toJson(person);
Person restored = gson.fromJson(json, Person.class);
```

### Inheritance

```java
class Animal {
    String name;
}

class Dog extends Animal {
    String breed;
}

Dog dog = new Dog();
dog.name = "Buddy";
dog.breed = "Golden Retriever";

// Serialize as Dog type
String json = gson.toJson(dog, Dog.class);
Dog restored = gson.fromJson(json, Dog.class);
```

## Error Handling

```java { .api }
class JsonSyntaxException extends JsonParseException {
    // Thrown when JSON syntax is invalid
}

class JsonIOException extends JsonParseException {
    // Thrown when I/O errors occur during reading/writing
}
```

**Usage:**
```java
try {
    Person person = gson.fromJson(invalidJson, Person.class);
} catch (JsonSyntaxException e) {
    System.err.println("Invalid JSON syntax: " + e.getMessage());
} catch (JsonIOException e) {
    System.err.println("I/O error: " + e.getMessage());
}
```