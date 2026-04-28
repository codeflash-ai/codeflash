# Streaming API

Gson's streaming API provides memory-efficient processing of JSON data through JsonReader and JsonWriter classes. This is essential for handling large JSON documents without loading them entirely into memory.

## JsonReader

Read JSON content in a streaming fashion.

```java { .api }
public class JsonReader implements Closeable {
    public JsonReader(Reader in);
    
    // Configuration
    public void setLenient(boolean lenient);
    public boolean isLenient();
    public void setStrictness(Strictness strictness);
    public Strictness getStrictness();
    
    // Navigation
    public JsonToken peek() throws IOException;
    public JsonToken nextToken() throws IOException;
    public void skipValue() throws IOException;
    
    // Object navigation
    public void beginObject() throws IOException;
    public void endObject() throws IOException;
    public String nextName() throws IOException;
    
    // Array navigation
    public void beginArray() throws IOException;
    public void endArray() throws IOException;
    
    // Value reading
    public String nextString() throws IOException;
    public boolean nextBoolean() throws IOException;
    public void nextNull() throws IOException;
    public double nextDouble() throws IOException;
    public long nextLong() throws IOException;
    public int nextInt() throws IOException;
    
    // State
    public boolean hasNext() throws IOException;
    public String getPath();
    
    // Resource management
    public void close() throws IOException;
}
```

### JsonToken Enum

```java { .api }
public enum JsonToken {
    BEGIN_ARRAY,
    END_ARRAY,
    BEGIN_OBJECT,
    END_OBJECT,
    NAME,
    STRING,
    NUMBER,
    BOOLEAN,
    NULL,
    END_DOCUMENT
}
```

**Reading JSON objects:**
```java
String json = "{\"name\":\"Alice\",\"age\":30,\"active\":true}";
JsonReader reader = new JsonReader(new StringReader(json));

reader.beginObject();
while (reader.hasNext()) {
    String name = reader.nextName();
    switch (name) {
        case "name":
            String personName = reader.nextString();
            break;
        case "age":
            int age = reader.nextInt();
            break;
        case "active":
            boolean active = reader.nextBoolean();
            break;
        default:
            reader.skipValue(); // Skip unknown properties
            break;
    }
}
reader.endObject();
reader.close();
```

**Reading JSON arrays:**
```java
String json = "[1,2,3,4,5]";
JsonReader reader = new JsonReader(new StringReader(json));

reader.beginArray();
while (reader.hasNext()) {
    int value = reader.nextInt();
    System.out.println(value);
}
reader.endArray();
reader.close();
```

**Reading complex nested structures:**
```java
String json = "{\"users\":[{\"name\":\"Alice\",\"age\":30},{\"name\":\"Bob\",\"age\":25}]}";
JsonReader reader = new JsonReader(new StringReader(json));

reader.beginObject();
while (reader.hasNext()) {
    String name = reader.nextName();
    if ("users".equals(name)) {
        reader.beginArray();
        while (reader.hasNext()) {
            reader.beginObject();
            String userName = null;
            int age = 0;
            while (reader.hasNext()) {
                String fieldName = reader.nextName();
                if ("name".equals(fieldName)) {
                    userName = reader.nextString();
                } else if ("age".equals(fieldName)) {
                    age = reader.nextInt();
                } else {
                    reader.skipValue();
                }
            }
            reader.endObject();
            System.out.println("User: " + userName + ", Age: " + age);
        }
        reader.endArray();
    } else {
        reader.skipValue();
    }
}
reader.endObject();
reader.close();
```

## JsonWriter

Write JSON content in a streaming fashion.

```java { .api }
public class JsonWriter implements Closeable, Flushable {
    public JsonWriter(Writer out);
    
    // Configuration
    public void setLenient(boolean lenient);
    public boolean isLenient();
    public void setStrictness(Strictness strictness);
    public Strictness getStrictness();
    public void setIndent(String indent);
    public void setHtmlSafe(boolean htmlSafe);
    public boolean isHtmlSafe();
    public void setSerializeNulls(boolean serializeNulls);
    public boolean getSerializeNulls();
    
    // Object writing
    public JsonWriter beginObject() throws IOException;
    public JsonWriter endObject() throws IOException;
    public JsonWriter name(String name) throws IOException;
    
    // Array writing
    public JsonWriter beginArray() throws IOException;
    public JsonWriter endArray() throws IOException;
    
    // Value writing
    public JsonWriter value(String value) throws IOException;
    public JsonWriter nullValue() throws IOException;
    public JsonWriter value(boolean value) throws IOException;
    public JsonWriter value(Boolean value) throws IOException;
    public JsonWriter value(double value) throws IOException;
    public JsonWriter value(long value) throws IOException;
    public JsonWriter value(Number value) throws IOException;
    
    // JSON value writing
    public JsonWriter jsonValue(String value) throws IOException;
    
    // Resource management
    public void flush() throws IOException;
    public void close() throws IOException;
}
```

**Writing JSON objects:**
```java
StringWriter stringWriter = new StringWriter();
JsonWriter writer = new JsonWriter(stringWriter);

writer.beginObject();
writer.name("name").value("Alice");
writer.name("age").value(30);
writer.name("active").value(true);
writer.name("salary").nullValue();
writer.endObject();

writer.close();
String json = stringWriter.toString();
// Result: {"name":"Alice","age":30,"active":true,"salary":null}
```

**Writing JSON arrays:**
```java
StringWriter stringWriter = new StringWriter();
JsonWriter writer = new JsonWriter(stringWriter);

writer.beginArray();
writer.value(1);
writer.value(2);
writer.value(3);
writer.endArray();

writer.close();
String json = stringWriter.toString();
// Result: [1,2,3]
```

**Writing complex nested structures:**
```java
StringWriter stringWriter = new StringWriter();
JsonWriter writer = new JsonWriter(stringWriter);

writer.beginObject();
writer.name("users");
writer.beginArray();

writer.beginObject();
writer.name("name").value("Alice");
writer.name("age").value(30);
writer.name("hobbies");
writer.beginArray();
writer.value("reading");
writer.value("swimming");
writer.endArray();
writer.endObject();

writer.beginObject();
writer.name("name").value("Bob");
writer.name("age").value(25);
writer.name("hobbies");
writer.beginArray();
writer.value("gaming");
writer.endArray();
writer.endObject();

writer.endArray();
writer.endObject();

writer.close();
```

## Gson Integration

Create JsonReader and JsonWriter through Gson for consistent configuration:

```java { .api }
public JsonWriter newJsonWriter(Writer writer) throws IOException;
public JsonReader newJsonReader(Reader reader);
```

**Usage:**
```java
Gson gson = new GsonBuilder()
    .setPrettyPrinting()
    .setLenient()
    .create();

// Create configured writer
StringWriter stringWriter = new StringWriter();
JsonWriter writer = gson.newJsonWriter(stringWriter);

// Create configured reader
JsonReader reader = gson.newJsonReader(new StringReader(json));
```

## Error Handling

```java { .api }
class MalformedJsonException extends IOException {
    // Thrown when JSON structure is invalid
}
```

**Error handling example:**
```java
try {
    JsonReader reader = new JsonReader(new StringReader(malformedJson));
    reader.beginObject();
    // ... reading logic
} catch (MalformedJsonException e) {
    System.err.println("Malformed JSON: " + e.getMessage());
} catch (IOException e) {
    System.err.println("I/O error: " + e.getMessage());
}
```

## Advanced Usage

### Lenient Mode

Allow relaxed JSON syntax:
```java
JsonReader reader = new JsonReader(new StringReader(json));
reader.setLenient(true);

// Now can read: {name: 'Alice', age: 30} (unquoted names, single quotes)
```

### Pretty Printing

Configure indentation for readable output:
```java
JsonWriter writer = new JsonWriter(new FileWriter("output.json"));
writer.setIndent("  "); // 2-space indentation

writer.beginObject();
writer.name("name").value("Alice");
writer.name("details");
writer.beginObject();
writer.name("age").value(30);
writer.endObject();
writer.endObject();
```

### Large File Processing

Process large JSON files efficiently:
```java
// Reading large file
try (FileReader fileReader = new FileReader("large-data.json");
     JsonReader reader = new JsonReader(fileReader)) {
    
    reader.beginArray();
    while (reader.hasNext()) {
        // Process each object without loading entire file
        processObject(reader);
    }
    reader.endArray();
}

private void processObject(JsonReader reader) throws IOException {
    reader.beginObject();
    while (reader.hasNext()) {
        String name = reader.nextName();
        // Handle each field as needed
        if ("data".equals(name)) {
            // Process data field
        } else {
            reader.skipValue();
        }
    }
    reader.endObject();
}
```

### Custom Adapter Integration

Use streaming API within custom TypeAdapters:
```java
public class PersonAdapter extends TypeAdapter<Person> {
    @Override
    public void write(JsonWriter out, Person person) throws IOException {
        out.beginObject();
        out.name("name").value(person.getName());
        out.name("age").value(person.getAge());
        out.endObject();
    }
    
    @Override
    public Person read(JsonReader in) throws IOException {
        Person person = new Person();
        in.beginObject();
        while (in.hasNext()) {
            String name = in.nextName();
            switch (name) {
                case "name":
                    person.setName(in.nextString());
                    break;
                case "age":
                    person.setAge(in.nextInt());
                    break;
                default:
                    in.skipValue();
                    break;
            }
        }
        in.endObject();
        return person;
    }
}
```