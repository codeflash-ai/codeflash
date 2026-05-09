# JSON Tree Model

Gson provides a tree-based API for building and manipulating JSON structures programmatically. This is useful when you need to modify JSON before serialization or after deserialization.

## JsonElement Hierarchy

```java { .api }
public abstract class JsonElement {
    public abstract JsonElement deepCopy();
    
    // Type checking methods
    public boolean isJsonArray();
    public boolean isJsonObject();
    public boolean isJsonPrimitive();
    public boolean isJsonNull();
    
    // Conversion methods
    public JsonObject getAsJsonObject();
    public JsonArray getAsJsonArray();
    public JsonPrimitive getAsJsonPrimitive();
    public JsonNull getAsJsonNull();
    
    // Primitive value extraction
    public boolean getAsBoolean();
    public Number getAsNumber();
    public String getAsString();
    public double getAsDouble();
    public float getAsFloat();
    public long getAsLong();
    public int getAsInt();
    public byte getAsByte();
    public char getAsCharacter();
}
```

## JsonObject

Represents JSON objects with key-value pairs.

```java { .api }
public final class JsonObject extends JsonElement {
    public JsonObject();
    public JsonObject deepCopy();
    
    // Adding elements
    public void add(String property, JsonElement value);
    public void addProperty(String property, String value);
    public void addProperty(String property, Number value);
    public void addProperty(String property, Boolean value);
    public void addProperty(String property, Character value);
    
    // Accessing elements
    public JsonElement get(String memberName);
    public JsonPrimitive getAsJsonPrimitive(String memberName);
    public JsonArray getAsJsonArray(String memberName);
    public JsonObject getAsJsonObject(String memberName);
    
    // Management
    public JsonElement remove(String property);
    public boolean has(String memberName);
    public Set<Map.Entry<String, JsonElement>> entrySet();
    public Set<String> keySet();
    public int size();
    public boolean isEmpty();
}
```

**Usage:**
```java
// Create JSON object programmatically
JsonObject person = new JsonObject();
person.addProperty("name", "Alice");
person.addProperty("age", 30);
person.addProperty("active", true);

// Convert to string
Gson gson = new Gson();
String json = gson.toJson(person);
// Result: {"name":"Alice","age":30,"active":true}

// Access properties
String name = person.get("name").getAsString();
int age = person.get("age").getAsInt();
boolean active = person.get("active").getAsBoolean();

// Check if property exists
if (person.has("email")) {
    String email = person.get("email").getAsString();
}
```

## JsonArray

Represents JSON arrays.

```java { .api }
public final class JsonArray extends JsonElement {
    public JsonArray();
    public JsonArray(int capacity);
    public JsonArray deepCopy();
    
    // Adding elements
    public void add(JsonElement element);
    public void add(Boolean bool);
    public void add(Character character);
    public void add(Number number);
    public void add(String string);
    
    // Accessing elements
    public JsonElement get(int index);
    
    // Management
    public JsonElement remove(int index);
    public boolean remove(JsonElement element);
    public boolean contains(JsonElement element);
    public int size();
    public boolean isEmpty();
    
    // Iteration
    public Iterator<JsonElement> iterator();
}
```

**Usage:**
```java
// Create JSON array
JsonArray numbers = new JsonArray();
numbers.add(1);
numbers.add(2);
numbers.add(3);

// Convert to string
String json = gson.toJson(numbers);
// Result: [1,2,3]

// Access elements
for (int i = 0; i < numbers.size(); i++) {
    int value = numbers.get(i).getAsInt();
    System.out.println(value);
}

// Using iterator
for (JsonElement element : numbers) {
    int value = element.getAsInt();
    System.out.println(value);
}
```

## JsonPrimitive

Represents JSON primitive values (strings, numbers, booleans).

```java { .api }
public final class JsonPrimitive extends JsonElement {
    public JsonPrimitive(Boolean bool);
    public JsonPrimitive(Number number);
    public JsonPrimitive(String string);
    public JsonPrimitive(Character c);
    public JsonPrimitive deepCopy();
    
    public boolean isBoolean();
    public boolean isNumber();
    public boolean isString();
}
```

**Usage:**
```java
JsonPrimitive stringValue = new JsonPrimitive("Hello");
JsonPrimitive numberValue = new JsonPrimitive(42);
JsonPrimitive boolValue = new JsonPrimitive(true);

// Check types
if (stringValue.isString()) {
    String str = stringValue.getAsString();
}

if (numberValue.isNumber()) {
    int num = numberValue.getAsInt();
    double dbl = numberValue.getAsDouble();
}
```

## JsonNull

Represents JSON null values.

```java { .api }
public final class JsonNull extends JsonElement {
    public static final JsonNull INSTANCE;
    public JsonNull deepCopy();
}
```

**Usage:**
```java
JsonObject obj = new JsonObject();
obj.add("nullValue", JsonNull.INSTANCE);
```

## Tree Conversion

Convert between objects and JSON trees:

```java { .api }
// Object to tree
public JsonElement toJsonTree(Object src);
public JsonElement toJsonTree(Object src, Type typeOfSrc);

// Tree to object
public <T> T fromJson(JsonElement json, Class<T> classOfT) throws JsonSyntaxException;
public <T> T fromJson(JsonElement json, Type typeOfT) throws JsonSyntaxException;
public <T> T fromJson(JsonElement json, TypeToken<T> typeOfT) throws JsonSyntaxException;
```

**Usage:**
```java
// Object to tree
Person person = new Person("Bob", 25);
JsonElement tree = gson.toJsonTree(person);

// Modify the tree
JsonObject obj = tree.getAsJsonObject();
obj.addProperty("email", "bob@example.com");

// Tree back to object
Person modifiedPerson = gson.fromJson(obj, Person.class);

// Tree to JSON string
String json = gson.toJson(tree);
```

## JsonParser

Utility for parsing JSON strings into JsonElement trees.

```java { .api }
public final class JsonParser {
    public static JsonElement parseString(String json) throws JsonSyntaxException;
    public static JsonElement parseReader(Reader reader) throws JsonIOException, JsonSyntaxException;
    public static JsonElement parseReader(JsonReader reader) throws JsonIOException, JsonSyntaxException;
}
```

**Usage:**
```java
String json = "{\"name\":\"Alice\",\"age\":30}";
JsonElement element = JsonParser.parseString(json);

if (element.isJsonObject()) {
    JsonObject obj = element.getAsJsonObject();
    String name = obj.get("name").getAsString();
}
```

## JsonStreamParser

Streaming parser for processing multiple JSON values from a single reader.

```java { .api }
public final class JsonStreamParser implements Iterator<JsonElement> {
    public JsonStreamParser(Reader reader);
    public JsonStreamParser(String json);
    
    public boolean hasNext();
    public JsonElement next() throws JsonParseException;
}
```

**Usage:**
```java
String multipleJson = "{\"name\":\"Alice\"} {\"name\":\"Bob\"} [1,2,3]";
JsonStreamParser parser = new JsonStreamParser(multipleJson);

while (parser.hasNext()) {
    JsonElement element = parser.next();
    System.out.println(element);
}
```

## Complex Tree Manipulation

**Building nested structures:**
```java
JsonObject address = new JsonObject();
address.addProperty("street", "123 Main St");
address.addProperty("city", "Springfield");

JsonObject person = new JsonObject();
person.addProperty("name", "John");
person.add("address", address);

JsonArray hobbies = new JsonArray();
hobbies.add("reading");
hobbies.add("swimming");
person.add("hobbies", hobbies);
```

**Traversing complex structures:**
```java
JsonObject root = JsonParser.parseString(complexJson).getAsJsonObject();

for (Map.Entry<String, JsonElement> entry : root.entrySet()) {
    String key = entry.getKey();
    JsonElement value = entry.getValue();
    
    if (value.isJsonObject()) {
        JsonObject nested = value.getAsJsonObject();
        // Process nested object
    } else if (value.isJsonArray()) {
        JsonArray array = value.getAsJsonArray();
        for (JsonElement item : array) {
            // Process array item
        }
    }
}
```