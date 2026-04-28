# Type Adapters

Type adapters provide fine-grained control over JSON serialization and deserialization for specific types. They are the most flexible way to customize Gson's behavior.

## TypeAdapter Abstract Class

The base class for all type adapters.

```java { .api }
public abstract class TypeAdapter<T> {
    public abstract void write(JsonWriter out, T value) throws IOException;
    public abstract T read(JsonReader in) throws IOException;
    
    // Convenience methods
    public final String toJson(T value);
    public final void toJson(Writer out, T value) throws IOException;
    public final T fromJson(String json) throws IOException;
    public final T fromJson(Reader in) throws IOException;
    
    // Null handling
    public final TypeAdapter<T> nullSafe();
}
```

## Creating Custom Type Adapters

**Basic type adapter example:**
```java
public class PersonAdapter extends TypeAdapter<Person> {
    @Override
    public void write(JsonWriter out, Person person) throws IOException {
        if (person == null) {
            out.nullValue();
            return;
        }
        
        out.beginObject();
        out.name("name").value(person.getName());
        out.name("age").value(person.getAge());
        out.name("email").value(person.getEmail());
        out.endObject();
    }
    
    @Override
    public Person read(JsonReader in) throws IOException {
        if (in.peek() == JsonToken.NULL) {
            in.nextNull();
            return null;
        }
        
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
                case "email":
                    person.setEmail(in.nextString());
                    break;
                default:
                    in.skipValue(); // Skip unknown properties
                    break;
            }
        }
        in.endObject();
        return person;
    }
}
```

**Registering the adapter:**
```java
Gson gson = new GsonBuilder()
    .registerTypeAdapter(Person.class, new PersonAdapter())
    .create();

Person person = new Person("Alice", 30, "alice@example.com");
String json = gson.toJson(person);
Person restored = gson.fromJson(json, Person.class);
```

## TypeAdapterFactory Interface

Factory pattern for creating type adapters dynamically.

```java { .api }
public interface TypeAdapterFactory {
    public <T> TypeAdapter<T> create(Gson gson, TypeToken<T> type);
}
```

**Example factory for enum handling:**
```java
public class LowercaseEnumTypeAdapterFactory implements TypeAdapterFactory {
    @Override
    @SuppressWarnings("unchecked")
    public <T> TypeAdapter<T> create(Gson gson, TypeToken<T> type) {
        Class<T> rawType = (Class<T>) type.getRawType();
        if (!rawType.isEnum()) {
            return null; // This factory only handles enums
        }
        
        return new TypeAdapter<T>() {
            @Override
            public void write(JsonWriter out, T value) throws IOException {
                if (value == null) {
                    out.nullValue();
                } else {
                    out.value(value.toString().toLowerCase());
                }
            }
            
            @Override
            @SuppressWarnings("unchecked")
            public T read(JsonReader in) throws IOException {
                if (in.peek() == JsonToken.NULL) {
                    in.nextNull();
                    return null;
                }
                
                String value = in.nextString().toUpperCase();
                return (T) Enum.valueOf((Class<Enum>) rawType, value);
            }
        };
    }
}
```

**Registering the factory:**
```java
Gson gson = new GsonBuilder()
    .registerTypeAdapterFactory(new LowercaseEnumTypeAdapterFactory())
    .create();
```

## Built-in Type Adapters

Gson provides many built-in type adapters accessible through the TypeAdapters utility class.

```java { .api }
public final class TypeAdapters {
    public static final TypeAdapter<Class> CLASS;
    public static final TypeAdapter<Boolean> BOOLEAN;
    public static final TypeAdapter<Boolean> BOOLEAN_AS_STRING;
    public static final TypeAdapter<Number> BYTE;
    public static final TypeAdapter<Number> SHORT;
    public static final TypeAdapter<Number> INTEGER;
    public static final TypeAdapter<AtomicInteger> ATOMIC_INTEGER;
    public static final TypeAdapter<AtomicBoolean> ATOMIC_BOOLEAN;
    public static final TypeAdapter<Number> LONG;
    public static final TypeAdapter<Number> FLOAT;
    public static final TypeAdapter<Number> DOUBLE;
    public static final TypeAdapter<Character> CHARACTER;
    public static final TypeAdapter<String> STRING;
    public static final TypeAdapter<StringBuilder> STRING_BUILDER;
    public static final TypeAdapter<StringBuffer> STRING_BUFFER;
    public static final TypeAdapter<URL> URL;
    public static final TypeAdapter<URI> URI;
    public static final TypeAdapter<UUID> UUID;
    public static final TypeAdapter<Currency> CURRENCY;
    public static final TypeAdapter<Calendar> CALENDAR;
    public static final TypeAdapter<Locale> LOCALE;
    public static final TypeAdapter<JsonElement> JSON_ELEMENT;
    
    // Factory methods
    public static <TT> TypeAdapterFactory newFactory(Class<TT> type, TypeAdapter<TT> typeAdapter);
    public static <TT> TypeAdapterFactory newFactory(TypeToken<TT> type, TypeAdapter<TT> typeAdapter);
    public static TypeAdapterFactory newFactoryForMultipleTypes(Class<?> base, Class<?> sub, TypeAdapter<?> typeAdapter);
}
```

## Advanced Type Adapter Patterns

### Generic Type Handling

```java
public class ListTypeAdapterFactory implements TypeAdapterFactory {
    @Override
    @SuppressWarnings("unchecked")
    public <T> TypeAdapter<T> create(Gson gson, TypeToken<T> type) {
        Type rawType = type.getRawType();
        if (rawType != List.class) {
            return null;
        }
        
        Type elementType = ((ParameterizedType) type.getType()).getActualTypeArguments()[0];
        TypeAdapter<?> elementAdapter = gson.getAdapter(TypeToken.get(elementType));
        
        return (TypeAdapter<T>) new TypeAdapter<List<?>>() {
            @Override
            public void write(JsonWriter out, List<?> list) throws IOException {
                if (list == null) {
                    out.nullValue();
                    return;
                }
                
                out.beginArray();
                for (Object element : list) {
                    elementAdapter.write(out, element);
                }
                out.endArray();
            }
            
            @Override
            public List<?> read(JsonReader in) throws IOException {
                if (in.peek() == JsonToken.NULL) {
                    in.nextNull();
                    return null;
                }
                
                List<Object> list = new ArrayList<>();
                in.beginArray();
                while (in.hasNext()) {
                    list.add(elementAdapter.read(in));
                }
                in.endArray();
                return list;
            }
        };
    }
}
```

### Delegating Adapters

Sometimes you need to modify the behavior of an existing adapter:

```java
public class TimestampAdapter extends TypeAdapter<Date> {
    private final TypeAdapter<Date> delegate;
    
    public TimestampAdapter(TypeAdapter<Date> delegate) {
        this.delegate = delegate;
    }
    
    @Override
    public void write(JsonWriter out, Date date) throws IOException {
        if (date == null) {
            out.nullValue();
        } else {
            out.value(date.getTime()); // Write as timestamp
        }
    }
    
    @Override
    public Date read(JsonReader in) throws IOException {
        if (in.peek() == JsonToken.NULL) {
            in.nextNull();
            return null;
        }
        
        long timestamp = in.nextLong();
        return new Date(timestamp);
    }
}
```

### Hierarchical Type Adapters

Handle inheritance hierarchies:

```java
Gson gson = new GsonBuilder()
    .registerTypeHierarchyAdapter(Animal.class, new AnimalAdapter())
    .create();

// This adapter will be used for Animal and all its subclasses
public class AnimalAdapter implements JsonSerializer<Animal>, JsonDeserializer<Animal> {
    @Override
    public JsonElement serialize(Animal src, Type typeOfSrc, JsonSerializationContext context) {
        JsonObject result = new JsonObject();
        result.add("type", new JsonPrimitive(src.getClass().getSimpleName()));
        result.add("properties", context.serialize(src, src.getClass()));
        return result;
    }
    
    @Override
    public Animal deserialize(JsonElement json, Type typeOfT, JsonDeserializationContext context) throws JsonParseException {
        JsonObject jsonObject = json.getAsJsonObject();
        String type = jsonObject.get("type").getAsString();
        JsonElement element = jsonObject.get("properties");
        
        try {
            Class<?> clazz = Class.forName("com.example." + type);
            return context.deserialize(element, clazz);
        } catch (ClassNotFoundException e) {
            throw new JsonParseException("Unknown element type: " + type, e);
        }
    }
}
```

## Null Handling

Type adapters can handle nulls explicitly or use the `nullSafe()` wrapper:

```java
public class NullSafeStringAdapter extends TypeAdapter<String> {
    @Override
    public void write(JsonWriter out, String value) throws IOException {
        out.value(value == null ? "NULL" : value);
    }
    
    @Override
    public String read(JsonReader in) throws IOException {
        String value = in.nextString();
        return "NULL".equals(value) ? null : value;
    }
}

// Or use nullSafe wrapper
TypeAdapter<String> adapter = new StringAdapter().nullSafe();
```

## Accessing Delegate Adapters

Get the next adapter in the chain to avoid infinite recursion:

```java
Gson gson = new GsonBuilder()
    .registerTypeAdapterFactory(new LoggingAdapterFactory())
    .create();

public class LoggingAdapterFactory implements TypeAdapterFactory {
    @Override
    public <T> TypeAdapter<T> create(Gson gson, TypeToken<T> type) {
        TypeAdapter<T> delegate = gson.getDelegateAdapter(this, type);
        
        return new TypeAdapter<T>() {
            @Override
            public void write(JsonWriter out, T value) throws IOException {
                System.out.println("Serializing: " + value);
                delegate.write(out, value);
            }
            
            @Override
            public T read(JsonReader in) throws IOException {
                T result = delegate.read(in);
                System.out.println("Deserialized: " + result);
                return result;
            }
        };
    }
}
```

## Legacy Serializer/Deserializer Interfaces

For simpler cases, you can use the legacy interfaces:

```java { .api }
public interface JsonSerializer<T> {
    public JsonElement serialize(T src, Type typeOfSrc, JsonSerializationContext context);
}

public interface JsonDeserializer<T> {
    public T deserialize(JsonElement json, Type typeOfT, JsonDeserializationContext context) throws JsonParseException;
}
```

**Context interfaces:**
```java { .api }
public interface JsonSerializationContext {
    public JsonElement serialize(Object src);
    public JsonElement serialize(Object src, Type typeOfSrc);
}

public interface JsonDeserializationContext {
    public <T> T deserialize(JsonElement json, Type typeOfT) throws JsonParseException;
}
```

**Usage:**
```java
public class PersonSerializer implements JsonSerializer<Person> {
    @Override
    public JsonElement serialize(Person src, Type typeOfSrc, JsonSerializationContext context) {
        JsonObject obj = new JsonObject();
        obj.addProperty("fullName", src.getFirstName() + " " + src.getLastName());
        obj.addProperty("age", src.getAge());
        return obj;
    }
}

Gson gson = new GsonBuilder()
    .registerTypeAdapter(Person.class, new PersonSerializer())
    .create();
```