# Annotations

Gson provides several annotations to control serialization and deserialization behavior directly on fields and classes without requiring configuration changes.

## @SerializedName

Specifies alternative names for fields during serialization and deserialization.

```java { .api }
@Target({ElementType.FIELD, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
public @interface SerializedName {
    String value();
    String[] alternate() default {};
}
```

**Basic usage:**
```java
public class Person {
    @SerializedName("full_name")
    private String name;
    
    @SerializedName("years_old")
    private int age;
}

// JSON: {"full_name":"Alice","years_old":30}
```

**Multiple alternative names for deserialization:**
```java
public class Person {
    @SerializedName(value = "name", alternate = {"full_name", "firstName", "first_name"})
    private String name;
}

// Can deserialize from any of these JSON formats:
// {"name":"Alice"}
// {"full_name":"Alice"}  
// {"firstName":"Alice"}
// {"first_name":"Alice"}
```

## @Expose

Controls which fields are included in serialization and deserialization when using `excludeFieldsWithoutExposeAnnotation()`.

```java { .api }
@Target(ElementType.FIELD)
@Retention(RetentionPolicy.RUNTIME)
public @interface Expose {
    boolean serialize() default true;
    boolean deserialize() default true;
}
```

**Usage:**
```java
public class User {
    @Expose
    private String name;
    
    @Expose(serialize = false) // Only for deserialization
    private String password;
    
    @Expose(deserialize = false) // Only for serialization
    private String token;
    
    private String internalId; // Not exposed
}

Gson gson = new GsonBuilder()
    .excludeFieldsWithoutExposeAnnotation()
    .create();

// Only fields marked with @Expose will be processed
```

## @Since

Marks fields as available since a specific API version.

```java { .api }
@Target({ElementType.FIELD, ElementType.TYPE})
@Retention(RetentionPolicy.RUNTIME)
public @interface Since {
    double value();
}
```

**Usage:**
```java
public class ApiResponse {
    private String status;
    
    @Since(1.1)
    private String message;
    
    @Since(2.0)
    private List<String> errors;
}

// Only include fields from version 1.1 and later
Gson gson = new GsonBuilder()
    .setVersion(1.1)
    .create();

// 'status' and 'message' will be included, but 'errors' will be excluded
```

## @Until

Marks fields as available until a specific API version.

```java { .api }
@Target({ElementType.FIELD, ElementType.TYPE})
@Retention(RetentionPolicy.RUNTIME)
public @interface Until {
    double value();
}
```

**Usage:**
```java
public class LegacyResponse {
    private String data;
    
    @Until(2.0)
    private String oldFormat; // Deprecated in version 2.0
    
    @Since(2.0)
    private String newFormat; // Added in version 2.0
}

// Version 1.5: includes 'data' and 'oldFormat'
Gson gson15 = new GsonBuilder().setVersion(1.5).create();

// Version 2.1: includes 'data' and 'newFormat'
Gson gson21 = new GsonBuilder().setVersion(2.1).create();
```

## @JsonAdapter

Specifies a custom TypeAdapter, JsonSerializer, JsonDeserializer, or TypeAdapterFactory for a field or class.

```java { .api }
@Target({ElementType.TYPE, ElementType.FIELD})
@Retention(RetentionPolicy.RUNTIME)
public @interface JsonAdapter {
    Class<?> value();
    boolean nullSafe() default true;
}
```

**Field-level adapter:**
```java
public class Event {
    private String name;
    
    @JsonAdapter(DateAdapter.class)
    private Date timestamp;
}

public class DateAdapter extends TypeAdapter<Date> {
    private final DateFormat format = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    
    @Override
    public void write(JsonWriter out, Date date) throws IOException {
        if (date == null) {
            out.nullValue();
        } else {
            out.value(format.format(date));
        }
    }
    
    @Override
    public Date read(JsonReader in) throws IOException {
        if (in.peek() == JsonToken.NULL) {
            in.nextNull();
            return null;
        }
        try {
            return format.parse(in.nextString());
        } catch (ParseException e) {
            throw new IOException(e);
        }
    }
}
```

**Class-level adapter:**
```java
@JsonAdapter(ColorAdapter.class)
public class Color {
    private int red, green, blue;
    
    public Color(int red, int green, int blue) {
        this.red = red;
        this.green = green;
        this.blue = blue;
    }
}

public class ColorAdapter extends TypeAdapter<Color> {
    @Override
    public void write(JsonWriter out, Color color) throws IOException {
        if (color == null) {
            out.nullValue();
        } else {
            String hex = String.format("#%02x%02x%02x", color.red, color.green, color.blue);
            out.value(hex);
        }
    }
    
    @Override
    public Color read(JsonReader in) throws IOException {
        if (in.peek() == JsonToken.NULL) {
            in.nextNull();
            return null;
        }
        
        String hex = in.nextString();
        if (hex.startsWith("#")) {
            hex = hex.substring(1);
        }
        
        int rgb = Integer.parseInt(hex, 16);
        return new Color((rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF);
    }
}

// Usage: Color serializes as "#ff0000" instead of {"red":255,"green":0,"blue":0}
```

**Using with JsonSerializer/JsonDeserializer:**
```java
public class User {
    private String name;
    
    @JsonAdapter(PasswordSerializer.class)
    private String password;
}

public class PasswordSerializer implements JsonSerializer<String>, JsonDeserializer<String> {
    @Override
    public JsonElement serialize(String src, Type typeOfSrc, JsonSerializationContext context) {
        return new JsonPrimitive("***"); // Always serialize as stars
    }
    
    @Override
    public String deserialize(JsonElement json, Type typeOfT, JsonDeserializationContext context) {
        return json.getAsString(); // Deserialize normally
    }
}
```

**Using with TypeAdapterFactory:**
```java
@JsonAdapter(CaseInsensitiveEnumAdapterFactory.class)
public enum Status {
    ACTIVE, INACTIVE, PENDING
}

public class CaseInsensitiveEnumAdapterFactory implements TypeAdapterFactory {
    @Override
    @SuppressWarnings("unchecked")
    public <T> TypeAdapter<T> create(Gson gson, TypeToken<T> type) {
        Class<T> rawType = (Class<T>) type.getRawType();
        if (!rawType.isEnum()) {
            return null;
        }
        
        return new TypeAdapter<T>() {
            @Override
            public void write(JsonWriter out, T value) throws IOException {
                if (value == null) {
                    out.nullValue();
                } else {
                    out.value(value.toString());
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

## Combining Annotations

Annotations can be combined for comprehensive control:

```java
public class Product {
    @SerializedName("product_id")
    @Since(1.0)
    private String id;
    
    @SerializedName("product_name")
    @Expose
    private String name;
    
    @SerializedName("price_cents")
    @JsonAdapter(MoneyAdapter.class)
    private Money price;
    
    @Until(2.0)
    private String oldCategory;
    
    @Since(2.0)
    @SerializedName("category_info")
    private Category category;
    
    @Expose(serialize = false)
    private String internalNotes;
}
```

## Annotation Processing Order

Gson processes annotations in the following priority:

1. **@JsonAdapter** - Takes highest precedence
2. **@SerializedName** - Controls field naming
3. **@Expose** - Controls field inclusion (when enabled)
4. **@Since/@Until** - Controls version-based inclusion
5. **Field modifiers** - Processed based on GsonBuilder configuration

**Example:**
```java
public class Example {
    @JsonAdapter(CustomAdapter.class)  // 1. Custom adapter used
    @SerializedName("custom_field")    // 2. Field name in JSON
    @Expose(serialize = false)         // 3. Only deserialize (if expose filtering enabled)
    @Since(1.1)                        // 4. Only if version >= 1.1
    private String field;
}
```

## Best Practices

**Use @SerializedName for API compatibility:**
```java
public class ApiModel {
    @SerializedName("user_id")
    private String userId;  // Java convention: camelCase
    
    @SerializedName("created_at")
    private Date createdAt; // API uses snake_case
}
```

**Use @Expose for security:**
```java
public class User {
    @Expose
    private String username;
    
    @Expose
    private String email;
    
    private String passwordHash; // Never expose sensitive data
    
    @Expose(serialize = false)
    private String password; // Accept for input, never output
}
```

**Use versioning for API evolution:**
```java
public class ApiResponse {
    private String status;
    
    @Until(1.9)
    private String oldErrorMessage;
    
    @Since(2.0)
    private ErrorDetails errorDetails;
}
```

**Use @JsonAdapter for domain-specific serialization:**
```java
public class Order {
    @JsonAdapter(CurrencyAdapter.class)
    private BigDecimal totalAmount;
    
    @JsonAdapter(TimestampAdapter.class)
    private Instant orderTime;
    
    @JsonAdapter(Base64Adapter.class)
    private byte[] signature;
}
```