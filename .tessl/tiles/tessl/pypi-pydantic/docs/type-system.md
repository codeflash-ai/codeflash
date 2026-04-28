# Type System and Constraints

Specialized types for common data patterns including network addresses, file paths, dates, colors, and constrained types with built-in validation.

## Capabilities

### Network and URL Types

Specialized string types for network addresses, URLs, and email validation.

```python { .api }
class AnyUrl(Url):
    """
    Base class for URL validation.
    
    Validates URL format and structure.
    """

class AnyHttpUrl(AnyUrl):
    """
    URL that must use HTTP or HTTPS scheme.
    """

class HttpUrl(AnyHttpUrl):
    """
    HTTP or HTTPS URL with additional validation.
    """

class FileUrl(AnyUrl):
    """
    File URL (file:// scheme).
    """

class PostgresDsn(AnyUrl):
    """
    PostgreSQL data source name (DSN).
    """

class MySQLDsn(AnyUrl):
    """
    MySQL data source name (DSN).
    """

class MariaDBDsn(AnyUrl):
    """
    MariaDB data source name (DSN).
    """

class CockroachDsn(AnyUrl):
    """
    CockroachDB data source name (DSN).
    """

class AmqpDsn(AnyUrl):
    """
    AMQP (Advanced Message Queuing Protocol) DSN.
    """

class RedisDsn(AnyUrl):
    """
    Redis data source name (DSN).
    """

class MongoDsn(AnyUrl):
    """
    MongoDB data source name (DSN).
    """

class KafkaDsn(AnyUrl):
    """
    Apache Kafka data source name (DSN).
    """

class NatsDsn(AnyUrl):
    """
    NATS messaging system DSN.
    """

class ClickHouseDsn(AnyUrl):
    """
    ClickHouse database DSN.
    """

class SnowflakeDsn(AnyUrl):
    """
    Snowflake data warehouse DSN.
    """

class EmailStr(str):
    """
    String that must be a valid email address.
    """

class NameEmail(str):
    """
    Email address that can include a display name.
    Format: "Display Name <email@example.com>" or "email@example.com"
    """

class IPvAnyAddress:
    """
    IPv4 or IPv6 address.
    """

class IPvAnyInterface:
    """
    IPv4 or IPv6 interface (address with network mask).
    """

class IPvAnyNetwork:
    """
    IPv4 or IPv6 network.
    """

class UrlConstraints:
    """
    Constraints for URL validation.
    
    Allows customization of URL validation rules.
    """
    
    def __init__(self, *, max_length=None, allowed_schemes=None, host_required=None, 
                 default_host=None, default_port=None, default_path=None):
        """
        Initialize URL constraints.
        
        Args:
            max_length (int): Maximum URL length
            allowed_schemes (set): Set of allowed URL schemes
            host_required (bool): Whether host is required
            default_host (str): Default host if not provided
            default_port (int): Default port if not provided
            default_path (str): Default path if not provided
        """
```

### Constrained Types

Generic constrained types that add validation rules to base Python types.

```python { .api }
def constr(*, min_length=None, max_length=None, strict=None, strip_whitespace=None,
           to_lower=None, to_upper=None, pattern=None):
    """
    Create constrained string type.
    
    Args:
        min_length (int): Minimum string length
        max_length (int): Maximum string length
        strict (bool): Strict mode validation
        strip_whitespace (bool): Strip leading/trailing whitespace
        to_lower (bool): Convert to lowercase
        to_upper (bool): Convert to uppercase
        pattern (str): Regex pattern to match
        
    Returns:
        Constrained string type
    """

def conint(*, strict=None, gt=None, ge=None, lt=None, le=None, multiple_of=None):
    """
    Create constrained integer type.
    
    Args:
        strict (bool): Strict mode validation
        gt: Greater than constraint
        ge: Greater than or equal constraint
        lt: Less than constraint
        le: Less than or equal constraint
        multiple_of: Multiple of constraint
        
    Returns:
        Constrained integer type
    """

def confloat(*, strict=None, gt=None, ge=None, lt=None, le=None, multiple_of=None,
             allow_inf_nan=None):
    """
    Create constrained float type.
    
    Args:
        strict (bool): Strict mode validation
        gt: Greater than constraint
        ge: Greater than or equal constraint
        lt: Less than constraint
        le: Less than or equal constraint
        multiple_of: Multiple of constraint
        allow_inf_nan (bool): Allow infinity and NaN values
        
    Returns:
        Constrained float type
    """

def condecimal(*, strict=None, gt=None, ge=None, lt=None, le=None, multiple_of=None,
               max_digits=None, decimal_places=None, allow_inf_nan=None):
    """
    Create constrained decimal type.
    
    Args:
        strict (bool): Strict mode validation
        gt: Greater than constraint
        ge: Greater than or equal constraint
        lt: Less than constraint
        le: Less than or equal constraint
        multiple_of: Multiple of constraint
        max_digits (int): Maximum number of digits
        decimal_places (int): Maximum decimal places
        allow_inf_nan (bool): Allow infinity and NaN values
        
    Returns:
        Constrained decimal type
    """

def conlist(item_type, *, min_length=None, max_length=None, strict=None):
    """
    Create constrained list type.
    
    Args:
        item_type: Type of list items
        min_length (int): Minimum list length
        max_length (int): Maximum list length
        strict (bool): Strict mode validation
        
    Returns:
        Constrained list type
    """

def conset(item_type, *, min_length=None, max_length=None, strict=None):
    """
    Create constrained set type.
    
    Args:
        item_type: Type of set items
        min_length (int): Minimum set length
        max_length (int): Maximum set length
        strict (bool): Strict mode validation
        
    Returns:
        Constrained set type
    """

def confrozenset(item_type, *, min_length=None, max_length=None, strict=None):
    """
    Create constrained frozenset type.
    
    Args:
        item_type: Type of frozenset items
        min_length (int): Minimum frozenset length
        max_length (int): Maximum frozenset length
        strict (bool): Strict mode validation
        
    Returns:
        Constrained frozenset type
    """

class StringConstraints:
    """
    Modern string constraints class (alternative to constr).
    
    Provides more flexible string validation than the legacy constr function.
    """
    
    def __init__(self, *, min_length=None, max_length=None, pattern=None, 
                 strip_whitespace=None, to_lower=None, to_upper=None):
        """
        Initialize string constraints.
        
        Args:
            min_length (int): Minimum string length
            max_length (int): Maximum string length
            pattern (str): Regex pattern to match
            strip_whitespace (bool): Strip leading/trailing whitespace
            to_lower (bool): Convert to lowercase
            to_upper (bool): Convert to uppercase
        """
```

### Numeric Constraint Types

Pre-defined constrained numeric types for common use cases.

```python { .api }
class PositiveInt(int):
    """Integer that must be positive (> 0)."""

class NegativeInt(int):
    """Integer that must be negative (< 0)."""

class NonNegativeInt(int):
    """Integer that must be non-negative (>= 0)."""

class NonPositiveInt(int):
    """Integer that must be non-positive (<= 0)."""

class PositiveFloat(float):
    """Float that must be positive (> 0)."""

class NegativeFloat(float):
    """Float that must be negative (< 0)."""

class NonNegativeFloat(float):
    """Float that must be non-negative (>= 0)."""

class NonPositiveFloat(float):
    """Float that must be non-positive (<= 0)."""

class FiniteFloat(float):
    """Float that must be finite (not infinity or NaN)."""
```

### Date and Time Types

Enhanced date and time types with additional validation and parsing capabilities.

```python { .api }
class PastDate(date):
    """Date that must be in the past."""

class FutureDate(date):
    """Date that must be in the future."""

class PastDatetime(datetime):
    """Datetime that must be in the past."""

class FutureDatetime(datetime):
    """Datetime that must be in the future."""

class AwareDatetime(datetime):
    """Datetime that must be timezone-aware."""

class NaiveDatetime(datetime):
    """Datetime that must be timezone-naive."""
```

### File and Path Types

Types for file system paths and file validation.

```python { .api }
class FilePath(Path):
    """Path that must point to an existing file."""

class DirectoryPath(Path):
    """Path that must point to an existing directory."""

class NewPath(Path):
    """Path that must not exist (for creating new files/directories)."""

class SocketPath(Path):
    """Path that must point to a Unix socket file."""
```

### UUID Types

UUID validation with different version constraints.

```python { .api }
class UUID1(UUID):
    """UUID version 1."""

class UUID3(UUID):
    """UUID version 3."""

class UUID4(UUID):
    """UUID version 4."""

class UUID5(UUID):
    """UUID version 5."""

class UUID6(UUID):
    """UUID version 6."""

class UUID7(UUID):
    """UUID version 7."""

class UUID8(UUID):
    """UUID version 8."""
```

### JSON and Encoding Types

Types for JSON data and encoded strings.

```python { .api }
class Json:
    """
    JSON string that gets parsed into Python objects.
    """

class Base64Str(str):
    """String that must be valid base64."""

class Base64Bytes(bytes):
    """Bytes that must be valid base64."""

class Base64UrlStr(str):
    """String that must be valid base64url."""

class Base64UrlBytes(bytes):
    """Bytes that must be valid base64url."""

class EncoderProtocol:
    """
    Protocol for defining custom encoders.
    
    Used with EncodedBytes and EncodedStr for custom encoding schemes.
    """
    
    def encode(self, value):
        """
        Encode value to bytes/string.
        
        Args:
            value: Value to encode
            
        Returns:
            Encoded value
        """
    
    def decode(self, value):
        """
        Decode bytes/string to original value.
        
        Args:
            value: Encoded value to decode
            
        Returns:  
            Decoded value
        """

class EncodedBytes(bytes):
    """
    Bytes type with custom encoding/decoding via EncoderProtocol.
    """

class EncodedStr(str):
    """
    String type with custom encoding/decoding via EncoderProtocol.
    """

class Base64Encoder:
    """
    Built-in encoder for base64 encoding.
    """
    
    def encode(self, value):
        """Encode value using base64."""
    
    def decode(self, value):
        """Decode base64 value."""
```

### Advanced Type Utilities

Utility types and annotations for advanced validation control and custom type creation.

```python { .api }
class GetPydanticSchema:
    """
    Utility for creating custom type annotations with schema hooks.
    
    Allows advanced customization of validation and serialization behavior.
    """
    
    def __init__(self, get_schema):
        """
        Initialize with schema generation function.
        
        Args:
            get_schema: Function to generate custom schema
        """

class Tag:
    """
    Tag annotation for discriminated unions with custom logic.
    
    Provides more control over union discrimination than simple string tags.
    """
    
    def __init__(self, tag):
        """
        Initialize with tag value.
        
        Args:
            tag: Tag identifier for discrimination
        """

class Discriminator:
    """
    Advanced discriminator for union types.
    
    Supports custom discrimination logic and field mapping.
    """
    
    def __init__(self, discriminator, *, custom_error_type=None, 
                 custom_error_message=None, custom_error_context=None):
        """
        Initialize discriminator.
        
        Args:
            discriminator: Discriminator field name or function
            custom_error_type (str): Custom error type
            custom_error_message (str): Custom error message
            custom_error_context (dict): Custom error context
        """

class JsonValue:
    """
    Recursive type for JSON-serializable values.
    
    Represents values that can be safely serialized to JSON.
    """

class Secret:
    """
    Generic secret type that can wrap any type.
    
    More flexible than SecretStr and SecretBytes, can wrap any type.
    """
    
    def __init__(self, secret_value):
        """
        Initialize with secret value.
        
        Args:
            secret_value: Value to keep secret (any type)
        """
    
    def get_secret_value(self):
        """
        Get the secret value.
        
        Returns:
            The wrapped secret value
        """

class OnErrorOmit:
    """
    Annotation to omit invalid items from collections instead of failing.
    
    Used with list/set validation to skip invalid items rather than raise errors.
    """

class FailFast:
    """
    Annotation to stop validation at first error for performance.
    
    Useful for large collections where you want to fail quickly.
    """

class Strict:
    """
    Generic strict mode annotation.
    
    Can be applied to any type to enable strict validation.
    """

class AllowInfNan:
    """
    Annotation to control whether float types allow infinity/NaN values.
    
    Can enable or disable inf/nan for specific float fields.
    """
    
    def __init__(self, allow=True):
        """
        Initialize with allow flag.
        
        Args:
            allow (bool): Whether to allow inf/nan values
        """

class ImportString(str):
    """
    String type for dynamically importing Python objects.
    
    Validates that the string represents a valid import path and can load the object.
    """

class PaymentCardNumber(str):
    """
    String type for payment card number validation.
    
    Note: Deprecated in favor of pydantic-extra-types
    """
```

### Color Types

Types for color representation and validation.

```python { .api }
class Color:
    """
    Color representation supporting various formats.
    
    Supports: hex, rgb, rgba, hsl, hsla, and named colors.
    """
    
    def as_hex(self, format='long'):
        """
        Return color as hex string.
        
        Args:
            format (str): 'long' or 'short' format
            
        Returns:
            str: Hex color string
        """
    
    def as_rgb(self):
        """
        Return color as RGB tuple.
        
        Returns:
            tuple: (r, g, b) values
        """
    
    def as_rgb_tuple(self, alpha=None):
        """
        Return color as RGB tuple.
        
        Args:
            alpha (bool): Include alpha channel
            
        Returns:
            tuple: RGB(A) values
        """
```

## Usage Examples

### Network Types

```python
from pydantic import BaseModel, EmailStr, HttpUrl

class UserProfile(BaseModel):
    email: EmailStr
    website: HttpUrl
    avatar_url: HttpUrl

# Valid usage
profile = UserProfile(
    email="user@example.com",
    website="https://example.com",
    avatar_url="https://example.com/avatar.jpg"
)

# Invalid email would raise ValidationError
# profile = UserProfile(email="invalid-email", ...)
```

### Constrained Types

```python
from pydantic import BaseModel, constr, conint, conlist

class Product(BaseModel):
    name: constr(min_length=1, max_length=100, strip_whitespace=True)
    price: confloat(gt=0, le=10000.0)
    quantity: conint(ge=0)
    tags: conlist(str, min_length=1, max_length=10)

# Usage
product = Product(
    name="  Laptop  ",  # Will be stripped to "Laptop"
    price=999.99,
    quantity=5,
    tags=["electronics", "computer"]
)
```

### Numeric Constraints

```python
from pydantic import BaseModel, PositiveInt, NonNegativeFloat

class Transaction(BaseModel):
    id: PositiveInt
    amount: NonNegativeFloat
    fee: NonNegativeFloat

# Valid transaction
transaction = Transaction(id=123, amount=100.0, fee=2.50)

# Invalid - negative amount would raise ValidationError
# transaction = Transaction(id=123, amount=-100.0, fee=2.50)
```

### Date and Time Constraints

```python
from pydantic import BaseModel, PastDate, FutureDatetime
from datetime import date, datetime

class Event(BaseModel):
    birth_date: PastDate
    event_datetime: FutureDatetime

# Usage
event = Event(
    birth_date=date(1990, 1, 1),
    event_datetime=datetime(2024, 12, 31, 18, 0)
)
```

### File Path Types

```python
from pydantic import BaseModel, FilePath, DirectoryPath
from pathlib import Path

class Config(BaseModel):
    config_file: FilePath
    output_dir: DirectoryPath

# Usage (paths must exist)
config = Config(
    config_file=Path("config.json"),
    output_dir=Path("./output")
)
```

### UUID Types

```python
from pydantic import BaseModel, UUID4
import uuid

class Resource(BaseModel):
    id: UUID4
    name: str

# Usage
resource = Resource(
    id=uuid.uuid4(),
    name="My Resource"
)

# String UUIDs are automatically converted
resource2 = Resource(
    id="123e4567-e89b-12d3-a456-426614174000",
    name="Another Resource"
)
```

### JSON Types

```python
from pydantic import BaseModel, Json

class Settings(BaseModel):
    config: Json
    metadata: Json

# JSON strings are parsed automatically
settings = Settings(
    config='{"debug": true, "port": 8000}',
    metadata='["tag1", "tag2", "tag3"]'
)

# Access parsed data
print(settings.config)    # {'debug': True, 'port': 8000}
print(settings.metadata) # ['tag1', 'tag2', 'tag3']
```