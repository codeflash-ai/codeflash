# File Size Formatting

Convert byte counts into human-readable file size representations with support for decimal (SI), binary (IEC), and GNU-style formatting options.

## Capabilities

### Natural Size

Formats byte counts into human-readable file size strings with automatic unit selection and customizable formatting options.

```python { .api }
def naturalsize(
    value: float | str,
    binary: bool = False,
    gnu: bool = False,
    format: str = "%.1f"
) -> str:
    """
    Format a number of bytes like a human-readable filesize.
    
    Args:
        value: Number of bytes (int, float, or string)
        binary: Use binary suffixes (KiB, MiB) with base 2^10 instead of 10^3
        gnu: Use GNU-style prefixes (K, M) with 2^10 definition, ignores binary
        format: Custom formatter for the numeric portion
    
    Returns:
        Human readable representation of a filesize
    
    Examples:
        >>> naturalsize(3000000)
        '3.0 MB'
        >>> naturalsize(3000, binary=True)
        '2.9 KiB'
        >>> naturalsize(3000, gnu=True)
        '2.9K'
        >>> naturalsize(10**28)
        '10.0 RB'
    """
```

## Size Unit Systems

The function supports three different unit systems:

### Decimal (SI) Units - Default
Uses base 1000 with decimal prefixes:
- **B** (Bytes)
- **kB** (kilobytes) = 1,000 bytes
- **MB** (megabytes) = 1,000,000 bytes  
- **GB** (gigabytes) = 1,000,000,000 bytes
- **TB, PB, EB, ZB, YB, RB, QB** (continuing the pattern)

### Binary (IEC) Units
Uses base 1024 with binary prefixes when `binary=True`:
- **B** (Bytes)
- **KiB** (kibibytes) = 1,024 bytes
- **MiB** (mebibytes) = 1,048,576 bytes
- **GiB** (gibibytes) = 1,073,741,824 bytes
- **TiB, PiB, EiB, ZiB, YiB, RiB, QiB** (continuing the pattern)

### GNU Style Units
Uses base 1024 with single-character prefixes when `gnu=True`:
- **B** (Bytes)
- **K** (kilobytes) = 1,024 bytes
- **M** (megabytes) = 1,048,576 bytes
- **G, T, P, E, Z, Y, R, Q** (continuing the pattern)

## Constants

```python { .api }
# Suffix mappings for different size formats
suffixes: dict[str, tuple[str, ...] | str] = {
    "decimal": (" kB", " MB", " GB", " TB", " PB", " EB", " ZB", " YB", " RB", " QB"),
    "binary": (" KiB", " MiB", " GiB", " TiB", " PiB", " EiB", " ZiB", " YiB", " RiB", " QiB"),
    "gnu": "KMGTPEZYRQ"
}
```

## Usage Examples

### Basic File Size Formatting

```python
import humanize

# Default decimal (SI) units
print(humanize.naturalsize(1024))        # "1.0 kB"
print(humanize.naturalsize(1048576))     # "1.0 MB"
print(humanize.naturalsize(1073741824))  # "1.1 GB"

# Binary (IEC) units - more accurate for computer storage
print(humanize.naturalsize(1024, binary=True))        # "1.0 KiB"
print(humanize.naturalsize(1048576, binary=True))     # "1.0 MiB"
print(humanize.naturalsize(1073741824, binary=True))  # "1.0 GiB"

# GNU-style units - compact representation
print(humanize.naturalsize(1024, gnu=True))        # "1.0K"
print(humanize.naturalsize(1048576, gnu=True))     # "1.0M"
print(humanize.naturalsize(1073741824, gnu=True))  # "1.0G"
```

### Custom Formatting

```python
import humanize

# Custom precision
print(humanize.naturalsize(1234567, format="%.3f"))  # "1.235 MB"
print(humanize.naturalsize(1024, format="%.0f"))     # "1 kB"

# Very large numbers
print(humanize.naturalsize(10**28))    # "10.0 RB" (ronnabytes)
print(humanize.naturalsize(10**31))    # "100.0 RB"
print(humanize.naturalsize(10**34))    # "100.0 QB" (quettabytes)
```

### Special Cases

```python
import humanize

# Single byte
print(humanize.naturalsize(1))        # "1 Byte"
print(humanize.naturalsize(1, gnu=True))  # "1B"

# Zero bytes
print(humanize.naturalsize(0))        # "0 Bytes"
print(humanize.naturalsize(0, gnu=True))  # "0B"

# Negative values
print(humanize.naturalsize(-4096, binary=True))  # "-4.0 KiB"

# String input
print(humanize.naturalsize("1048576"))  # "1.0 MB"
```

## Compatibility

The non-GNU modes are compatible with Jinja2's `filesizeformat` filter, making this function suitable for use in web templates and similar contexts where consistent file size formatting is needed.

## Error Handling

- Invalid numeric strings return the original string unchanged
- Non-convertible values return their string representation
- The function handles very large numbers gracefully up to quettabyte scale
- Negative values are properly handled with negative signs preserved