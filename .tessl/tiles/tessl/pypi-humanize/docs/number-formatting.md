# Number Formatting

Comprehensive number formatting utilities for converting numeric values into human-readable formats including comma separation, word conversion, ordinals, fractions, scientific notation, and metric prefixes.

## Capabilities

### Comma Formatting

Adds comma thousands separators to numbers for improved readability, supporting both integers and floats with optional decimal precision control.

```python { .api }
def intcomma(value: float | str, ndigits: int | None = None) -> str:
    """
    Converts an integer to a string containing commas every three digits.
    
    Args:
        value: Integer, float, or string to convert
        ndigits: Optional digits of precision for rounding after decimal point
    
    Returns:
        String with commas every three digits
    
    Examples:
        >>> intcomma(1000)
        '1,000'
        >>> intcomma(1_234_567.25)
        '1,234,567.25'
        >>> intcomma(1234.5454545, 2)
        '1,234.55'
    """
```

### Word Conversion

Converts large numbers into friendly text representations using words like "million", "billion", etc. Supports numbers up to decillion and googol.

```python { .api }
def intword(value: float | str, format: str = "%.1f") -> str:
    """
    Converts a large integer to a friendly text representation.
    
    Args:
        value: Integer, float, or string to convert
        format: Format string for the number portion (default "%.1f")
    
    Returns:
        Friendly text representation, or original string if conversion fails
    
    Examples:
        >>> intword(1000000)
        '1.0 million'
        >>> intword(1_200_000_000)
        '1.2 billion'
        >>> intword("1234000", "%0.3f")
        '1.234 million'
    """
```

### Ordinal Numbers

Converts integers to ordinal form (1st, 2nd, 3rd, etc.) with support for gendered translations in supported languages.

```python { .api }
def ordinal(value: float | str, gender: str = "male") -> str:
    """
    Converts an integer to its ordinal as a string.
    
    Args:
        value: Integer, float, or string to convert
        gender: Gender for translations ("male" or "female")
    
    Returns:
        Ordinal string representation
    
    Examples:
        >>> ordinal(1)
        '1st'
        >>> ordinal(22)
        '22nd'
        >>> ordinal(103)
        '103rd'
    """
```

### Associated Press Style

Converts numbers 0-9 to Associated Press style word format, returning digits for larger numbers.

```python { .api }
def apnumber(value: float | str) -> str:
    """
    Converts an integer to Associated Press style.
    
    Args:
        value: Integer, float, or string to convert
    
    Returns:
        For 0-9: number spelled out; otherwise: the number as string
    
    Examples:
        >>> apnumber(0)
        'zero'
        >>> apnumber(5)
        'five'
        >>> apnumber(10)
        '10'
    """
```

### Fractional Representation

Converts decimal numbers to human-readable fractional form including proper fractions, improper fractions, and mixed numbers.

```python { .api }
def fractional(value: float | str) -> str:
    """
    Convert to fractional number representation.
    
    Args:
        value: Integer, float, or string to convert
    
    Returns:
        Fractional representation as string
    
    Examples:
        >>> fractional(0.3)
        '3/10'
        >>> fractional(1.3)
        '1 3/10'
        >>> fractional(float(1/3))
        '1/3'
    """
```

### Scientific Notation

Formats numbers in scientific notation with customizable precision and Unicode superscript exponents.

```python { .api }
def scientific(value: float | str, precision: int = 2) -> str:
    """
    Return number in scientific notation z.wq x 10ⁿ.
    
    Args:
        value: Input number to format
        precision: Number of decimal places for mantissa
    
    Returns:
        Number in scientific notation with Unicode superscripts
    
    Examples:
        >>> scientific(500)
        '5.00 x 10²'
        >>> scientific(-1000)
        '-1.00 x 10³'
        >>> scientific(1000, 1)
        '1.0 x 10³'
    """
```

### Clamped Formatting

Returns formatted numbers clamped between floor and ceiling values, with tokens indicating when limits are exceeded.

```python { .api }
def clamp(
    value: float,
    format: str = "{:}",
    floor: float | None = None,
    ceil: float | None = None,
    floor_token: str = "<",
    ceil_token: str = ">"
) -> str:
    """
    Returns number clamped between floor and ceil with tokens.
    
    Args:
        value: Input number to clamp and format
        format: Format string or callable function
        floor: Minimum value before clamping
        ceil: Maximum value before clamping  
        floor_token: Token prepended when value < floor
        ceil_token: Token prepended when value > ceil
    
    Returns:
        Formatted number with clamp tokens if needed
    
    Examples:
        >>> clamp(0.0001, floor=0.01)
        '<0.01'
        >>> clamp(0.999, format="{:.0%}", ceil=0.99)
        '>99%'
        >>> clamp(1, format=intword, floor=1e6, floor_token="under ")
        'under 1.0 million'
    """
```

### Metric Prefixes

Formats numbers with SI metric unit prefixes, automatically choosing the most appropriate scale to avoid leading or trailing zeros.

```python { .api }
def metric(value: float, unit: str = "", precision: int = 3) -> str:
    """
    Return a value with a metric SI unit-prefix appended.
    
    Args:
        value: Input number to format
        unit: Optional base unit (e.g., "V", "W", "F")
        precision: Number of significant digits
    
    Returns:
        Number with appropriate metric prefix
    
    Examples:
        >>> metric(1500, "V")
        '1.50 kV'
        >>> metric(220e-6, "F")
        '220 μF'
        >>> metric(2e8, "W")
        '200 MW'
    """
```

## Constants

```python { .api }
# Powers of 10 used for large number conversion
powers: list[int]

# Human-readable names for large numbers
human_powers: tuple[tuple[str, str], ...]
```

## Error Handling

All number formatting functions handle invalid inputs gracefully:

- Non-numeric strings return the original string unchanged
- `None` values return the string `"None"`
- Infinite values return `"+Inf"`, `"-Inf"`, or `"NaN"` as appropriate
- Mathematical edge cases are handled consistently across all functions