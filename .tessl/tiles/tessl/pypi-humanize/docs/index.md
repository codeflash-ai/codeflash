# Humanize

A comprehensive Python library that transforms machine-readable data into human-friendly formats. Humanize provides utilities for converting numbers into readable formats, dates and times into natural language expressions, file sizes into human-readable units, and includes extensive localization support for over 25 languages.

## Package Information

- **Package Name**: humanize
- **Language**: Python
- **Installation**: `pip install humanize`
- **Minimum Python Version**: 3.9+

## Core Imports

```python
import humanize
```

Import specific functions:

```python
from humanize import naturalsize, intcomma, naturaltime
```

Import from submodules:

```python
from humanize.time import naturaldelta, precisedelta
from humanize.number import intword, ordinal
```

## Basic Usage

```python
import humanize
import datetime as dt

# Format numbers with commas
print(humanize.intcomma(1000000))  # "1,000,000"

# Convert numbers to words  
print(humanize.intword(1200000))   # "1.2 million"

# Format file sizes
print(humanize.naturalsize(1024))  # "1.0 kB"

# Natural time expressions
now = dt.datetime.now()
past = now - dt.timedelta(minutes=30)
print(humanize.naturaltime(past))  # "30 minutes ago"

# Create natural lists
items = ["apples", "oranges", "bananas"]
print(humanize.natural_list(items))  # "apples, oranges and bananas"
```

## Architecture

The humanize library is organized into focused modules:

- **Number Module**: Comprehensive number formatting utilities including comma separation, word conversion, ordinals, fractions, scientific notation, and metric prefixes
- **Time Module**: Natural language time and date formatting with relative expressions, precision control, and internationalization
- **File Size Module**: Human-readable file size formatting with binary/decimal unit support
- **Lists Module**: Natural language list formatting with proper conjunction usage
- **Internationalization Module**: Locale activation, deactivation, and formatting customization for 25+ languages

All functions handle edge cases gracefully, support various input types, and provide consistent error handling by returning string representations of invalid inputs.

## Capabilities

### Number Formatting

Convert numbers into human-readable formats including comma separation, word conversion, ordinals, fractions, scientific notation, and metric prefixes with SI units.

```python { .api }
def intcomma(value: float | str, ndigits: int | None = None) -> str: ...
def intword(value: float | str, format: str = "%.1f") -> str: ...
def ordinal(value: float | str, gender: str = "male") -> str: ...
def apnumber(value: float | str) -> str: ...
def fractional(value: float | str) -> str: ...
def scientific(value: float | str, precision: int = 2) -> str: ...
def clamp(value: float, format: str = "{:}", floor: float | None = None, ceil: float | None = None, floor_token: str = "<", ceil_token: str = ">") -> str: ...
def metric(value: float, unit: str = "", precision: int = 3) -> str: ...
```

[Number Formatting](./number-formatting.md)

### Time and Date Formatting

Natural language time and date formatting with relative expressions, precision control, and support for various input types including datetime objects, timedeltas, and seconds.

```python { .api }
def naturaltime(value: dt.datetime | dt.timedelta | float, future: bool = False, months: bool = True, minimum_unit: str = "seconds", when: dt.datetime | None = None) -> str: ...
def naturaldelta(value: dt.timedelta | float, months: bool = True, minimum_unit: str = "seconds") -> str: ...
def naturaldate(value: dt.date | dt.datetime) -> str: ...
def naturalday(value: dt.date | dt.datetime, format: str = "%b %d") -> str: ...
def precisedelta(value: dt.timedelta | float | None, minimum_unit: str = "seconds", suppress: Iterable[str] = (), format: str = "%0.2f") -> str: ...
```

[Time and Date Formatting](./time-formatting.md)

### File Size Formatting

Convert byte counts into human-readable file size representations with support for decimal (SI) and binary (IEC) units, plus GNU-style formatting.

```python { .api }
def naturalsize(value: float | str, binary: bool = False, gnu: bool = False, format: str = "%.1f") -> str: ...
```

[File Size Formatting](./filesize-formatting.md)

### List Formatting

Convert Python lists into natural language with proper comma placement and conjunction usage.

```python { .api }
def natural_list(items: list[Any]) -> str: ...
```

[List Formatting](./list-formatting.md)

### Internationalization

Activate and manage localization for 25+ supported languages, with locale-specific number and date formatting.

```python { .api }
def activate(locale: str | None, path: str | os.PathLike[str] | None = None) -> gettext.NullTranslations: ...
def deactivate() -> None: ...
def thousands_separator() -> str: ...
def decimal_separator() -> str: ...
```

[Internationalization](./internationalization.md)

## Common Types

```python { .api }
# Type aliases used throughout the library
NumberOrString = float | str

# Datetime/time types
import datetime as dt
from typing import Any, Iterable
import os
import gettext
```