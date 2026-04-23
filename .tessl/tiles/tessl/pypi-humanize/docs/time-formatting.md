# Time and Date Formatting

Natural language time and date formatting utilities that convert datetime objects, timedeltas, and time values into human-readable expressions with support for relative time, precision control, and internationalization.

## Capabilities

### Natural Time

Converts timestamps, datetime objects, or time differences into natural language expressions with automatic tense detection and configurable precision.

```python { .api }
def naturaltime(
    value: dt.datetime | dt.timedelta | float,
    future: bool = False,
    months: bool = True,
    minimum_unit: str = "seconds",
    when: dt.datetime | None = None
) -> str:
    """
    Return a natural representation of time in human-readable format.
    
    Args:
        value: datetime, timedelta, or number of seconds
        future: Force future tense for numeric inputs (ignored for datetime/timedelta)
        months: Use months for year calculations (based on 30.5 days)
        minimum_unit: Lowest unit to display ("seconds", "milliseconds", "microseconds")
        when: Reference point in time (defaults to current time)
    
    Returns:
        Natural time representation with appropriate tense
    
    Examples:
        >>> naturaltime(dt.datetime.now() - dt.timedelta(minutes=30))
        '30 minutes ago'
        >>> naturaltime(3600, future=True)
        'an hour from now'
        >>> naturaltime(dt.datetime.now() + dt.timedelta(days=1))
        'a day from now'
    """
```

### Natural Delta

Converts time differences into natural language without tense, focusing purely on the duration or time span.

```python { .api }
def naturaldelta(
    value: dt.timedelta | float,
    months: bool = True,
    minimum_unit: str = "seconds"
) -> str:
    """
    Return a natural representation of a timedelta without tense.
    
    Args:
        value: timedelta or number of seconds
        months: Use months for calculations between years
        minimum_unit: Lowest unit to display
    
    Returns:
        Natural duration string without tense indicators
    
    Examples:
        >>> naturaldelta(dt.timedelta(minutes=30))
        '30 minutes'
        >>> naturaldelta(3661)
        'an hour'
        >>> naturaldelta(dt.timedelta(days=400))
        'a year'
    """
```

### Natural Day

Converts dates to natural expressions like "today", "yesterday", "tomorrow", or formatted date strings for other dates.

```python { .api }
def naturalday(value: dt.date | dt.datetime, format: str = "%b %d") -> str:
    """
    Return a natural day representation.
    
    Args:
        value: date or datetime object
        format: strftime format for non-relative dates
    
    Returns:
        "today", "yesterday", "tomorrow", or formatted date string
    
    Examples:
        >>> naturalday(dt.date.today())
        'today'
        >>> naturalday(dt.date.today() - dt.timedelta(days=1))
        'yesterday'
        >>> naturalday(dt.date.today() + dt.timedelta(days=1))
        'tomorrow'
    """
```

### Natural Date

Similar to naturalday but includes the year for dates that are more than approximately five months away from today.

```python { .api }
def naturaldate(value: dt.date | dt.datetime) -> str:
    """
    Like naturalday, but append year for dates >5 months away.
    
    Args:
        value: date or datetime object
    
    Returns:
        Natural date string with year when appropriate
    
    Examples:
        >>> naturaldate(dt.date.today())
        'today'
        >>> naturaldate(dt.date(2020, 1, 1))  # if far in past/future
        'Jan 01 2020'
    """
```

### Precise Delta

Provides precise timedelta representation with multiple units, customizable precision, and the ability to suppress specific units.

```python { .api }
def precisedelta(
    value: dt.timedelta | float | None,
    minimum_unit: str = "seconds",
    suppress: Iterable[str] = (),
    format: str = "%0.2f"
) -> str:
    """
    Return a precise representation of a timedelta.
    
    Args:
        value: timedelta, number of seconds, or None
        minimum_unit: Smallest unit to display
        suppress: List of units to exclude from output
        format: Format string for fractional parts
    
    Returns:
        Precise multi-unit time representation
    
    Examples:
        >>> precisedelta(dt.timedelta(seconds=3633, days=2, microseconds=123000))
        '2 days, 1 hour and 33.12 seconds'
        >>> precisedelta(dt.timedelta(seconds=90), suppress=['seconds'])
        '1.50 minutes'
        >>> precisedelta(delta, minimum_unit="microseconds")
        '2 days, 1 hour, 33 seconds and 123 milliseconds'
    """
```

## Time Units

```python { .api }
from enum import Enum

class Unit(Enum):
    """Time units for precise delta calculations."""
    MICROSECONDS = 0
    MILLISECONDS = 1
    SECONDS = 2
    MINUTES = 3
    HOURS = 4
    DAYS = 5
    MONTHS = 6
    YEARS = 7
```

## Usage Examples

### Basic Time Formatting

```python
import datetime as dt
import humanize

# Current time references
now = dt.datetime.now()
past = now - dt.timedelta(hours=2, minutes=30)
future = now + dt.timedelta(days=3)

# Natural time expressions
print(humanize.naturaltime(past))    # "2 hours ago"
print(humanize.naturaltime(future))  # "3 days from now"

# Pure duration without tense
print(humanize.naturaldelta(dt.timedelta(hours=2, minutes=30)))  # "2 hours"

# Date formatting
today = dt.date.today()
print(humanize.naturalday(today))  # "today"
print(humanize.naturalday(today - dt.timedelta(days=1)))  # "yesterday"
```

### Precise Time Calculations

```python
import datetime as dt
import humanize

# Complex time period
delta = dt.timedelta(days=2, hours=1, minutes=33, seconds=12, microseconds=123000)

# Default precision
print(humanize.precisedelta(delta))
# "2 days, 1 hour, 33 minutes and 12.12 seconds"

# Higher precision
print(humanize.precisedelta(delta, minimum_unit="microseconds"))  
# "2 days, 1 hour, 33 minutes, 12 seconds and 123 milliseconds"

# Suppressing units
print(humanize.precisedelta(delta, suppress=['minutes', 'seconds']))
# "2 days and 1.55 hours"
```

### Minimum Unit Control

```python
import datetime as dt
import humanize

short_time = dt.timedelta(milliseconds=500)

print(humanize.naturaldelta(short_time, minimum_unit="milliseconds"))
# "500 milliseconds"

print(humanize.naturaldelta(short_time, minimum_unit="seconds"))
# "a moment"
```

## Error Handling

Time formatting functions handle edge cases gracefully:

- Invalid date/datetime objects return their string representation
- Non-convertible values return the original value as a string
- Overflow errors in date calculations are handled safely
- Timezone-aware datetimes are converted to naive datetimes automatically