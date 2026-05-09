# Parameter Types and Validation

Rich type system for command-line parameter validation and conversion, including built-in types for common data formats and custom type creation. Click's type system ensures user input is properly validated and converted to the expected Python types.

## Capabilities

### Base Parameter Type

Foundation class for all parameter types with validation and conversion capabilities.

```python { .api }
class ParamType:
    """
    Base class for all parameter types.
    
    Attributes:
    - name: str, descriptive name of the type
    - is_composite: bool, whether type expects multiple values
    - arity: int, number of arguments expected
    - envvar_list_splitter: str, character to split env var lists
    """
    
    def convert(self, value, param, ctx):
        """
        Convert value to the correct type.
        
        Parameters:
        - value: Input value to convert
        - param: Parameter instance
        - ctx: Current context
        
        Returns:
        Converted value
        
        Raises:
        BadParameter: If conversion fails
        """
    
    def fail(self, message, param=None, ctx=None):
        """Raise BadParameter with message."""
    
    def get_metavar(self, param, ctx):
        """Get metavar string for help display."""
    
    def get_missing_message(self, param, ctx):
        """Get message for missing required parameter."""
    
    def shell_complete(self, ctx, param, incomplete):
        """Return shell completions for this type."""
```

### Built-in Simple Types

Basic parameter types for common data formats.

```python { .api }
STRING: StringParamType
"""String parameter type that handles encoding and conversion."""

INT: IntParamType
"""Integer parameter type with validation."""

FLOAT: FloatParamType
"""Float parameter type with validation."""

BOOL: BoolParamType
"""Boolean parameter type supporting various string representations."""

UUID: UUIDParameterType
"""UUID parameter type that converts to uuid.UUID objects."""

UNPROCESSED: UnprocessedParamType
"""Unprocessed type that passes values through without conversion."""
```

**Usage Examples:**

```python
@click.command()
@click.option('--name', type=click.STRING, help='Your name')
@click.option('--age', type=click.INT, help='Your age')
@click.option('--height', type=click.FLOAT, help='Height in meters')
@click.option('--verbose', type=click.BOOL, help='Enable verbose output')
@click.option('--session-id', type=click.UUID, help='Session UUID')
def profile(name, age, height, verbose, session_id):
    """Update user profile."""
    click.echo(f'Name: {name} (type: {type(name).__name__})')
    click.echo(f'Age: {age} (type: {type(age).__name__})')
    click.echo(f'Height: {height}m (type: {type(height).__name__})')
    click.echo(f'Verbose: {verbose} (type: {type(verbose).__name__})')
    click.echo(f'Session: {session_id} (type: {type(session_id).__name__})')
```

### Choice Types

Restrict parameter values to a predefined set of options.

```python { .api }
class Choice(ParamType):
    def __init__(self, choices, case_sensitive=True):
        """
        Parameter type that restricts values to a fixed set.
        
        Parameters:
        - choices: Sequence of valid choice strings
        - case_sensitive: Whether matching is case sensitive
        """
    
    def get_metavar(self, param, ctx):
        """Return metavar showing available choices."""
    
    def get_missing_message(self, param, ctx):
        """Return message showing available choices."""
    
    def shell_complete(self, ctx, param, incomplete):
        """Complete with matching choices."""
```

**Usage Examples:**

```python
@click.command()
@click.option('--log-level', 
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO',
              help='Set logging level')
@click.option('--format', 
              type=click.Choice(['json', 'xml', 'csv'], case_sensitive=False),
              help='Output format')
def export(log_level, format):
    """Export data with specified format."""
    click.echo(f'Log level: {log_level}')
    click.echo(f'Format: {format}')
```

### Range Types

Numeric types with minimum and maximum constraints.

```python { .api }
class IntRange(IntParamType):
    def __init__(self, min=None, max=None, min_open=False, max_open=False, clamp=False):
        """
        Integer type with range constraints.
        
        Parameters:
        - min: Minimum value (inclusive by default)
        - max: Maximum value (inclusive by default)
        - min_open: Whether minimum bound is exclusive
        - max_open: Whether maximum bound is exclusive
        - clamp: Clamp out-of-range values instead of failing
        """

class FloatRange(FloatParamType):
    def __init__(self, min=None, max=None, min_open=False, max_open=False, clamp=False):
        """
        Float type with range constraints.
        
        Parameters:
        - min: Minimum value (inclusive by default)
        - max: Maximum value (inclusive by default)
        - min_open: Whether minimum bound is exclusive
        - max_open: Whether maximum bound is exclusive
        - clamp: Clamp out-of-range values instead of failing
        """
```

**Usage Examples:**

```python
@click.command()
@click.option('--port', type=click.IntRange(1, 65535), 
              help='Port number (1-65535)')
@click.option('--timeout', type=click.FloatRange(0.1, 300.0), 
              default=30.0, help='Timeout in seconds')
@click.option('--percentage', type=click.IntRange(0, 100, clamp=True),
              help='Percentage (clamped to 0-100)')
def connect(port, timeout, percentage):
    """Connect with validation."""
    click.echo(f'Port: {port}')
    click.echo(f'Timeout: {timeout}s')
    click.echo(f'Percentage: {percentage}%')
```

### Date and Time Types

Handle date and time parsing with multiple format support.

```python { .api }
class DateTime(ParamType):
    def __init__(self, formats=None):
        """
        DateTime parameter type.
        
        Parameters:
        - formats: List of strptime format strings to try
                  (defaults to common ISO formats)
        """
    
    def get_metavar(self, param, ctx):
        """Return metavar showing expected formats."""
```

**Usage Examples:**

```python
@click.command()
@click.option('--start-date', 
              type=click.DateTime(['%Y-%m-%d', '%m/%d/%Y']),
              help='Start date (YYYY-MM-DD or MM/DD/YYYY)')
@click.option('--timestamp',
              type=click.DateTime(['%Y-%m-%d %H:%M:%S']),
              help='Timestamp (YYYY-MM-DD HH:MM:SS)')
def schedule(start_date, timestamp):
    """Schedule operation."""
    click.echo(f'Start date: {start_date}')
    click.echo(f'Timestamp: {timestamp}')
    click.echo(f'Date type: {type(start_date).__name__}')
```

### File and Path Types

Handle file operations and path validation with comprehensive options.

```python { .api }
class File(ParamType):
    def __init__(self, mode="r", encoding=None, errors="strict", lazy=None, atomic=False):
        """
        File parameter type for reading/writing files.
        
        Parameters:
        - mode: File open mode ('r', 'w', 'a', 'rb', 'wb', etc.)
        - encoding: Text encoding (None for binary modes)
        - errors: Error handling strategy ('strict', 'ignore', 'replace')
        - lazy: Open file lazily on first access
        - atomic: Write to temporary file then move (for write modes)
        """

class Path(ParamType):
    def __init__(self, exists=False, file_okay=True, dir_okay=True, 
                 writable=False, readable=True, resolve_path=False, 
                 allow_dash=False, path_type=None, executable=False):
        """
        Path parameter type with validation.
        
        Parameters:
        - exists: Path must exist
        - file_okay: Allow files
        - dir_okay: Allow directories  
        - writable: Path must be writable
        - readable: Path must be readable
        - resolve_path: Make absolute and resolve symlinks
        - allow_dash: Allow '-' for stdin/stdout
        - path_type: Convert to this type (str, bytes, pathlib.Path)
        - executable: Path must be executable
        """
    
    def shell_complete(self, ctx, param, incomplete):
        """Provide path completion."""
```

**Usage Examples:**

```python
@click.command()
@click.option('--input', type=click.File('r'), help='Input file')
@click.option('--output', type=click.File('w'), help='Output file')
@click.option('--config', 
              type=click.Path(exists=True, readable=True),
              help='Configuration file path')
@click.option('--log-dir',
              type=click.Path(exists=True, dir_okay=True, file_okay=False, writable=True),
              help='Log directory')
def process(input, output, config, log_dir):
    """Process files."""
    content = input.read()
    output.write(f'Processed: {content}')
    click.echo(f'Config: {config}')
    click.echo(f'Log directory: {log_dir}')

# Using pathlib.Path
from pathlib import Path

@click.command()
@click.option('--workspace',
              type=click.Path(path_type=Path, exists=True, dir_okay=True),
              help='Workspace directory')
def workspace_cmd(workspace):
    """Work with pathlib.Path objects."""
    click.echo(f'Workspace: {workspace}')
    click.echo(f'Absolute path: {workspace.absolute()}')
    click.echo(f'Is directory: {workspace.is_dir()}')
```

### Composite Types

Handle multiple values and complex data structures.

```python { .api }
class Tuple(ParamType):
    def __init__(self, types):
        """
        Tuple parameter type for multiple typed values.
        
        Parameters:
        - types: Sequence of ParamType instances for each tuple element
        
        Attributes:
        - is_composite: True
        - arity: Length of types sequence
        """
    
    def convert(self, value, param, ctx):
        """Convert each tuple element with corresponding type."""
```

**Usage Examples:**

```python
@click.command()
@click.option('--point', 
              type=click.Tuple([click.FLOAT, click.FLOAT]),
              help='2D coordinate (x y)')
@click.option('--range',
              type=click.Tuple([click.INT, click.INT, click.STRING]),
              help='Range specification (start end unit)')
def geometry(point, range):
    """Work with tuples."""
    if point:
        x, y = point
        click.echo(f'Point: ({x}, {y})')
    
    if range:
        start, end, unit = range
        click.echo(f'Range: {start}-{end} {unit}')
```

### Custom Parameter Types

Create custom parameter types for specialized validation and conversion.

```python
# Custom email type
class EmailType(click.ParamType):
    name = "email"
    
    def convert(self, value, param, ctx):
        import re
        if not re.match(r'^[^@]+@[^@]+\.[^@]+$', value):
            self.fail(f'{value} is not a valid email address', param, ctx)
        return value

EMAIL = EmailType()

# Custom JSON type
class JSONType(click.ParamType):
    name = "json"
    
    def convert(self, value, param, ctx):
        import json
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            self.fail(f'Invalid JSON: {e}', param, ctx)

JSON = JSONType()

@click.command()
@click.option('--email', type=EMAIL, help='Email address')
@click.option('--config', type=JSON, help='JSON configuration')
def custom_types(email, config):
    """Example using custom types."""
    click.echo(f'Email: {email}')
    click.echo(f'Config: {config}')
```