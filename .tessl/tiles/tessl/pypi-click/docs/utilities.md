# Utilities and Helper Functions

Utility functions for file handling, stream management, text formatting, and other common CLI application needs. Click provides a comprehensive set of helper functions that simplify common command-line application tasks.

## Capabilities

### File and Stream Operations

Functions for handling files, streams, and I/O operations with enhanced features.

```python { .api }
def open_file(filename, mode="r", encoding=None, errors="strict", lazy=False, atomic=False):
    """
    Open file with special handling for "-" (stdin/stdout).
    
    Parameters:
    - filename: File path or "-" for stdin/stdout
    - mode: File open mode ('r', 'w', 'a', 'rb', 'wb', etc.)
    - encoding: Text encoding (None for binary modes)
    - errors: Error handling strategy ('strict', 'ignore', 'replace')
    - lazy: Open file lazily on first access
    - atomic: Write to temporary file then move (for write modes)
    
    Returns:
    File object or LazyFile/KeepOpenFile wrapper
    """

def get_binary_stream(name):
    """
    Get system binary stream.
    
    Parameters:
    - name: Stream name ("stdin", "stdout", "stderr")
    
    Returns:
    Binary stream object
    """

def get_text_stream(name, encoding=None, errors="strict"):
    """
    Get system text stream.
    
    Parameters:
    - name: Stream name ("stdin", "stdout", "stderr")
    - encoding: Text encoding (None for system default)
    - errors: Error handling strategy
    
    Returns:
    Text stream object
    """
```

**Usage Examples:**

```python
@click.command()
@click.option('--input', default='-', help='Input file (- for stdin)')
@click.option('--output', default='-', help='Output file (- for stdout)')
def process_streams(input, output):
    """Process data from input to output."""
    with click.open_file(input, 'r') as infile:
        with click.open_file(output, 'w') as outfile:
            for line in infile:
                processed = line.upper()
                outfile.write(processed)

@click.command()
@click.option('--output', type=click.Path())
def atomic_write(output):
    """Write file atomically."""
    with click.open_file(output, 'w', atomic=True) as f:
        f.write('This will be written atomically\n')
        f.write('Either all content is written or none\n')
    click.echo(f'File written atomically to {output}')

@click.command()
def stream_examples():
    """Demonstrate stream operations."""
    # Get system streams
    stdout = click.get_text_stream('stdout')
    stderr = click.get_text_stream('stderr')
    
    stdout.write('This goes to stdout\n')
    stderr.write('This goes to stderr\n')
    
    # Binary streams
    binary_stdout = click.get_binary_stream('stdout')
    binary_stdout.write(b'Binary data to stdout\n')
```

### Directory and Path Utilities

Functions for handling application directories and path operations.

```python { .api }
def get_app_dir(app_name, roaming=True, force_posix=False):
    """
    Get application configuration directory.
    
    Parameters:
    - app_name: Name of the application
    - roaming: Use roaming directory on Windows
    - force_posix: Force POSIX-style paths even on Windows
    
    Returns:
    Path to application directory
    
    Platform behavior:
    - Windows: %APPDATA%\\app_name (roaming) or %LOCALAPPDATA%\\app_name
    - macOS: ~/Library/Application Support/app_name
    - Linux: ~/.config/app_name (or $XDG_CONFIG_HOME/app_name)
    """

def format_filename(filename, shorten=False):
    """
    Format filename for display.
    
    Parameters:
    - filename: Filename to format
    - shorten: Shorten very long filenames
    
    Returns:
    Formatted filename string
    """
```

**Usage Examples:**

```python
import os

@click.command()
@click.argument('app_name')
def show_app_dir(app_name):
    """Show application directory for given app name."""
    app_dir = click.get_app_dir(app_name)
    click.echo(f'App directory: {app_dir}')
    
    # Create directory if it doesn't exist
    os.makedirs(app_dir, exist_ok=True)
    
    # Create a config file
    config_file = os.path.join(app_dir, 'config.ini')
    with open(config_file, 'w') as f:
        f.write(f'[{app_name}]\n')
        f.write('debug = false\n')
    
    click.echo(f'Created config: {click.format_filename(config_file)}')

@click.command()
def config_example():
    """Example of using app directory for configuration."""
    app_dir = click.get_app_dir('myapp')
    config_path = os.path.join(app_dir, 'settings.json')
    
    # Ensure directory exists
    os.makedirs(app_dir, exist_ok=True)
    
    if os.path.exists(config_path):
        click.echo(f'Loading config from {click.format_filename(config_path)}')
    else:
        click.echo(f'Creating default config at {click.format_filename(config_path)}')
        import json
        default_config = {
            'debug': False,
            'log_level': 'INFO',
            'max_retries': 3
        }
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
```

### Context Management

Functions for accessing and managing Click contexts.

```python { .api }
def get_current_context(silent=False):
    """
    Get current Click context from thread-local storage.
    
    Parameters:
    - silent: Return None instead of raising error if no context
    
    Returns:
    Current Context object or None
    
    Raises:
    RuntimeError: If no context is available and silent=False
    """
```

**Usage Examples:**

```python
def helper_function():
    """Helper function that needs access to current context."""
    ctx = click.get_current_context(silent=True)
    if ctx:
        return ctx.obj.get('debug', False)
    return False

@click.group()
@click.option('--debug', is_flag=True)
@click.pass_context
def main_cli(ctx, debug):
    """Main CLI with shared context."""
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug

@main_cli.command()
def subcommand():
    """Subcommand that uses helper function."""
    debug_mode = helper_function()
    click.echo(f'Debug mode: {debug_mode}')

# Context-aware logging
def log_message(message, level='INFO'):
    """Log message with context-aware formatting."""
    ctx = click.get_current_context(silent=True)
    prefix = ''
    
    if ctx and ctx.obj and ctx.obj.get('verbose'):
        prefix = f'[{level}] {ctx.info_name}: '
    
    click.echo(f'{prefix}{message}')

@main_cli.command()
@click.option('--verbose', is_flag=True)
@click.pass_context
def logging_example(ctx, verbose):
    """Example of context-aware logging."""
    ctx.obj['verbose'] = verbose
    
    log_message('Starting operation')
    log_message('Processing data', 'DEBUG')
    log_message('Operation complete')
```

### Text and String Utilities

Helper functions for text processing and string manipulation.

```python { .api }
# Internal utility functions (not directly exported but used by Click)
def make_str(value):
    """Convert value to valid string representation."""

def make_default_short_help(help, max_length=45):
    """Create condensed help string from longer help text."""

def safecall(func):
    """Wrap function to swallow exceptions silently."""
```

**Usage Examples:**

```python
# Custom help formatting
def create_short_help(long_help, max_len=40):
    """Create short help from long help text."""
    if not long_help:
        return None
    
    # Simple truncation with ellipsis
    if len(long_help) <= max_len:
        return long_help
    
    # Find last space before max length
    truncated = long_help[:max_len]
    last_space = truncated.rfind(' ')
    
    if last_space > max_len * 0.7:  # If space is reasonably close
        return truncated[:last_space] + '...'
    else:
        return truncated + '...'

@click.group()
def cli_with_short_help():
    """Main CLI group."""
    pass

@cli_with_short_help.command()
def long_command():
    """This is a very long help text that describes in great detail what this command does and provides comprehensive information about its purpose and usage patterns."""
    click.echo('Command executed')

# Set short help manually
long_command.short_help = create_short_help(long_command.help)
```

### File Wrapper Classes

Special file classes for advanced file handling scenarios.

```python { .api }
class LazyFile:
    """
    File wrapper that opens the file lazily on first access.
    Used internally by click.open_file() with lazy=True.
    """

class KeepOpenFile:
    """
    File wrapper that prevents closing when used as context manager.
    Useful for stdin/stdout that shouldn't be closed.
    """

class PacifyFlushWrapper:
    """
    Wrapper that suppresses BrokenPipeError on flush().
    Handles cases where output pipe is closed by receiver.
    """
```

**Usage Examples:**

```python
# Custom file handling with error recovery
@click.command()
@click.option('--input', type=click.File('r'))
@click.option('--output', type=click.File('w'))
def robust_copy(input, output):
    """Copy with error handling for broken pipes."""
    try:
        for line in input:
            output.write(line.upper())
            output.flush()  # May raise BrokenPipeError
    except BrokenPipeError:
        # Handle gracefully - receiver closed pipe
        click.echo('Output pipe closed by receiver', err=True)
    except KeyboardInterrupt:
        click.echo('\nOperation cancelled', err=True)
        raise click.Abort()

# Working with lazy files
@click.command()
@click.option('--config', type=click.Path())
def lazy_config(config):
    """Example of lazy file loading."""
    if config:
        # File won't be opened until first access
        with click.open_file(config, 'r', lazy=True) as f:
            if some_condition():
                # File is opened here on first read
                content = f.read()
                click.echo(f'Config loaded: {len(content)} bytes')
            # If condition was false, file never gets opened

# Atomic file operations
@click.command()
@click.argument('output_file')
def atomic_json_write(output_file):
    """Write JSON atomically to prevent corruption."""
    import json
    import time
    
    data = {
        'timestamp': time.time(),
        'status': 'processing',
        'items': list(range(1000))
    }
    
    # Atomic write - either succeeds completely or fails completely
    with click.open_file(output_file, 'w', atomic=True) as f:
        json.dump(data, f, indent=2)
        # If any error occurs here, original file is unchanged
    
    click.echo(f'Data written atomically to {output_file}')
```

### Text Formatting and Help Generation

Functions and classes for formatting help text and wrapping text content.

```python { .api }
class HelpFormatter:
    def __init__(self, indent_increment=2, width=None, max_width=None):
        """
        Format text-based help pages for commands.
        
        Parameters:
        - indent_increment: Additional increment for each indentation level
        - width: Text width (defaults to terminal width, max 78)
        - max_width: Maximum width allowed
        """
    
    def write(self, string):
        """Write a string into the internal buffer."""
    
    def indent(self):
        """Increase the indentation level."""
    
    def dedent(self):
        """Decrease the indentation level."""
    
    def write_usage(self, prog, args="", prefix=None):
        """
        Write a usage line into the buffer.
        
        Parameters:
        - prog: Program name
        - args: Whitespace separated list of arguments
        - prefix: Prefix for the first line (defaults to "Usage: ")
        """
    
    def write_heading(self, heading):
        """Write a heading into the buffer."""
    
    def write_paragraph(self):
        """Write a paragraph break into the buffer."""
    
    def write_text(self, text):
        """Write re-indented text with paragraph preservation."""
    
    def write_dl(self, rows, col_max=30, col_spacing=2):
        """
        Write a definition list (used for options and commands).
        
        Parameters:
        - rows: List of (term, description) tuples
        - col_max: Maximum width of first column
        - col_spacing: Spaces between columns
        """
    
    def section(self, name):
        """Context manager that writes heading and indents."""
    
    def indentation(self):
        """Context manager that increases indentation."""
    
    def getvalue(self):
        """Return the buffer contents as string."""

def wrap_text(text, width=78, initial_indent="", subsequent_indent="", preserve_paragraphs=False):
    """
    Intelligent text wrapping with paragraph handling.
    
    Parameters:
    - text: Text to wrap
    - width: Maximum line width
    - initial_indent: Indent for first line
    - subsequent_indent: Indent for continuation lines
    - preserve_paragraphs: Handle paragraphs intelligently (separated by empty lines)
    
    Returns:
    Wrapped text string
    
    Notes:
    - When preserve_paragraphs=True, paragraphs are defined by two empty lines
    - Lines starting with \\b character are not rewrapped
    - Handles mixed indentation levels within paragraphs
    """
```

**Usage Examples:**

```python
@click.command()
def formatting_examples():
    """Demonstrate text formatting utilities."""
    # Create a help formatter
    formatter = click.HelpFormatter(indent_increment=4, width=60)
    
    # Write structured help content
    formatter.write_heading('Options')
    formatter.indent()
    formatter.write_dl([
        ('--verbose, -v', 'Enable verbose output'),
        ('--config FILE', 'Configuration file path'),
        ('--help', 'Show this message and exit')
    ])
    formatter.dedent()
    
    # Get formatted result
    help_text = formatter.getvalue()
    click.echo(help_text)
    
    # Text wrapping examples
    long_text = "This is a very long line of text that needs to be wrapped to fit within a reasonable line length for display in terminal applications."
    
    wrapped = click.wrap_text(long_text, width=40)
    click.echo("Basic wrapping:")
    click.echo(wrapped)
    
    # With indentation
    indented = click.wrap_text(long_text, width=40, 
                              initial_indent="  * ",
                              subsequent_indent="    ")
    click.echo("\nWith indentation:")
    click.echo(indented)
    
    # Paragraph preservation
    paragraphs = """First paragraph with some text.

Second paragraph after empty line.

    Indented paragraph that should
    maintain its indentation level."""
    
    preserved = click.wrap_text(paragraphs, width=30, preserve_paragraphs=True)
    click.echo("\nParagraph preservation:")
    click.echo(preserved)

# Custom help formatter
class CustomHelpFormatter(click.HelpFormatter):
    """Custom help formatter with different styling."""
    
    def write_heading(self, heading):
        """Write heading with custom styling."""
        self.write(f"{'':>{self.current_indent}}=== {heading.upper()} ===\n")

@click.command()
def custom_help_example():
    """Command with custom help formatting."""
    pass

# Override the formatter class
custom_help_example.context_settings = {'formatter_class': CustomHelpFormatter}
```

### Integration Utilities

Functions for integrating Click with other systems and frameworks.

```python
# Environment integration
@click.command()
@click.option('--config-dir', 
              envvar='MY_APP_CONFIG_DIR',
              default=lambda: click.get_app_dir('myapp'),
              help='Configuration directory')
def env_integration(config_dir):
    """Example of environment variable integration."""
    click.echo(f'Using config directory: {config_dir}')

# Shell integration
@click.command()
@click.option('--shell-complete', is_flag=True, hidden=True)
def shell_integration(shell_complete):
    """Command with shell completion support."""
    if shell_complete:
        # This would be handled by Click's completion system
        return
    
    click.echo('Command executed')

# Testing utilities integration
def create_test_context(command, args=None):
    """Create context for testing Click commands."""
    if args is None:
        args = []
    
    return command.make_context('test', args)

# Example test helper
def test_command_helper():
    """Helper for testing Click commands."""
    @click.command()
    @click.argument('name')
    @click.option('--greeting', default='Hello')
    def greet(name, greeting):
        click.echo(f'{greeting}, {name}!')
    
    # Create test context
    ctx = create_test_context(greet, ['World', '--greeting', 'Hi'])
    
    # This would be used in actual testing
    return ctx
```