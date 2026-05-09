# Exception Handling and Error Management

Comprehensive exception classes for handling various error conditions in command-line applications, with proper error reporting and user-friendly messages. Click's exception system provides structured error handling that maintains good user experience.

## Capabilities

### Base Exception Classes

Foundation exception classes that provide structured error handling for CLI applications.

```python { .api }
class ClickException(Exception):
    """
    Base exception that Click can handle and show to user.
    
    Attributes:
    - exit_code: int, exit code when exception causes termination (default: 1)
    - message: str, error message to display
    """
    
    def __init__(self, message):
        """
        Create a Click exception.
        
        Parameters:
        - message: Error message to display to user
        """
    
    def format_message(self):
        """Format the error message for display."""
    
    def show(self, file=None):
        """
        Display the error message to user.
        
        Parameters:
        - file: File object to write to (defaults to stderr)
        """

class UsageError(ClickException):
    """
    Exception that signals incorrect usage and aborts further handling.
    
    Attributes:
    - exit_code: int, exit code (default: 2)
    - ctx: Context, context where error occurred
    - cmd: Command, command that caused the error
    """
    
    def __init__(self, message, ctx=None):
        """
        Create a usage error.
        
        Parameters:
        - message: Error message
        - ctx: Context where error occurred
        """
    
    def show(self, file=None):
        """Show error with usage information."""
```

**Usage Examples:**

```python
@click.command()
@click.argument('filename')
def process_file(filename):
    """Process a file."""
    import os
    
    if not os.path.exists(filename):
        raise click.ClickException(f'File "{filename}" does not exist.')
    
    if not filename.endswith('.txt'):
        raise click.UsageError('Only .txt files are supported.')
    
    click.echo(f'Processing {filename}...')

# Custom exception handling
@click.command()
@click.pass_context
def custom_error(ctx):
    """Demonstrate custom error handling."""
    try:
        # Some operation that might fail
        result = risky_operation()
    except SomeError as e:
        ctx.fail(f'Operation failed: {e}')
```

### Parameter-Related Exceptions

Exceptions for handling parameter validation and parsing errors.

```python { .api }
class BadParameter(UsageError):
    """
    Exception for bad parameter values.
    
    Attributes:
    - param: Parameter, parameter object that caused error
    - param_hint: str, alternative parameter name for error display
    """
    
    def __init__(self, message, ctx=None, param=None, param_hint=None):
        """
        Create a bad parameter error.
        
        Parameters:
        - message: Error message
        - ctx: Current context
        - param: Parameter that caused error
        - param_hint: Alternative name for parameter in error
        """

class MissingParameter(BadParameter):
    """
    Exception raised when a required parameter is missing.
    
    Attributes:
    - param_type: str, type of parameter ("option", "argument", "parameter")
    """
    
    def __init__(self, message=None, ctx=None, param=None, param_hint=None, param_type=None):
        """
        Create a missing parameter error.
        
        Parameters:
        - message: Custom error message
        - ctx: Current context
        - param: Missing parameter
        - param_hint: Alternative parameter name
        - param_type: Type of parameter for error message
        """

class BadArgumentUsage(UsageError):
    """Exception for incorrect argument usage."""

class BadOptionUsage(UsageError):
    """
    Exception for incorrect option usage.
    
    Attributes:
    - option_name: str, name of the incorrectly used option
    """
    
    def __init__(self, option_name, message, ctx=None):
        """
        Create a bad option usage error.
        
        Parameters:
        - option_name: Name of the problematic option
        - message: Error message
        - ctx: Current context
        """

class NoSuchOption(UsageError):
    """
    Exception when an option doesn't exist.
    
    Attributes:
    - option_name: str, name of the invalid option
    - possibilities: list, suggested alternative options
    """
    
    def __init__(self, option_name, message=None, possibilities=None, ctx=None):
        """
        Create a no such option error.
        
        Parameters:
        - option_name: Name of invalid option
        - message: Custom error message
        - possibilities: List of suggested alternatives
        - ctx: Current context
        """
```

**Usage Examples:**

```python
# Custom parameter validation
def validate_email(ctx, param, value):
    """Validate email parameter."""
    import re
    if not re.match(r'^[^@]+@[^@]+\.[^@]+$', value):
        raise click.BadParameter('Invalid email format', ctx, param)
    return value

@click.command()
@click.option('--email', callback=validate_email, required=True)
def send_email(email):
    """Send email with validation."""
    click.echo(f'Sending email to {email}')

# Custom parameter type with exceptions
class PortType(click.ParamType):
    name = 'port'
    
    def convert(self, value, param, ctx):
        try:
            port = int(value)
        except ValueError:
            self.fail(f'{value} is not a valid integer', param, ctx)
        
        if port < 1 or port > 65535:
            self.fail(f'{port} is not in valid range 1-65535', param, ctx)
        
        return port

@click.command()
@click.option('--port', type=PortType(), default=8080)
def start_server(port):
    """Start server with port validation."""
    click.echo(f'Starting server on port {port}')

# Handling missing parameters
@click.command()
@click.option('--config', required=True, 
              help='Configuration file (required)')
def run_with_config(config):
    """Command requiring configuration."""
    click.echo(f'Using config: {config}')
```

### File-Related Exceptions

Exceptions for file operations and I/O errors.

```python { .api }
class FileError(ClickException):
    """
    Exception when a file cannot be opened.
    
    Attributes:
    - ui_filename: str, formatted filename for display
    - filename: str, original filename
    """
    
    def __init__(self, filename, hint=None):
        """
        Create a file error.
        
        Parameters:
        - filename: Name of problematic file
        - hint: Additional hint for the error
        """
    
    def format_message(self):
        """Format error message with filename."""
```

**Usage Examples:**

```python
@click.command()
@click.argument('input_file', type=click.File('r'))
@click.argument('output_file', type=click.File('w'))
def copy_file(input_file, output_file):
    """Copy file with error handling."""
    try:
        content = input_file.read()
        output_file.write(content)
        click.echo('File copied successfully')
    except IOError as e:
        raise click.FileError(input_file.name, str(e))

# Manual file handling with exceptions
@click.command()
@click.argument('filename')
def read_file(filename):
    """Read file with custom error handling."""
    try:
        with open(filename, 'r') as f:
            content = f.read()
            click.echo(content)
    except FileNotFoundError:
        raise click.FileError(filename, 'File not found')
    except PermissionError:
        raise click.FileError(filename, 'Permission denied')
    except IOError as e:
        raise click.FileError(filename, str(e))
```

### Control Flow Exceptions

Special exceptions for controlling application flow and termination.

```python { .api }
class Abort(RuntimeError):
    """
    Internal exception to signal that Click should abort.
    Used internally by Click for graceful termination.
    """

class Exit(RuntimeError):
    """
    Exception that indicates the application should exit with a status code.
    
    Attributes:
    - exit_code: int, status code to exit with
    """
    
    def __init__(self, code=0):
        """
        Create an exit exception.
        
        Parameters:
        - code: Exit status code (0 for success)
        """
```

**Usage Examples:**

```python
@click.command()
@click.option('--force', is_flag=True, help='Force operation')
def dangerous_operation(force):
    """Perform dangerous operation."""
    if not force:
        click.echo('This is a dangerous operation!')
        if not click.confirm('Are you sure?'):
            raise click.Abort()
    
    click.echo('Performing dangerous operation...')

@click.command()
@click.argument('exit_code', type=int, default=0)
def exit_with_code(exit_code):
    """Exit with specified code."""
    click.echo(f'Exiting with code {exit_code}')
    raise click.Exit(exit_code)

# Context methods for error handling
@click.command()
@click.pass_context
def context_errors(ctx):
    """Demonstrate context error methods."""
    
    # ctx.fail() raises UsageError
    if some_condition:
        ctx.fail('Something went wrong with usage')
    
    # ctx.abort() raises Abort
    if another_condition:
        ctx.abort()
    
    # ctx.exit() raises Exit
    if success_condition:
        ctx.exit(0)
```

### Exception Handling Best Practices

Comprehensive examples showing proper exception handling patterns in Click applications.

```python
# Comprehensive error handling example
@click.group()
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """Main CLI with error handling."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose

@cli.command()
@click.argument('config_file', type=click.Path(exists=True, readable=True))
@click.option('--output', type=click.Path(writable=True), required=True)
@click.pass_context
def process_config(ctx, config_file, output):
    """Process configuration file with comprehensive error handling."""
    import json
    import yaml
    
    try:
        # Try to determine file format
        if config_file.endswith('.json'):
            with open(config_file, 'r') as f:
                config = json.load(f)
        elif config_file.endswith(('.yml', '.yaml')):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise click.UsageError(
                'Configuration file must be JSON (.json) or YAML (.yml/.yaml)'
            )
    
    except json.JSONDecodeError as e:
        raise click.FileError(config_file, f'Invalid JSON: {e}')
    except yaml.YAMLError as e:
        raise click.FileError(config_file, f'Invalid YAML: {e}')
    except PermissionError:
        raise click.FileError(config_file, 'Permission denied')
    
    # Validate configuration
    required_keys = ['name', 'version']
    for key in required_keys:
        if key not in config:
            raise click.BadParameter(
                f'Missing required key: {key}',
                ctx=ctx,
                param_hint='config_file'
            )
    
    # Process and write output
    try:
        processed = process_configuration(config)
        with open(output, 'w') as f:
            json.dump(processed, f, indent=2)
        
        if ctx.obj['verbose']:
            click.echo(f'Successfully processed {config_file} -> {output}')
    
    except IOError as e:
        raise click.FileError(output, f'Cannot write output: {e}')
    except Exception as e:
        # Catch-all for unexpected errors
        raise click.ClickException(f'Unexpected error: {e}')

# Error recovery example
@cli.command()
@click.option('--retry', default=3, help='Number of retries')
@click.pass_context
def unreliable_operation(ctx, retry):
    """Operation with retry logic."""
    import random
    import time
    
    for attempt in range(retry + 1):
        try:
            if random.random() < 0.7:  # 70% chance of failure
                raise Exception('Random failure')
            
            click.echo('Operation succeeded!')
            return
        
        except Exception as e:
            if attempt < retry:
                if ctx.obj['verbose']:
                    click.echo(f'Attempt {attempt + 1} failed: {e}')
                    click.echo(f'Retrying in 1 second...')
                time.sleep(1)
            else:
                raise click.ClickException(
                    f'Operation failed after {retry + 1} attempts: {e}'
                )
```