# Command Definition and Decorators

Core decorators for creating commands and command groups, along with parameter decorators for handling command-line arguments and options. These decorators transform Python functions into Click commands with minimal boilerplate.

## Capabilities

### Command Creation

Transform Python functions into executable commands with automatic help generation and parameter parsing.

```python { .api }
def command(name=None, cls=None, **attrs):
    """
    Decorator that converts a function into a Click command.
    
    Parameters:
    - name: str, optional name for the command (defaults to function name)
    - cls: Command class to use (defaults to Command)
    - context_settings: dict, default Context settings
    - help: str, help text for the command
    - epilog: str, text to show after help
    - short_help: str, short help for command listings
    - add_help_option: bool, whether to add --help option
    - no_args_is_help: bool, show help when no arguments provided
    - hidden: bool, hide from help listings
    - deprecated: bool, mark as deprecated
    
    Returns:
    Decorated function as Command instance
    """

def group(name=None, cls=None, **attrs):
    """
    Decorator that converts a function into a Click command group.
    
    Parameters:
    - name: str, optional name for the group (defaults to function name)
    - cls: Group class to use (defaults to Group)
    - commands: dict, initial subcommands
    - invoke_without_command: bool, invoke group callback without subcommand
    - no_args_is_help: bool, show help when no arguments provided
    - subcommand_metavar: str, how to represent subcommands in help
    - chain: bool, allow chaining multiple subcommands
    - result_callback: callable, callback for processing results
    
    Returns:
    Decorated function as Group instance
    """
```

**Usage Examples:**

```python
@click.command()
def simple():
    """A simple command."""
    click.echo('Hello World!')

@click.group()
def cli():
    """A command group."""
    pass

@cli.command()
def subcommand():
    """A subcommand."""
    click.echo('This is a subcommand')

# Chaining commands
@click.group(chain=True)
def pipeline():
    """Pipeline processing."""
    pass

@pipeline.command()
def step1():
    click.echo('Step 1')

@pipeline.command() 
def step2():
    click.echo('Step 2')
```

### Parameter Decorators

Add command-line arguments and options to commands with automatic type conversion and validation.

```python { .api }
def argument(*param_decls, cls=None, **attrs):
    """
    Decorator to add a positional argument to a command.
    
    Parameters:
    - param_decls: parameter declarations (argument name)
    - cls: Argument class to use
    - type: parameter type for conversion
    - required: bool, whether argument is required (default True)
    - default: default value if not provided
    - callback: function to process the value
    - nargs: number of arguments to consume
    - multiple: bool, accept multiple values
    - metavar: how to display in help
    - envvar: environment variable name(s)
    - shell_complete: shell completion function
    """

def option(*param_decls, cls=None, **attrs):
    """
    Decorator to add a command-line option to a command.
    
    Parameters:
    - param_decls: option flags (e.g., '--verbose', '-v')
    - cls: Option class to use
    - type: parameter type for conversion
    - required: bool, whether option is required
    - default: default value
    - help: help text for the option
    - show_default: bool/str, show default value in help
    - prompt: bool/str, prompt for value if not provided
    - confirmation_prompt: bool, require confirmation
    - hide_input: bool, hide input when prompting
    - is_flag: bool, treat as boolean flag
    - flag_value: value when flag is present
    - multiple: bool, allow multiple values
    - count: bool, count number of times option is used
    - envvar: environment variable name(s)
    - show_envvar: bool, show env var in help
    - show_choices: bool, show choices in help
    - hidden: bool, hide from help
    """
```

**Usage Examples:**

```python
@click.command()
@click.argument('filename')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--count', default=1, help='Number of times to process')
@click.option('--output', type=click.Path(), help='Output file path')
def process(filename, verbose, count, output):
    """Process a file with options."""
    for i in range(count):
        if verbose:
            click.echo(f'Processing {filename} (iteration {i+1})')
        # Process file...

@click.command()
@click.option('--name', prompt=True, help='Your name')
@click.option('--password', prompt=True, hide_input=True)
def login(name, password):
    """Login command with prompts."""
    click.echo(f'Logging in {name}...')
```

### Special Option Decorators

Pre-configured option decorators for common CLI patterns.

```python { .api }
def confirmation_option(*param_decls, **kwargs):
    """
    Add a confirmation option (typically --yes).
    
    Parameters:
    - param_decls: option flags (defaults to '--yes')
    - expose_value: bool, expose value to callback (default False)
    - prompt: str, confirmation prompt text
    """

def password_option(*param_decls, **kwargs):
    """
    Add a password option with hidden input.
    
    Parameters:
    - param_decls: option flags (defaults to '--password')
    - prompt: bool/str, prompt text (default True)
    - hide_input: bool, hide input (default True)
    - confirmation_prompt: bool, require confirmation (default False)
    """

def version_option(version=None, *param_decls, package_name=None, 
                   prog_name=None, message=None, **kwargs):
    """
    Add a version option.
    
    Parameters:
    - version: str, version string or None to auto-detect
    - param_decls: option flags (defaults to '--version')
    - package_name: str, package name for auto-detection
    - prog_name: str, program name in message
    - message: str, custom version message template
    """

def help_option(*param_decls, **kwargs):
    """
    Add a help option.
    
    Parameters:
    - param_decls: option flags (defaults to '--help')
    - help: str, help text for the option
    """
```

**Usage Examples:**

```python
@click.command()
@click.confirmation_option(prompt='Are you sure you want to delete all files?')
def cleanup():
    """Cleanup command with confirmation."""
    click.echo('Cleaning up...')

@click.command()
@click.password_option()
def secure_operation(password):
    """Command requiring password."""
    click.echo('Performing secure operation...')

@click.command()
@click.version_option(version='1.0.0')
def myapp():
    """Application with version info."""
    click.echo('Running myapp...')
```

### Context Passing Decorators

Pass context information to command callbacks for advanced functionality.

```python { .api }
def pass_context(f):
    """
    Decorator to pass the current context as the first argument.
    
    Returns:
    Decorated function receiving Context as first parameter
    """

def pass_obj(f):
    """
    Decorator to pass the context object (ctx.obj) as the first argument.
    
    Returns:
    Decorated function receiving ctx.obj as first parameter
    """

def make_pass_decorator(object_type, ensure=False):
    """
    Factory to create custom pass decorators.
    
    Parameters:
    - object_type: type to look for in context hierarchy
    - ensure: bool, create object if not found
    
    Returns:
    Decorator function that passes the found object
    """
```

**Usage Examples:**

```python
@click.group()
@click.pass_context
def cli(ctx):
    """Main CLI with context."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = False

@cli.command()
@click.option('--verbose', is_flag=True)
@click.pass_obj
def subcommand(config, verbose):
    """Subcommand accessing shared config."""
    if verbose:
        config['verbose'] = True
    click.echo(f'Verbose mode: {config["verbose"]}')

# Custom pass decorator
pass_config = click.make_pass_decorator(dict)

@cli.command()
@pass_config
def another_command(config):
    """Another command with custom decorator."""
    click.echo(f'Config: {config}')
```