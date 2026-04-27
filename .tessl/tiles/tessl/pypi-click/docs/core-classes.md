# Core Classes and Context Management

The foundational classes that represent commands, groups, parameters, and execution context, providing the structure for command-line applications. These classes form the backbone of Click's architecture and enable sophisticated CLI functionality.

## Capabilities

### Context Management

The Context class holds command state, parameters, and execution environment, providing access to shared data and command hierarchy.

```python { .api }
class Context:
    def __init__(self, command, parent=None, info_name=None, obj=None, 
                 auto_envvar_prefix=None, default_map=None, terminal_width=None, 
                 max_content_width=None, resilient_parsing=False, 
                 allow_extra_args=None, allow_interspersed_args=None, 
                 ignore_unknown_options=None, help_option_names=None, 
                 token_normalize_func=None, color=None, show_default=None):
        """
        Create a new context.
        
        Parameters:
        - command: Command instance this context belongs to
        - parent: Parent context (for nested commands)
        - info_name: Descriptive name for documentation
        - obj: User object for sharing data between commands
        - auto_envvar_prefix: Prefix for automatic environment variables
        - default_map: Dict of parameter defaults
        - terminal_width: Width of terminal for formatting
        - max_content_width: Maximum width for help content
        - resilient_parsing: Continue parsing on errors
        - allow_extra_args: Allow extra arguments
        - allow_interspersed_args: Allow interspersed arguments and options
        - ignore_unknown_options: Ignore unknown options
        - help_option_names: Names for help options
        - token_normalize_func: Function to normalize option names
        - color: Enable/disable color output
        - show_default: Show default values in help
        """
    
    def find_root(self):
        """Find the outermost context in the hierarchy."""
    
    def find_object(self, object_type):
        """Find the closest object of given type in context hierarchy."""
    
    def ensure_object(self, object_type):
        """Find object or create if missing."""
    
    def lookup_default(self, name, call=True):
        """Get default value from default_map."""
    
    def fail(self, message):
        """Abort execution with error message."""
    
    def abort(self):
        """Abort the script execution."""
    
    def exit(self, code=0):
        """Exit with specified code."""
    
    def get_usage(self):
        """Get formatted usage string."""
    
    def get_help(self):
        """Get formatted help page."""
    
    def invoke(self, callback, *args, **kwargs):
        """Invoke a callback or command."""
    
    def forward(self, cmd, *args, **kwargs):
        """Forward execution to another command."""
```

**Usage Examples:**

```python
@click.group()
@click.pass_context
def cli(ctx):
    """Main CLI group."""
    ctx.ensure_object(dict)
    ctx.obj['database_url'] = 'sqlite:///app.db'

@cli.command()
@click.pass_context
def status(ctx):
    """Show application status."""
    db_url = ctx.obj['database_url']
    click.echo(f'Database: {db_url}')
    
    # Access parent context
    if ctx.parent:
        click.echo(f'Parent command: {ctx.parent.info_name}')
```

### Command Classes

Basic command structures for creating executable CLI commands.

```python { .api }
class Command:
    def __init__(self, name, context_settings=None, callback=None, params=None, 
                 help=None, epilog=None, short_help=None, options_metavar="[OPTIONS]", 
                 add_help_option=True, no_args_is_help=False, hidden=False, 
                 deprecated=False):
        """
        Create a command.
        
        Parameters:
        - name: Command name
        - context_settings: Dict with default context settings
        - callback: Function to execute when command is invoked
        - params: List of parameters (options/arguments)
        - help: Help text for the command
        - epilog: Text shown after help
        - short_help: Short help for command listings
        - options_metavar: How to show options in usage
        - add_help_option: Whether to add --help option
        - no_args_is_help: Show help when no arguments given
        - hidden: Hide from help listings
        - deprecated: Mark as deprecated
        """
    
    def main(self, args=None, prog_name=None, complete_var=None, 
             standalone_mode=True, **extra):
        """
        Main entry point for the command.
        
        Parameters:
        - args: Arguments to parse (defaults to sys.argv)
        - prog_name: Program name
        - complete_var: Environment variable for shell completion
        - standalone_mode: Handle exceptions and exit
        """
    
    def make_context(self, info_name, args, parent=None, **extra):
        """Create a context for this command."""
    
    def invoke(self, ctx):
        """Invoke the command callback."""
    
    def get_help(self, ctx):
        """Get formatted help text."""
    
    def get_usage(self, ctx):
        """Get formatted usage line."""
    
    def get_params(self, ctx):
        """Get all parameters including help option."""

class Group(Command):
    def __init__(self, name=None, commands=None, invoke_without_command=False, 
                 no_args_is_help=None, subcommand_metavar=None, chain=False, 
                 result_callback=None, **kwargs):
        """
        Create a command group.
        
        Parameters:
        - name: Group name
        - commands: Dict of initial subcommands
        - invoke_without_command: Invoke group callback without subcommand
        - no_args_is_help: Show help when no arguments provided
        - subcommand_metavar: How to represent subcommands in help
        - chain: Allow chaining multiple subcommands
        - result_callback: Callback for processing command results
        """
    
    def add_command(self, cmd, name=None):
        """Register a command with this group."""
    
    def command(self, *args, **kwargs):
        """Decorator to create and register a command."""
    
    def group(self, *args, **kwargs):
        """Decorator to create and register a subgroup."""
    
    def get_command(self, ctx, cmd_name):
        """Get a command by name."""
    
    def list_commands(self, ctx):
        """List all available command names."""
    
    def result_callback(self, replace=False):
        """Decorator for result processing callback."""

class CommandCollection(Group):
    def __init__(self, name=None, sources=None, **kwargs):
        """
        Create a command collection from multiple groups.
        
        Parameters:
        - name: Collection name
        - sources: List of Groups to collect commands from
        """
    
    def add_source(self, group):
        """Add a group as a source of commands."""
```

**Usage Examples:**

```python
# Basic command
@click.command()
@click.option('--verbose', is_flag=True)
def simple_command(verbose):
    """A simple command example."""
    if verbose:
        click.echo('Verbose mode enabled')

# Command group
@click.group()
def database():
    """Database management commands."""
    pass

@database.command()
def init():
    """Initialize database."""
    click.echo('Database initialized')

@database.command()
def migrate():
    """Run database migrations."""
    click.echo('Running migrations...')

# Command collection
users_cli = click.Group(name='users')
admin_cli = click.Group(name='admin')

@users_cli.command()
def list_users():
    """List all users."""
    click.echo('Listing users...')

@admin_cli.command()
def backup():
    """Create backup."""
    click.echo('Creating backup...')

# Combine groups
main_cli = click.CommandCollection(sources=[users_cli, admin_cli])
```

### Parameter Classes

Base classes for handling command-line parameters with validation and type conversion.

```python { .api }
class Parameter:
    def __init__(self, param_decls=None, type=None, required=False, default=None, 
                 callback=None, nargs=None, multiple=False, metavar=None, 
                 expose_value=True, is_eager=False, envvar=None, shell_complete=None, 
                 deprecated=False):
        """
        Base parameter class.
        
        Parameters:
        - param_decls: Parameter declarations (names/flags)
        - type: ParamType for validation and conversion
        - required: Whether parameter is required
        - default: Default value
        - callback: Function to process the value
        - nargs: Number of arguments to consume
        - multiple: Accept multiple values
        - metavar: How to display in help
        - expose_value: Store value in context.params
        - is_eager: Process before non-eager parameters
        - envvar: Environment variable name(s)
        - shell_complete: Shell completion function
        - deprecated: Mark as deprecated
        """
    
    def get_default(self, ctx, call=True):
        """Get the default value for this parameter."""
    
    def type_cast_value(self, ctx, value):
        """Convert and validate the value."""
    
    def process_value(self, ctx, value):
        """Process value with type conversion and callback."""
    
    def resolve_envvar_value(self, ctx):
        """Find value from environment variables."""

class Option(Parameter):
    def __init__(self, param_decls=None, show_default=None, prompt=False, 
                 confirmation_prompt=False, prompt_required=True, hide_input=False, 
                 is_flag=None, flag_value=None, multiple=False, count=False, 
                 allow_from_autoenv=True, type=None, help=None, hidden=False, 
                 show_choices=True, show_envvar=False, deprecated=False, **attrs):
        """
        Command-line option parameter.
        
        Parameters:
        - param_decls: Option flags (e.g., '--verbose', '-v')
        - show_default: Show default value in help
        - prompt: Prompt for value if not provided
        - confirmation_prompt: Prompt for confirmation
        - prompt_required: Whether prompt is required
        - hide_input: Hide input when prompting
        - is_flag: Treat as boolean flag
        - flag_value: Value when flag is present
        - multiple: Allow multiple values
        - count: Count number of times option is used
        - allow_from_autoenv: Allow automatic environment variables
        - help: Help text for the option
        - hidden: Hide from help output
        - show_choices: Show choices in help
        - show_envvar: Show environment variable in help
        """
    
    def prompt_for_value(self, ctx):
        """Prompt user for the value."""

class Argument(Parameter):
    def __init__(self, param_decls, required=None, **attrs):
        """
        Positional command-line argument.
        
        Parameters:
        - param_decls: Argument name
        - required: Whether argument is required (default True)
        """
```

**Usage Examples:**

```python
# Custom parameter with callback
def validate_port(ctx, param, value):
    """Validate port number."""
    if value < 1 or value > 65535:
        raise click.BadParameter('Port must be between 1 and 65535')
    return value

@click.command()
@click.option('--port', type=int, default=8080, callback=validate_port)
def server(port):
    """Start server on specified port."""
    click.echo(f'Starting server on port {port}')

# Environment variable support
@click.command()
@click.option('--debug', is_flag=True, envvar='DEBUG')
@click.argument('config_file', envvar='CONFIG_FILE')
def run(debug, config_file):
    """Run application with config."""
    click.echo(f'Debug: {debug}, Config: {config_file}')
```