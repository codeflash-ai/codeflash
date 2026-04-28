# Click

A comprehensive Python library for creating beautiful command line interfaces with minimal code. Click provides a decorator-based API for building command line tools with automatic help generation, argument validation, and option parsing, supporting arbitrary nesting of commands, lazy loading of subcommands at runtime, and advanced features like shell completion, prompts, progress bars, and styled output.

## Package Information

- **Package Name**: click
- **Language**: Python
- **Installation**: `pip install click`
- **Requires**: Python >=3.10

## Core Imports

```python
import click
```

For individual components:

```python
from click import command, option, argument, group, Context, Command
```

## Basic Usage

```python
import click

@click.command()
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name', help='The person to greet.')
def hello(count, name):
    """Simple program that greets NAME for a total of COUNT times."""
    for _ in range(count):
        click.echo(f'Hello, {name}!')

if __name__ == '__main__':
    hello()
```

## Architecture

Click's design is built around composable command structures:

- **Context**: Holds command state, parameters, and execution environment
- **Command**: Represents individual commands that can be executed
- **Group**: Multi-command containers that can nest other commands and groups
- **Parameter**: Base for Option and Argument classes handling command-line parameters
- **ParamType**: Type system for parameter validation and conversion
- **Decorators**: Function decorators that transform Python functions into Click commands

This architecture enables building complex CLI applications through simple decorators and intuitive patterns, making it ideal for creating development tools, automation scripts, system administration utilities, and any application requiring sophisticated command-line interaction.

## Capabilities

### Command Definition and Decorators

Core decorators for creating commands and command groups, along with parameter decorators for handling command-line arguments and options.

```python { .api }
def command(name=None, cls=None, **attrs): ...
def group(name=None, cls=None, **attrs): ...
def argument(*param_decls, cls=None, **attrs): ...
def option(*param_decls, cls=None, **attrs): ...
```

[Command Definition](./command-definition.md)

### Core Classes and Context Management

The foundational classes that represent commands, groups, parameters, and execution context, providing the structure for command-line applications.

```python { .api }
class Context: ...
class Command: ...
class Group: ...
class Parameter: ...
class Option(Parameter): ...
class Argument(Parameter): ...
```

[Core Classes](./core-classes.md)

### Parameter Types and Validation

Rich type system for command-line parameter validation and conversion, including built-in types for common data formats and custom type creation.

```python { .api }
class ParamType: ...
STRING: StringParamType
INT: IntParamType
FLOAT: FloatParamType
BOOL: BoolParamType
UUID: UUIDParameterType
class Choice(ParamType): ...
class Path(ParamType): ...
class File(ParamType): ...
```

[Parameter Types](./parameter-types.md)

### Terminal UI and User Interaction

Interactive terminal functionality including prompts, confirmations, progress bars, styled output, and text editing capabilities.

```python { .api }
def echo(message=None, file=None, nl=True, err=False, color=None): ...
def prompt(text, default=None, hide_input=False, confirmation_prompt=False, type=None, **kwargs): ...
def confirm(text, default=False, abort=False, **kwargs): ...
def style(text, fg=None, bg=None, bold=None, dim=None, underline=None, **kwargs): ...
def progressbar(iterable=None, length=None, **kwargs): ...
```

[Terminal UI](./terminal-ui.md)

### Exception Handling and Error Management

Comprehensive exception classes for handling various error conditions in command-line applications, with proper error reporting and user-friendly messages.

```python { .api }
class ClickException(Exception): ...
class UsageError(ClickException): ...
class BadParameter(UsageError): ...
class MissingParameter(BadParameter): ...
class NoSuchOption(UsageError): ...
```

[Exception Handling](./exception-handling.md)

### Utilities and Helper Functions

Utility functions for file handling, stream management, text formatting, and other common CLI application needs.

```python { .api }
def get_app_dir(app_name, roaming=True, force_posix=False): ...
def open_file(filename, mode="r", encoding=None, errors="strict", lazy=False, atomic=False): ...
def format_filename(filename, shorten=False): ...
def get_current_context(silent=False): ...
class HelpFormatter: ...
def wrap_text(text, width=78, initial_indent="", subsequent_indent="", preserve_paragraphs=False): ...
```

[Utilities](./utilities.md)

## Types

### Basic Context and Command Types

```python { .api }
class Context:
    def __init__(self, command, parent=None, info_name=None, obj=None, 
                 auto_envvar_prefix=None, default_map=None, terminal_width=None, 
                 max_content_width=None, resilient_parsing=False, 
                 allow_extra_args=None, allow_interspersed_args=None, 
                 ignore_unknown_options=None, help_option_names=None, 
                 token_normalize_func=None, color=None, show_default=None): ...

class Command:
    def __init__(self, name, context_settings=None, callback=None, params=None, 
                 help=None, epilog=None, short_help=None, options_metavar="[OPTIONS]", 
                 add_help_option=True, no_args_is_help=False, hidden=False, 
                 deprecated=False): ...

class Group(Command):
    def __init__(self, name=None, commands=None, invoke_without_command=False, 
                 no_args_is_help=None, subcommand_metavar=None, chain=False, 
                 result_callback=None, **kwargs): ...
```