# Inquirer

A comprehensive Python library for creating interactive command line user interfaces, based on Inquirer.js. Inquirer enables developers to create sophisticated CLI prompts including text input, password fields, single and multiple choice lists, checkboxes, file path selectors, and external editor integration with built-in validation, custom themes, and cross-platform compatibility.

## Package Information

- **Package Name**: inquirer
- **Language**: Python
- **Installation**: `pip install inquirer`
- **Requires**: Python >= 3.9.2

## Core Imports

```python
import inquirer
```

For direct access to question types:

```python
from inquirer import Text, Password, List, Checkbox, Confirm, Editor, Path
```

For shortcut functions:

```python
from inquirer import text, password, list_input, checkbox, confirm, editor, path
```

For themes:

```python
from inquirer.themes import Default, GreenPassion, RedSolace, BlueComposure
```

## Basic Usage

```python
import inquirer

# Define questions
questions = [
    inquirer.Text('name', message="What's your name?"),
    inquirer.List('size',
                  message="What size do you need?",
                  choices=['Large', 'Medium', 'Small']),
    inquirer.Checkbox('features',
                      message="What features do you want?",
                      choices=['Feature A', 'Feature B', 'Feature C']),
    inquirer.Confirm('proceed', message="Do you want to proceed?", default=True)
]

# Get answers
answers = inquirer.prompt(questions)
print(answers)  # {'name': 'John', 'size': 'Medium', 'features': ['Feature A'], 'proceed': True}

# Using shortcuts for single questions
name = inquirer.text(message="Enter your name")
confirmed = inquirer.confirm("Are you sure?")
```

## Architecture

Inquirer follows a modular architecture:

- **Question Classes**: Seven question types (Text, Password, Editor, Confirm, List, Checkbox, Path) each handling specific input scenarios
- **Prompt System**: Central `prompt()` function processes question lists and manages interaction flow
- **Render Engine**: Console-based UI rendering with terminal control and event handling
- **Theme System**: Customizable color schemes and visual styling
- **Validation Framework**: Built-in and custom validation support with error handling
- **Shortcut Interface**: Simplified single-question functions for quick interactions

## Capabilities

### Question Types

Core question classes for different input scenarios including text, passwords, selections, confirmations, file paths, and multi-line editing. Each question type provides specific validation and interaction patterns optimized for its use case.

```python { .api }
class Text(name, message="", default=None, autocomplete=None, **kwargs): ...
class Password(name, echo="*", **kwargs): ...
class List(name, message="", choices=None, default=None, carousel=False, **kwargs): ...
class Checkbox(name, message="", choices=None, locked=None, carousel=False, **kwargs): ...
class Confirm(name, default=False, **kwargs): ...
class Editor(name, **kwargs): ...
class Path(name, default=None, path_type="any", exists=None, **kwargs): ...
```

[Question Types](./question-types.md)

### Prompt System

Main prompt function and question loading utilities for processing question lists, managing state, and handling user interactions with comprehensive error handling and validation.

```python { .api }
def prompt(questions, render=None, answers=None, theme=themes.Default(), raise_keyboard_interrupt=False): ...
def load_from_dict(question_dict): ...
def load_from_list(question_list): ...
def load_from_json(question_json): ...
```

[Prompt System](./prompt-system.md)

### Shortcut Functions

Simplified interface for single questions without creating question objects. These functions provide immediate input collection for quick interactions and scripting scenarios.

```python { .api }
def text(message, autocomplete=None, **kwargs): ...
def password(message, **kwargs): ...
def list_input(message, **kwargs): ...
def checkbox(message, **kwargs): ...
def confirm(message, **kwargs): ...
def editor(message, **kwargs): ...
def path(message, **kwargs): ...
```

[Shortcuts](./shortcuts.md)

### Themes and Customization

Theme system providing visual customization including colors, icons, and styling. Includes built-in themes and support for custom theme creation from JSON or dictionaries.

```python { .api }
class Default(): ...
class GreenPassion(): ...
class RedSolace(): ...
class BlueComposure(): ...
def load_theme_from_json(json_theme): ...
def load_theme_from_dict(dict_theme): ...
```

[Themes](./themes.md)

### Render System

Console-based rendering engine providing terminal UI control, event handling, and visual presentation for interactive prompts with customizable themes and cross-platform terminal compatibility.

```python { .api }
class ConsoleRender:
    def __init__(self, theme=None): ...
    def render(self, question, answers=None): ...

class Render:
    def __init__(self, impl=ConsoleRender): ...
    def render(self, question, answers): ...
```

[Render System](./render-system.md)

## Types and Constants

```python { .api }
class TaggedValue:
    """Tagged value for complex choice handling with display/value separation."""
    def __init__(self, tag: str, value: any): ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __hash__(self) -> int: ...
    
    @property
    def tag(self) -> str: ...
    
    @property
    def value(self) -> any: ...
    
    @property
    def tuple(self) -> tuple: ...

# Path type constants
Path.ANY = "any"
Path.FILE = "file" 
Path.DIRECTORY = "directory"

# Exception classes
class ValidationError(Exception):
    """Raised when input validation fails."""
    def __init__(self, value, reason: str | None = None): ...
    
    @property
    def value(self): ...
    
    @property  
    def reason(self) -> str | None: ...

class UnknownQuestionTypeError(Exception):
    """Raised when question factory receives unknown question type."""

class ThemeError(AttributeError):
    """Raised when theme configuration is invalid."""

class EndOfInput(Exception):
    """Raised when input stream ends unexpectedly."""
    def __init__(self, selection, *args): ...
    
    @property
    def selection(self): ...
```