# Shortcuts

Simplified interface for single questions without creating question objects. These functions provide immediate input collection for quick interactions and scripting scenarios, returning the answer directly rather than requiring a prompt() call.

## Capabilities

### Text Input Shortcut

Quick text input function that immediately prompts for input and returns the result.

```python { .api }
def text(
    message: str,
    autocomplete: list | None = None,
    render: ConsoleRender | None = None,
    default: str | callable | None = None,
    validate: callable | bool = True,
    ignore: bool | callable = False,
    show_default: bool = False,
    hints: str | None = None,
    other: bool = False
) -> str:
    """
    Quick text input prompt.
    
    Args:
        message: Prompt message to display
        autocomplete: List of completion options for tab completion
        render: Custom render engine (defaults to ConsoleRender())
        default: Default value (string, callable, or None)
        validate: Validation function or boolean
        ignore: Skip question if True or callable returns True
        show_default: Display default value in prompt
        hints: Help text to display
        other: Allow "Other" option for custom input
        
    Returns:
        User's text input as string
    """
```

**Usage Examples:**

```python
import inquirer

# Simple text input
name = inquirer.text("What's your name?")
print(f"Hello, {name}!")

# With validation
def validate_length(answers, current):
    if len(current) < 3:
        raise inquirer.errors.ValidationError(current, reason="Minimum 3 characters")
    return True

username = inquirer.text(
    "Enter username (min 3 chars)",
    default="user",
    validate=validate_length
)

# With autocompletion
city = inquirer.text(
    "Enter city",
    autocomplete=['New York', 'Los Angeles', 'Chicago', 'Houston']
)
```

### Password Input Shortcut

Quick password input with masked display.

```python { .api }
def password(
    message: str,
    render: ConsoleRender | None = None,
    echo: str = "*",
    default: str | callable | None = None,
    validate: callable | bool = True,
    ignore: bool | callable = False,
    show_default: bool = False,
    hints: str | None = None,
    other: bool = False
) -> str:
    """
    Quick password input prompt with masked display.
    
    Args:
        message: Prompt message to display
        render: Custom render engine (defaults to ConsoleRender())
        echo: Character to display instead of actual input
        default: Default value (string, callable, or None)
        validate: Validation function or boolean
        ignore: Skip question if True or callable returns True
        show_default: Display default value in prompt
        hints: Help text to display
        other: Allow "Other" option for custom input
        
    Returns:
        User's password input as string
    """
```

**Usage Example:**

```python
# Basic password input
pwd = inquirer.password("Enter your password:")

# Custom echo character
secret = inquirer.password("Enter secret key:", echo="•")
```

### List Selection Shortcut

Quick single-choice selection from a list of options.

```python { .api }
def list_input(
    message: str,
    render: ConsoleRender | None = None,
    choices: list | callable | None = None,
    default: any = None,
    validate: callable | bool = True,
    ignore: bool | callable = False,
    show_default: bool = False,
    hints: str | None = None,
    carousel: bool = False,
    other: bool = False,
    autocomplete: list | None = None
):
    """
    Quick list selection prompt.
    
    Args:
        message: Prompt message to display
        render: Custom render engine (defaults to ConsoleRender())
        choices: List of options (strings, (tag, value) tuples, or callable)
        default: Default selection (can be callable)
        validate: Validation function or boolean
        ignore: Skip question if True or callable returns True
        show_default: Display default value in prompt
        hints: Help text to display
        carousel: Enable wraparound navigation at list ends
        other: Allow "Other" option for custom input
        autocomplete: Autocompletion options for filtering
        
    Returns:
        Selected choice value
    """
```

**Usage Examples:**

```python
# Simple list selection
color = inquirer.list_input(
    "Pick a color",
    choices=['Red', 'Green', 'Blue']
)

# With tagged values
size = inquirer.list_input(
    "Select size",
    choices=[
        ('Small (up to 10 items)', 'small'),
        ('Medium (up to 100 items)', 'medium'),
        ('Large (unlimited)', 'large')
    ],
    default='medium'
)

# With carousel navigation
option = inquirer.list_input(
    "Navigate options",
    choices=['Option 1', 'Option 2', 'Option 3', 'Option 4'],
    carousel=True  # Wrap around at ends
)
```

### Checkbox Selection Shortcut

Quick multiple-choice selection with checkboxes.

```python { .api }
def checkbox(
    message: str,
    render: ConsoleRender | None = None,
    choices: list | callable | None = None,
    locked: list | None = None,
    default: list | callable | None = None,
    validate: callable | bool = True,
    ignore: bool | callable = False,
    show_default: bool = False,
    hints: str | None = None,
    carousel: bool = False,
    other: bool = False,
    autocomplete: list | None = None
) -> list:
    """
    Quick checkbox selection prompt.
    
    Args:
        message: Prompt message to display
        render: Custom render engine (defaults to ConsoleRender())
        choices: List of options (strings, (tag, value) tuples, or callable)
        locked: Choices that cannot be deselected (always checked)
        default: Initially selected choices (list or callable)
        validate: Validation function or boolean
        ignore: Skip question if True or callable returns True
        show_default: Display default value in prompt
        hints: Help text to display
        carousel: Enable wraparound navigation at list ends
        other: Allow "Other" option for custom input
        autocomplete: Autocompletion options for filtering
        
    Returns:
        List of selected choice values
    """
```

**Usage Examples:**

```python
# Basic multiple selection
features = inquirer.checkbox(
    "Select features to enable",
    choices=['Authentication', 'Database', 'Caching', 'Logging']
)

# With locked options and defaults
components = inquirer.checkbox(
    "Select components",
    choices=['Core', 'UI', 'API', 'Tests'],
    locked=['Core'],  # Cannot be deselected
    default=['Core', 'UI']
)

# With validation
def validate_at_least_one(answers, current):
    if not current:
        raise inquirer.errors.ValidationError(current, reason="Select at least one option")
    return True

selected = inquirer.checkbox(
    "Choose services (minimum 1)",
    choices=['Web Server', 'Database', 'Cache', 'Queue'],
    validate=validate_at_least_one
)
```

### Confirmation Shortcut

Quick yes/no confirmation prompt.

```python { .api }
def confirm(
    message: str,
    render: ConsoleRender | None = None,
    default: bool | callable = False,
    validate: callable | bool = True,
    ignore: bool | callable = False,
    show_default: bool = False,
    hints: str | None = None,
    other: bool = False
) -> bool:
    """
    Quick confirmation prompt.
    
    Args:
        message: Prompt message to display
        render: Custom render engine (defaults to ConsoleRender())
        default: Default boolean value (bool or callable)
        validate: Validation function or boolean
        ignore: Skip question if True or callable returns True
        show_default: Display default value in prompt
        hints: Help text to display
        other: Allow "Other" option for custom input
        
    Returns:
        Boolean confirmation result
    """
```

**Usage Examples:**

```python
# Simple confirmation
proceed = inquirer.confirm("Do you want to continue?")
if proceed:
    print("Continuing...")

# With default value
delete_files = inquirer.confirm(
    "Delete all files? This cannot be undone!",
    default=False  # Default to "No" for destructive actions
)

# With validation
def confirm_understanding(answers, current):
    if not current:
        raise inquirer.errors.ValidationError(
            current, 
            reason="You must confirm to proceed"
        )
    return True

understood = inquirer.confirm(
    "I understand the risks and want to proceed",
    validate=confirm_understanding
)
```

### Editor Input Shortcut

Quick multi-line text input using external editor.

```python { .api }
def editor(
    message: str,
    render: ConsoleRender | None = None,
    default: str | callable | None = None,
    validate: callable | bool = True,
    ignore: bool | callable = False,
    show_default: bool = False,
    hints: str | None = None,
    other: bool = False
) -> str:
    """
    Quick editor input prompt using external editor.
    
    Args:
        message: Prompt message to display
        render: Custom render engine (defaults to ConsoleRender())
        default: Default text content (string, callable, or None)
        validate: Validation function or boolean
        ignore: Skip question if True or callable returns True
        show_default: Display default value in prompt
        hints: Help text to display
        other: Allow "Other" option for custom input
        
    Returns:
        Multi-line text content from editor
    """
```

**Usage Examples:**

```python
# Basic editor input
description = inquirer.editor("Enter project description")
print("Description:", description)

# With default content
config = inquirer.editor(
    "Edit configuration",
    default="# Default configuration\nkey=value\n"
)

# The editor uses $VISUAL or $EDITOR environment variables,
# falling back to vim -> emacs -> nano based on availability
```

### Path Input Shortcut

Quick file or directory path input with validation.

```python { .api }
def path(
    message: str,
    render: ConsoleRender | None = None,
    default: str | callable | None = None,
    path_type: str = "any",
    exists: bool | None = None,
    validate: callable | bool = True,
    ignore: bool | callable = False,
    show_default: bool = False,
    hints: str | None = None,
    other: bool = False
) -> str:
    """
    Quick path input prompt with validation.
    
    Args:
        message: Prompt message to display
        render: Custom render engine (defaults to ConsoleRender())
        default: Default path value (string, callable, or None)
        path_type: Path type ("any", "file", "directory")
        exists: Require path to exist (True), not exist (False), or ignore (None)
        validate: Additional validation function or boolean
        ignore: Skip question if True or callable returns True
        show_default: Display default value in prompt
        hints: Help text to display
        other: Allow "Other" option for custom input
        
    Returns:
        Validated file or directory path
    """
```

**Usage Examples:**

```python
# Basic path input
file_path = inquirer.path("Enter file path")

# Directory only
log_dir = inquirer.path(
    "Select log directory",
    path_type=inquirer.Path.DIRECTORY,
    exists=True  # Must exist
)

# File that may not exist yet
output_file = inquirer.path(
    "Output file path",
    path_type=inquirer.Path.FILE,
    exists=False,  # Can be new file
    default="./output.txt"
)

# Existing file requirement
config_file = inquirer.path(
    "Select configuration file",
    path_type=inquirer.Path.FILE,
    exists=True  # Must exist
)
```

## Shortcut vs Question Class Usage

**Use shortcuts when:**
- Asking single questions
- Quick scripting or interactive sessions
- Simple validation requirements
- No need for complex question chaining

**Use question classes with prompt() when:**
- Multiple related questions
- Complex validation dependencies between questions
- Dynamic question generation
- Need to maintain question state
- Advanced features like conditional questions

**Combined usage example:**

```python
import inquirer

# Use shortcuts for standalone questions
project_name = inquirer.text("Project name?")
use_database = inquirer.confirm("Include database support?")

# Use question classes for related questions
questions = []
if use_database:
    questions.extend([
        inquirer.List('db_type', message="Database type?", 
                     choices=['postgresql', 'mysql', 'sqlite']),
        inquirer.Text('db_name', message="Database name?", 
                     default=f"{project_name}_db"),
        inquirer.Text('db_user', message="Database user?", default="admin")
    ])

if questions:
    db_config = inquirer.prompt(questions)
    print("Database configuration:", db_config)
```

## Custom Render Engines

All shortcut functions accept a custom render parameter for specialized rendering:

```python
from inquirer.render.console import ConsoleRender
from inquirer.themes import GreenPassion

# Custom render with theme
custom_render = ConsoleRender(theme=GreenPassion())

name = inquirer.text(
    "Your name?",
    render=custom_render
)
```