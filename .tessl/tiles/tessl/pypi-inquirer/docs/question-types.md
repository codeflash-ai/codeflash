# Question Types

Core question classes for different input scenarios. Each question type provides specific validation and interaction patterns optimized for its use case, supporting features like autocompletion, validation, default values, and conditional display.

## Capabilities

### Text Input

Text question for single-line string input with optional autocompletion and validation. Supports dynamic defaults, message formatting, and input validation functions.

```python { .api }
class Text:
    """Text input question with optional autocompletion."""
    
    def __init__(
        self,
        name: str,
        message: str = "",
        default: str | callable | None = None,
        autocomplete: list | None = None,
        validate: callable | bool = True,
        ignore: bool | callable = False,
        show_default: bool = False,
        hints: str | None = None,
        other: bool = False
    ):
        """
        Create a text input question.
        
        Args:
            name: Question identifier for answers dict
            message: Prompt message to display (can include format strings)
            default: Default value (string, callable function, or None)
            autocomplete: List of completion options for tab completion
            validate: Validation function or boolean (True means always valid)
            ignore: Skip question if True or callable returns True
            show_default: Display default value in prompt
            hints: Help text to display
            other: Allow "Other" option for custom input
        """

    kind = "text"
```

**Usage Example:**

```python
import inquirer

question = inquirer.Text(
    'username',
    message="Enter your username",
    default="admin",
    validate=lambda answers, current: len(current) >= 3
)
```

### Password Input

Password question with masked input display. Inherits from Text but hides user input with customizable echo character.

```python { .api }
class Password(Text):
    """Password input with masked display."""
    
    def __init__(
        self,
        name: str,
        echo: str = "*",
        message: str = "",
        default: str | callable | None = None,
        validate: callable | bool = True,
        ignore: bool | callable = False,
        show_default: bool = False,
        hints: str | None = None,
        other: bool = False
    ):
        """
        Create a password input question.
        
        Args:
            name: Question identifier
            echo: Character to display instead of actual input
            message: Prompt message to display
            default: Default value (string, callable function, or None)
            validate: Validation function or boolean
            ignore: Skip question if True or callable returns True
            show_default: Display default value in prompt
            hints: Help text to display
            other: Allow "Other" option for custom input
        """

    kind = "password"
```

**Usage Example:**

```python
password_q = inquirer.Password(
    'password',
    message="Enter your password",
    echo="•"  # Custom echo character
)
```

### List Selection

Single-choice selection from a list of options with keyboard navigation, optional carousel mode, and support for tagged values.

```python { .api }
class List:
    """Single-choice selection from list of options."""
    
    def __init__(
        self,
        name: str,
        message: str = "",
        choices: list | callable | None = None,
        hints: str | None = None,
        default: any = None,
        ignore: bool | callable = False,
        validate: callable | bool = True,
        show_default: bool = False,
        carousel: bool = False,
        other: bool = False,
        autocomplete: list | None = None
    ):
        """
        Create a list selection question.
        
        Args:
            name: Question identifier
            message: Prompt message (can include format strings)
            choices: List of options (strings, (tag, value) tuples, or callable)
            hints: Help text to display
            default: Default selection (can be callable)
            ignore: Skip question if True or callable returns True
            validate: Validation function or boolean
            show_default: Display default value in prompt
            carousel: Enable wraparound navigation at list ends
            other: Allow "Other" option for custom input
            autocomplete: Autocompletion options for filtering
        """

    kind = "list"
```

**Usage Example:**

```python
list_q = inquirer.List(
    'size',
    message="Select size",
    choices=[
        ('Small (S)', 'small'),
        ('Medium (M)', 'medium'), 
        ('Large (L)', 'large')
    ],
    carousel=True,
    default='medium'
)
```

### Multiple Choice Selection

Multiple-choice selection with checkboxes, supporting locked options, carousel navigation, and tagged values.

```python { .api }
class Checkbox:
    """Multiple-choice selection with checkboxes."""
    
    def __init__(
        self,
        name: str,
        message: str = "",
        choices: list | callable | None = None,
        hints: str | None = None,
        locked: list | None = None,
        default: list | callable | None = None,
        ignore: bool | callable = False,
        validate: callable | bool = True,
        show_default: bool = False,
        carousel: bool = False,
        other: bool = False,
        autocomplete: list | None = None
    ):
        """
        Create a checkbox question.
        
        Args:
            name: Question identifier
            message: Prompt message (can include format strings)
            choices: List of options (strings, (tag, value) tuples, or callable)
            hints: Help text to display
            locked: Choices that cannot be deselected (always checked)
            default: Initially selected choices (list or callable)
            ignore: Skip question if True or callable returns True
            validate: Validation function or boolean
            show_default: Display default value in prompt
            carousel: Enable wraparound navigation at list ends
            other: Allow "Other" option for custom input
            autocomplete: Autocompletion options for filtering
        """

    kind = "checkbox"
```

**Usage Example:**

```python
checkbox_q = inquirer.Checkbox(
    'features',
    message="Select features",
    choices=['Feature A', 'Feature B', 'Feature C', 'Feature D'],
    locked=['Feature A'],  # Cannot be deselected
    default=['Feature A', 'Feature B']
)
```

### Yes/No Confirmation

Boolean confirmation question with customizable default value and yes/no response handling.

```python { .api }
class Confirm:
    """Yes/no confirmation question."""
    
    def __init__(
        self,
        name: str,
        message: str = "",
        default: bool | callable = False,
        ignore: bool | callable = False,
        validate: callable | bool = True,
        show_default: bool = False,
        hints: str | None = None,
        other: bool = False
    ):
        """
        Create a confirmation question.
        
        Args:
            name: Question identifier
            message: Prompt message (can include format strings)
            default: Default boolean value (bool or callable)
            ignore: Skip question if True or callable returns True
            validate: Validation function or boolean
            show_default: Display default value in prompt
            hints: Help text to display
            other: Allow "Other" option for custom input
        """

    kind = "confirm"
```

**Usage Example:**

```python
confirm_q = inquirer.Confirm(
    'proceed',
    message="Do you want to continue?",
    default=True
)
```

### External Editor Input

Multi-line text input using external editor (vim, emacs, nano, or $EDITOR/$VISUAL). Inherits from Text with editor-specific behavior.

```python { .api }
class Editor(Text):
    """Multi-line text input using external editor."""
    
    def __init__(
        self,
        name: str,
        message: str = "",
        default: str | callable | None = None,
        validate: callable | bool = True,
        ignore: bool | callable = False,
        show_default: bool = False,
        hints: str | None = None,
        other: bool = False
    ):
        """
        Create an editor input question.
        
        Args:
            name: Question identifier
            message: Prompt message (can include format strings)
            default: Default text content (string, callable, or None)
            validate: Validation function or boolean
            ignore: Skip question if True or callable returns True
            show_default: Display default value in prompt
            hints: Help text to display
            other: Allow "Other" option for custom input
        """

    kind = "editor"
```

**Usage Example:**

```python
editor_q = inquirer.Editor(
    'description',
    message="Enter detailed description"
)
```

### File System Path Input

File or directory path input with validation for path type, existence, and format. Includes built-in path validation and cross-platform support.

```python { .api }
class Path(Text):
    """File/directory path input with validation."""
    
    # Path type constants
    ANY = "any"
    FILE = "file"
    DIRECTORY = "directory"
    
    def __init__(
        self,
        name: str,
        message: str = "",
        default: str | callable | None = None,
        path_type: str = "any",
        exists: bool | None = None,
        validate: callable | bool = True,
        ignore: bool | callable = False,
        show_default: bool = False,
        hints: str | None = None,
        other: bool = False
    ):
        """
        Create a path input question.
        
        Args:
            name: Question identifier
            message: Prompt message (can include format strings)
            default: Default path value (string, callable, or None)
            path_type: Path type ("any", "file", "directory")
            exists: Require path to exist (True), not exist (False), or ignore (None)
            validate: Additional validation function or boolean
            ignore: Skip question if True or callable returns True
            show_default: Display default value in prompt
            hints: Help text to display
            other: Allow "Other" option for custom input
        """

    def validate(self, current: str):
        """
        Validate path according to type and existence requirements.
        
        Args:
            current: Current path input value
            
        Raises:
            ValidationError: If path fails validation criteria
        """

    kind = "path"
```

**Usage Example:**

```python
path_q = inquirer.Path(
    'log_directory',
    message="Select log directory",
    path_type=inquirer.Path.DIRECTORY,
    exists=True
)
```

## Base Question Class

All question types inherit from the base Question class, which provides common functionality for validation, choice management, and dynamic property resolution.

```python { .api }
class Question:
    """Base class for all question types."""
    
    def __init__(
        self,
        name: str,
        message: str = "",
        choices: list | None = None,
        default: any = None,
        ignore: bool | callable = False,
        validate: callable | bool = True,
        show_default: bool = False,
        hints: str | None = None,
        other: bool = False
    ):
        """Base question initialization."""

    def add_choice(self, choice):
        """Add a choice to the question's choice list."""
        
    def validate(self, current):
        """Validate the current answer."""
        
    @property
    def ignore(self) -> bool:
        """Whether this question should be skipped."""
        
    @property 
    def message(self) -> str:
        """Resolved message string."""
        
    @property
    def default(self):
        """Resolved default value."""
        
    @property
    def choices(self) -> list:
        """List of resolved choices."""

    kind = "base question"
```

## Question Factory

```python { .api }
def question_factory(kind: str, *args, **kwargs):
    """
    Create a question instance by type name.
    
    Args:
        kind: Question type ("text", "list", "checkbox", etc.)
        *args, **kwargs: Question constructor arguments
        
    Returns:
        Question instance
        
    Raises:
        UnknownQuestionTypeError: If kind is not recognized
    """
```

## Tagged Values for Complex Choices

```python { .api }
class TaggedValue:
    """Tagged value for complex choice handling with display/value separation."""
    
    def __init__(self, tag: str, value: any):
        """
        Create a tagged value with separate display and return values.
        
        Args:
            tag: Display text shown to user
            value: Actual value returned when selected
        """
        
    def __str__(self) -> str:
        """Return the display tag."""
        
    def __repr__(self) -> str:
        """Return representation of the value."""
        
    def __eq__(self, other) -> bool:
        """Compare with other TaggedValue, tuple, or value."""
        
    def __ne__(self, other) -> bool:
        """Not equal comparison."""
        
    def __hash__(self) -> int:
        """Hash based on (tag, value) tuple."""
        
    @property
    def tag(self) -> str:
        """Display tag shown to user."""
        
    @property
    def value(self) -> any:
        """Actual value returned when selected."""
        
    @property
    def tuple(self) -> tuple:
        """(tag, value) tuple representation."""
```

**Usage Example:**

```python
import inquirer

# Create tagged values for complex choices
choices = [
    inquirer.TaggedValue("Small (1-10 users)", "small"),
    inquirer.TaggedValue("Medium (11-100 users)", "medium"), 
    inquirer.TaggedValue("Large (100+ users)", "large")
]

# Or use tuples (automatically converted to TaggedValue)
choices = [
    ("Small (1-10 users)", "small"),
    ("Medium (11-100 users)", "medium"),
    ("Large (100+ users)", "large")
]

question = inquirer.List(
    'plan',
    message="Select plan size",
    choices=choices
)

# User sees: "Small (1-10 users)" but answer contains: "small"
```