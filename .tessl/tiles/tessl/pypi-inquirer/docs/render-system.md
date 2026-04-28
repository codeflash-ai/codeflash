# Render System

Console-based rendering engine providing terminal UI control, event handling, and visual presentation for interactive prompts. The render system handles terminal compatibility, visual styling, and user interaction processing with support for customizable themes.

## Capabilities

### Console Renderer

Primary rendering implementation for terminal-based interactive prompts with theme support and cross-platform terminal compatibility.

```python { .api }
class ConsoleRender:
    """Console-based renderer for terminal UI with theme support."""
    
    def __init__(self, theme=None):
        """
        Initialize console renderer.
        
        Args:
            theme: Theme instance for visual styling (defaults to Default theme)
        """
        
    def render(self, question, answers: dict | None = None):
        """
        Render a question and collect user input.
        
        Args:
            question: Question instance to render and process
            answers: Previous answers dictionary for dynamic content
            
        Returns:
            User's input/selection for the question
            
        Raises:
            KeyboardInterrupt: If user cancels with Ctrl+C
            ValidationError: If input fails validation
        """
```

**Usage Examples:**

```python
import inquirer
from inquirer.render.console import ConsoleRender
from inquirer.themes import GreenPassion

# Custom renderer with theme
render = ConsoleRender(theme=GreenPassion())

# Use with individual questions
question = inquirer.Text('name', message="Your name?")
answer = render.render(question)

# Use with prompt system
questions = [
    inquirer.Text('name', message="Your name?"),
    inquirer.List('color', message="Favorite color?", choices=['Red', 'Blue', 'Green'])
]

# Each question uses the custom renderer
answers = inquirer.prompt(questions, render=render)
```

### Generic Render Interface

Abstract rendering interface allowing for different UI implementations and render engine swapping.

```python { .api }
class Render:
    """Generic render interface with pluggable implementations."""
    
    def __init__(self, impl=ConsoleRender):
        """
        Initialize render interface.
        
        Args:
            impl: Render implementation class (defaults to ConsoleRender)
        """
        
    def render(self, question, answers: dict):
        """
        Render a question using the configured implementation.
        
        Args:
            question: Question instance to render
            answers: Answers dictionary for context
            
        Returns:
            User input collected by the implementation
        """
```

**Usage Example:**

```python
from inquirer.render import Render, ConsoleRender

# Default console implementation
render = Render()

# Explicit console implementation
render = Render(impl=ConsoleRender)

# Use with questions
question = inquirer.Confirm('proceed', message="Continue?")
result = render.render(question, {})
```

## Event System

Low-level event handling for keyboard input and terminal control.

### Event Classes

```python { .api }
class Event:
    """Base event class."""

class KeyPressed(Event):
    """Keyboard input event."""
    
    def __init__(self, value: str):
        """
        Create key press event.
        
        Args:
            value: Key character or escape sequence
        """
        
    @property
    def value(self) -> str:
        """Key value that was pressed."""

class Repaint(Event):
    """Screen repaint event for UI updates."""

class KeyEventGenerator:
    """Generator for keyboard input events."""
    
    def __init__(self, key_generator=None):
        """
        Initialize key event generator.
        
        Args:
            key_generator: Optional custom key input function
        """
        
    def next(self) -> KeyPressed:
        """
        Get next keyboard input event.
        
        Returns:
            KeyPressed event with input value
        """
```

**Usage Example:**

```python
from inquirer.events import KeyEventGenerator, KeyPressed, Repaint

# Create event generator
generator = KeyEventGenerator()

# Process keyboard events
while True:
    event = generator.next()
    if isinstance(event, KeyPressed):
        if event.value == '\r':  # Enter key
            break
        print(f"Key pressed: {event.value}")
```

## Terminal Integration

The render system integrates with terminal capabilities through the `blessed` library, providing:

- **Cross-platform support**: Windows, macOS, Linux terminal compatibility
- **Color and styling**: Full color palette with style combinations
- **Cursor control**: Positioning and visibility management  
- **Screen management**: Clear screen, line manipulation, scrolling
- **Input handling**: Raw keyboard input with special key detection

### Theme Integration

Renderers automatically apply theme styling to visual elements:

```python
import inquirer
from inquirer.render.console import ConsoleRender
from inquirer.themes import RedSolace

# Create themed renderer
themed_render = ConsoleRender(theme=RedSolace())

# Questions rendered with red theme styling
questions = [
    inquirer.List('action', message="Select action", 
                 choices=['Create', 'Update', 'Delete']),
    inquirer.Confirm('confirm', message="Are you sure?")
]

answers = inquirer.prompt(questions, render=themed_render)
```

## Advanced Rendering

### Custom Render Implementation

For specialized use cases, you can create custom render implementations:

```python
from inquirer.render.console import ConsoleRender

class CustomRender(ConsoleRender):
    def __init__(self, theme=None, prefix=">>> "):
        super().__init__(theme)
        self.prefix = prefix
        
    def render(self, question, answers=None):
        # Add custom prefix to all prompts
        original_message = question.message
        question._message = f"{self.prefix}{original_message}"
        
        try:
            return super().render(question, answers)
        finally:
            # Restore original message
            question._message = original_message

# Use custom renderer
custom_render = CustomRender(prefix="[CUSTOM] ")
answer = custom_render.render(
    inquirer.Text('name', message="Enter name"), 
    {}
)
```

### Error Handling in Rendering

The render system handles various error conditions:

```python
import inquirer
from inquirer.errors import ValidationError

try:
    questions = [
        inquirer.Text('email', 
                     message="Email address", 
                     validate=lambda _, x: '@' in x or ValidationError(x, "Invalid email"))
    ]
    answers = inquirer.prompt(questions)
except KeyboardInterrupt:
    print("\\nUser cancelled input")
except ValidationError as e:
    print(f"Validation failed: {e.reason}")
```