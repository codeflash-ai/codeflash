# Themes and Customization

Theme system providing visual customization including colors, icons, and styling for inquirer prompts. Includes built-in themes and support for custom theme creation from JSON or dictionaries with comprehensive color and symbol customization.

## Capabilities

### Built-in Themes

Four pre-designed themes with different color schemes and visual styles for various use cases and preferences.

```python { .api }
class Default:
    """Default inquirer theme with standard colors and symbols."""
    
    def __init__(self):
        """Initialize default theme with cyan selections and yellow marks."""

class GreenPassion:
    """Green-themed color scheme with nature-inspired styling."""
    
    def __init__(self):
        """Initialize green theme with bold green selections and nature symbols."""

class RedSolace:
    """Red-themed color scheme with warm, energetic styling."""
    
    def __init__(self):
        """Initialize red theme with bright red selections and fire symbols."""

class BlueComposure:
    """Blue-themed color scheme with calm, professional styling."""
    
    def __init__(self):
        """Initialize blue theme with blue selections and geometric symbols."""
```

**Usage Examples:**

```python
import inquirer
from inquirer.themes import Default, GreenPassion, RedSolace, BlueComposure

# Using built-in themes
questions = [
    inquirer.List('color', message="Pick a color", choices=['Red', 'Green', 'Blue'])
]

# Default theme (automatic)
answers = inquirer.prompt(questions)

# Explicit theme selection
answers = inquirer.prompt(questions, theme=GreenPassion())
answers = inquirer.prompt(questions, theme=RedSolace())  
answers = inquirer.prompt(questions, theme=BlueComposure())

# Using themes with shortcuts
name = inquirer.text("Your name?", render=inquirer.render.console.ConsoleRender(theme=GreenPassion()))
```

### Custom Theme Creation from JSON

Load custom themes from JSON configuration, enabling theme sharing and version control.

```python { .api }
def load_theme_from_json(json_theme: str):
    """
    Load a custom theme from JSON string.
    
    Args:
        json_theme: JSON string with theme configuration
        
    Returns:
        Custom theme instance based on Default theme
        
    Raises:
        ThemeError: If theme configuration is invalid
        json.JSONDecodeError: If JSON is malformed
    """
```

**Usage Examples:**

```python
import inquirer
from inquirer.themes import load_theme_from_json

# Define custom theme in JSON
custom_theme_json = '''
{
    "Question": {
        "mark_color": "bright_magenta",
        "brackets_color": "cyan",
        "default_color": "white"
    },
    "List": {
        "selection_color": "bold_yellow_on_black",
        "selection_cursor": "→",
        "unselected_color": "gray"
    },
    "Checkbox": {
        "selection_color": "bold_yellow_on_black", 
        "selection_icon": "►",
        "selected_icon": "✓",
        "unselected_icon": "◯",
        "selected_color": "green",
        "unselected_color": "gray",
        "locked_option_color": "dim_white"
    }
}
'''

# Load and use custom theme
custom_theme = load_theme_from_json(custom_theme_json)

questions = [
    inquirer.Text('name', message="Your name?"),
    inquirer.List('color', message="Favorite color?", choices=['Red', 'Blue', 'Green']),
    inquirer.Checkbox('features', message="Select features", 
                     choices=['Feature A', 'Feature B', 'Feature C'])
]

answers = inquirer.prompt(questions, theme=custom_theme)

# Loading from file
with open('my_theme.json', 'r') as f:
    file_theme = load_theme_from_json(f.read())
```

### Custom Theme Creation from Dictionary

Create custom themes programmatically using dictionary configuration for dynamic theme generation.

```python { .api }
def load_theme_from_dict(dict_theme: dict):
    """
    Load a custom theme from dictionary configuration.
    
    Args:
        dict_theme: Dictionary with theme configuration
        
    Returns:
        Custom theme instance based on Default theme
        
    Raises:
        ThemeError: If theme configuration is invalid
    """
```

**Usage Examples:**

```python
import inquirer
from inquirer.themes import load_theme_from_dict

# Define theme programmatically
corporate_theme = {
    "Question": {
        "mark_color": "blue",
        "brackets_color": "bright_blue",
        "default_color": "cyan"
    },
    "List": {
        "selection_color": "bold_white_on_blue",
        "selection_cursor": "▶",
        "unselected_color": "bright_black"
    },
    "Checkbox": {
        "selection_color": "bold_white_on_blue",
        "selection_icon": "▶",
        "selected_icon": "[✓]",
        "unselected_icon": "[ ]", 
        "selected_color": "bright_blue",
        "unselected_color": "bright_black",
        "locked_option_color": "dim_blue"
    }
}

theme = load_theme_from_dict(corporate_theme)

# Use with prompt
answers = inquirer.prompt(questions, theme=theme)

# Dynamic theme generation
def create_theme_for_user(user_preferences):
    base_color = user_preferences.get('color', 'cyan')
    return load_theme_from_dict({
        "Question": {"mark_color": base_color},
        "List": {"selection_color": f"bold_{base_color}"},
        "Checkbox": {"selected_color": base_color}
    })

user_theme = create_theme_for_user({'color': 'green'})
```

## Theme Configuration Reference

### Question Element Styling

Controls the appearance of question prompts and markers.

```python
"Question": {
    "mark_color": "yellow",        # Color of the [?] marker
    "brackets_color": "normal",    # Color of the brackets around marker
    "default_color": "normal"      # Color of default value display
}
```

### List Selection Styling

Controls the appearance of list-type questions (List and autocomplete).

```python
"List": {
    "selection_color": "cyan",     # Color of selected/highlighted item
    "selection_cursor": ">",       # Symbol for current selection
    "unselected_color": "normal"   # Color of non-selected items
}
```

### Checkbox Styling

Controls the appearance of checkbox questions with multiple selection options.

```python
"Checkbox": {
    "selection_color": "cyan",           # Color of currently highlighted item
    "selection_icon": ">",               # Symbol for current highlight cursor
    "selected_icon": "[X]",              # Symbol for checked items
    "unselected_icon": "[ ]",            # Symbol for unchecked items
    "selected_color": "yellow_bold",     # Color of checked items
    "unselected_color": "normal",        # Color of unchecked items
    "locked_option_color": "gray50"      # Color of locked (unchangeable) items
}
```

### Editor Styling

Controls the appearance of editor question prompts.

```python
"Editor": {
    "opening_prompt_color": "bright_black"  # Color of the "Press <enter> to launch editor" text
}
```

## Available Colors and Styles

Inquirer uses the `blessed` terminal library for colors. Available color names include:

### Basic Colors
- `black`, `red`, `green`, `yellow`, `blue`, `magenta`, `cyan`, `white`

### Bright Colors  
- `bright_black`, `bright_red`, `bright_green`, `bright_yellow`
- `bright_blue`, `bright_magenta`, `bright_cyan`, `bright_white`

### Styles
- `bold`, `dim`, `italic`, `underline`, `blink`, `reverse`
- Combine with colors: `bold_red`, `dim_blue`, `underline_green`

### Background Colors
- Format: `color_on_background`
- Examples: `white_on_blue`, `bold_yellow_on_black`, `red_on_white`

### Extended Colors
- 256-color support: `color1` through `color255`
- Named colors: `gray0` through `gray100`, `darkred`, `lightblue`, etc.

**Color Examples:**

```python
theme_config = {
    "Question": {
        "mark_color": "bold_bright_yellow",
        "brackets_color": "dim_white",
        "default_color": "italic_cyan"
    },
    "List": {
        "selection_color": "bold_white_on_blue",
        "selection_cursor": "❯",
        "unselected_color": "gray70"
    }
}
```

## Advanced Theme Customization

### Theme Inheritance and Extension

```python
from inquirer.themes import Default, load_theme_from_dict

# Start with default theme and modify specific elements
base_theme = Default()

# Extend with custom modifications
custom_modifications = {
    "Question": {
        "mark_color": "bright_magenta"  # Only change the question mark color
    },
    "Checkbox": {
        "selected_icon": "●",           # Use filled circle for selected
        "unselected_icon": "○"          # Use empty circle for unselected
    }
}

extended_theme = load_theme_from_dict(custom_modifications)
```

### Conditional Theme Selection

```python
import os
from inquirer.themes import Default, GreenPassion, RedSolace

def get_theme_for_environment():
    env = os.getenv('APP_ENV', 'development')
    if env == 'production':
        return RedSolace()  # Red for production warnings
    elif env == 'staging':
        return Default()    # Standard for staging
    else:
        return GreenPassion()  # Green for development

questions = [
    inquirer.Confirm('deploy', message="Deploy to {env}?".format(env=os.getenv('APP_ENV')))
]

answers = inquirer.prompt(questions, theme=get_theme_for_environment())
```

### Theme Validation and Error Handling

```python
from inquirer.themes import load_theme_from_dict
from inquirer.errors import ThemeError

def safe_load_theme(theme_config):
    try:
        return load_theme_from_dict(theme_config)
    except ThemeError as e:
        print(f"Theme error: {e}")
        print("Falling back to default theme")
        return Default()

# Potentially invalid theme config
theme_config = {
    "InvalidQuestionType": {  # This will cause ThemeError
        "mark_color": "red"
    }
}

safe_theme = safe_load_theme(theme_config)
```

## Base Theme Class

For advanced customization, you can create themes by extending the base Theme class:

```python { .api }
class Theme:
    """Base theme class defining theme structure."""
    
    def __init__(self):
        """
        Initialize theme with namedtuple definitions for each component.
        
        Creates namedtuple attributes for:
        - Question: mark_color, brackets_color, default_color
        - Editor: opening_prompt_color
        - Checkbox: selection_color, selection_icon, selected_color, unselected_color,
                   selected_icon, unselected_icon, locked_option_color
        - List: selection_color, selection_cursor, unselected_color
        """
        
    @property
    def Question(self):
        """Question styling namedtuple with mark_color, brackets_color, default_color."""
        
    @property  
    def Editor(self):
        """Editor styling namedtuple with opening_prompt_color."""
        
    @property
    def Checkbox(self):
        """Checkbox styling namedtuple with selection_color, selection_icon, selected_color, 
        unselected_color, selected_icon, unselected_icon, locked_option_color."""
        
    @property
    def List(self):
        """List styling namedtuple with selection_color, selection_cursor, unselected_color."""
```

**Custom Theme Class Example:**

```python
from inquirer.themes import Theme
from blessed import Terminal

term = Terminal()

class CustomTheme(Theme):
    def __init__(self):
        super().__init__()
        # Customize question appearance
        self.Question.mark_color = term.bold_magenta
        self.Question.brackets_color = term.bright_blue
        self.Question.default_color = term.cyan
        
        # Customize list appearance
        self.List.selection_color = term.bold_white_on_magenta
        self.List.selection_cursor = "❯"
        self.List.unselected_color = term.bright_black
        
        # Customize checkbox appearance
        self.Checkbox.selection_color = term.bold_white_on_magenta
        self.Checkbox.selected_icon = "✓"
        self.Checkbox.unselected_icon = "◯"
        self.Checkbox.selected_color = term.bold_green
        self.Checkbox.unselected_color = term.bright_black

# Use custom theme
custom_theme = CustomTheme()
answers = inquirer.prompt(questions, theme=custom_theme)
```