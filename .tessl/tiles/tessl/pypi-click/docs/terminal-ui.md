# Terminal UI and User Interaction

Interactive terminal functionality including prompts, confirmations, progress bars, styled output, and text editing capabilities. Click provides a comprehensive set of tools for creating engaging command-line user interfaces.

## Capabilities

### Output and Echo Functions

Basic output functions with enhanced features over standard print.

```python { .api }
def echo(message=None, file=None, nl=True, err=False, color=None):
    """
    Print message with better support than print().
    
    Parameters:
    - message: Message to print (None prints empty line)
    - file: File object to write to (defaults to stdout)
    - nl: Whether to print trailing newline
    - err: Write to stderr instead of stdout
    - color: Force enable/disable color support
    """

def secho(message=None, file=None, nl=True, err=False, color=None, **styles):
    """
    Styled echo that combines echo() and style().
    
    Parameters:
    - message: Message to print
    - file: File object to write to
    - nl: Whether to print trailing newline
    - err: Write to stderr instead of stdout
    - color: Force enable/disable color support
    - **styles: Style arguments (fg, bg, bold, etc.)
    """
```

**Usage Examples:**

```python
@click.command()
def output_examples():
    """Demonstrate output functions."""
    click.echo('Basic message')
    click.echo('Error message', err=True)
    click.echo('No newline', nl=False)
    click.echo(' - continuation')
    
    # Styled output
    click.secho('Success!', fg='green', bold=True)
    click.secho('Warning', fg='yellow')
    click.secho('Error', fg='red', bold=True)
    click.secho('Info', fg='blue', dim=True)
```

### Text Styling and Colors

Apply ANSI styling to terminal text with comprehensive color and formatting support.

```python { .api }
def style(text, fg=None, bg=None, bold=None, dim=None, underline=None, 
          overline=None, italic=None, blink=None, reverse=None, 
          strikethrough=None, reset=True):
    """
    Apply ANSI styles to text.
    
    Parameters:
    - text: Text to style
    - fg: Foreground color (name, RGB tuple, or hex)
    - bg: Background color (name, RGB tuple, or hex)
    - bold: Bold text
    - dim: Dim/faint text
    - underline: Underlined text
    - overline: Overlined text (limited support)
    - italic: Italic text
    - blink: Blinking text
    - reverse: Reverse foreground/background
    - strikethrough: Strikethrough text
    - reset: Reset all styles at end
    
    Returns:
    Styled text string with ANSI codes
    """

def unstyle(text):
    """
    Remove ANSI styling from text.
    
    Parameters:
    - text: Text with ANSI codes
    
    Returns:
    Plain text with ANSI codes removed
    """
```

**Available Colors:**
- Basic: black, red, green, yellow, blue, magenta, cyan, white
- Bright: bright_black, bright_red, bright_green, bright_yellow, bright_blue, bright_magenta, bright_cyan, bright_white
- RGB tuples: (255, 0, 0) for red
- Hex strings: '#ff0000' for red

**Usage Examples:**

```python
@click.command()
def styling_examples():
    """Demonstrate text styling."""
    # Basic colors
    click.echo(click.style('Red text', fg='red'))
    click.echo(click.style('Green background', bg='green'))
    
    # Text formatting
    click.echo(click.style('Bold text', bold=True))
    click.echo(click.style('Underlined text', underline=True))
    click.echo(click.style('Italic text', italic=True))
    click.echo(click.style('Dim text', dim=True))
    
    # Combined styles
    click.echo(click.style('Bold red on yellow', fg='red', bg='yellow', bold=True))
    
    # RGB and hex colors
    click.echo(click.style('RGB color', fg=(255, 165, 0)))  # Orange
    click.echo(click.style('Hex color', fg='#ff69b4'))  # Hot pink
    
    # Remove styling
    styled_text = click.style('Styled text', fg='blue', bold=True)
    plain_text = click.unstyle(styled_text)
    click.echo(f'Original: {styled_text}')
    click.echo(f'Unstyled: {plain_text}')
```

### Prompting and Input

Interactive input functions for gathering user data with validation and confirmation.

```python { .api }
def prompt(text, default=None, hide_input=False, confirmation_prompt=False, 
           type=None, value_proc=None, prompt_suffix=": ", show_default=True, 
           err=False, show_choices=True):
    """
    Prompt user for input with validation.
    
    Parameters:
    - text: Prompt text to display
    - default: Default value if user provides no input
    - hide_input: Hide input (for passwords)
    - confirmation_prompt: Prompt twice for confirmation
    - type: ParamType for validation and conversion
    - value_proc: Function to process the value
    - prompt_suffix: Text to append to prompt
    - show_default: Show default value in prompt
    - err: Print prompt to stderr
    - show_choices: Show choices for Choice types
    
    Returns:
    User input after validation and conversion
    """

def confirm(text, default=False, abort=False, prompt_suffix=": ", 
            show_default=True, err=False):
    """
    Prompt for yes/no confirmation.
    
    Parameters:
    - text: Confirmation prompt text
    - default: Default response (True/False)
    - abort: Abort if user chooses no
    - prompt_suffix: Text to append to prompt
    - show_default: Show default in prompt
    - err: Print prompt to stderr
    
    Returns:
    Boolean response
    """
```

**Usage Examples:**

```python
@click.command()
def prompt_examples():
    """Demonstrate prompting functions."""
    # Basic prompts
    name = click.prompt('Your name')
    age = click.prompt('Your age', type=int)
    
    # Default values
    city = click.prompt('Your city', default='Unknown')
    
    # Hidden input
    password = click.prompt('Password', hide_input=True)
    
    # Confirmation prompt
    new_password = click.prompt('New password', hide_input=True, 
                               confirmation_prompt=True)
    
    # Type validation
    email = click.prompt('Email', type=click.STRING)
    port = click.prompt('Port', type=click.IntRange(1, 65535), default=8080)
    
    # Choice prompts
    level = click.prompt('Log level', 
                        type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']))
    
    # Confirmations
    if click.confirm('Do you want to continue?'):
        click.echo('Continuing...')
    
    # Abort on no
    click.confirm('Are you sure?', abort=True)
    click.echo('User confirmed')
```

### Progress Indicators

Visual progress tracking for long-running operations.

```python { .api }
def progressbar(iterable=None, length=None, label=None, hidden=False, 
                show_eta=True, show_percent=None, show_pos=False, 
                item_show_func=None, fill_char="#", empty_char="-", 
                bar_template="%(label)s  [%(bar)s]  %(info)s", 
                info_sep="  ", width=36, file=None, color=None, 
                update_min_steps=1):
    """
    Create a progress bar context manager.
    
    Parameters:
    - iterable: Iterable to track progress for
    - length: Length of operation (if iterable not provided)
    - label: Label to show before progress bar
    - hidden: Hide progress bar completely
    - show_eta: Show estimated time remaining
    - show_percent: Show percentage complete
    - show_pos: Show current position
    - item_show_func: Function to display current item
    - fill_char: Character for filled portion
    - empty_char: Character for empty portion
    - bar_template: Template string for bar format
    - info_sep: Separator for info sections
    - width: Width of progress bar
    - file: File to write progress to
    - color: Force enable/disable color
    - update_min_steps: Minimum steps between updates
    
    Returns:
    ProgressBar context manager
    """
```

**Usage Examples:**

```python
import time

@click.command()
def progress_examples():
    """Demonstrate progress bars."""
    # Basic progress with iterable
    items = range(100)
    with click.progressbar(items, label='Processing items') as bar:
        for item in bar:
            time.sleep(0.01)  # Simulate work
    
    # Manual progress updates
    with click.progressbar(length=50, label='Manual progress') as bar:
        for i in range(50):
            # Do work
            time.sleep(0.02)
            bar.update(1)
    
    # Custom styling
    with click.progressbar(range(30), 
                          label='Custom bar',
                          fill_char='█',
                          empty_char='░',
                          show_percent=True,
                          show_pos=True) as bar:
        for item in bar:
            time.sleep(0.05)
    
    # With item display
    def show_item(item):
        return f'Item {item}'
    
    with click.progressbar(range(20),
                          label='Items',
                          item_show_func=show_item) as bar:
        for item in bar:
            time.sleep(0.1)
```

### Interactive Features

Advanced interactive terminal features for enhanced user experience.

```python { .api }
def getchar(echo=False):
    """
    Get a single character from terminal.
    
    Parameters:
    - echo: Whether to echo the character
    
    Returns:
    Single character string
    """

def pause(info=None, err=False):
    """
    Pause execution until user presses a key.
    
    Parameters:
    - info: Custom message to display (default: "Press any key to continue...")
    - err: Print message to stderr
    """

def edit(text=None, editor=None, env=None, require_save=True, 
         extension=".txt", filename=None):
    """
    Open text in external editor.
    
    Parameters:
    - text: Initial text content
    - editor: Editor command (defaults to $EDITOR environment variable)
    - env: Environment variables for editor
    - require_save: Require file to be saved
    - extension: File extension for temporary file
    - filename: Specific filename to use
    
    Returns:
    Edited text content
    """

def launch(url, wait=False, locate=False):
    """
    Launch URL or file in default application.
    
    Parameters:
    - url: URL or file path to launch
    - wait: Wait for application to close
    - locate: Show file in file manager instead of opening
    
    Returns:
    Exit code of launched application
    """
```

**Usage Examples:**

```python
@click.command()
def interactive_examples():
    """Demonstrate interactive features."""
    # Single character input
    click.echo('Press any key...')
    char = click.getchar()
    click.echo(f'You pressed: {char}')
    
    # Pause execution
    click.pause('Press any key to continue with the demo...')
    
    # Text editing
    initial_text = "# Configuration\n\nEdit this text:\n"
    edited_text = click.edit(initial_text, extension='.md')
    if edited_text:
        click.echo('You entered:')
        click.echo(edited_text)
    
    # Launch applications
    if click.confirm('Open documentation in browser?'):
        click.launch('https://click.palletsprojects.com/')
    
    # Show file in file manager
    if click.confirm('Show current directory?'):
        click.launch('.', locate=True)

@click.command()
def editor_config():
    """Edit configuration file."""
    config_text = """
# Application Configuration
debug = false
port = 8080
host = localhost
"""
    
    click.echo('Opening configuration editor...')
    result = click.edit(config_text, extension='.conf')
    
    if result:
        click.echo('Configuration updated:')
        click.echo(result)
    else:
        click.echo('Configuration unchanged')
```

### Display and Paging

Functions for displaying large amounts of text with user-friendly paging.

```python { .api }
def echo_via_pager(text_or_generator, color=None):
    """
    Display text through a pager (like 'less').
    
    Parameters:
    - text_or_generator: Text string or generator yielding lines
    - color: Force enable/disable color in pager
    """

def clear():
    """Clear the terminal screen."""
```

**Usage Examples:**

```python
@click.command()
def display_examples():
    """Demonstrate display functions."""
    # Clear screen
    click.clear()
    
    # Generate long text
    long_text = '\n'.join([f'Line {i}: This is a long document with many lines.' 
                          for i in range(100)])
    
    # Display via pager
    click.echo('Displaying long text via pager...')
    click.echo_via_pager(long_text)
    
    # Generator example
    def generate_lines():
        for i in range(50):
            yield f'Generated line {i}: Some content here'
            if i % 10 == 0:
                yield click.style(f'--- Section {i//10} ---', bold=True)
    
    click.echo('Displaying generated content...')
    click.echo_via_pager(generate_lines())
```