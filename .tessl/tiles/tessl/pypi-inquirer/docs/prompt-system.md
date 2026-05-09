# Prompt System

Main prompt function and question loading utilities for processing question lists, managing state, and handling user interactions. The prompt system provides the core interaction loop, answer collection, validation handling, and support for loading questions from various data formats.

## Capabilities

### Main Prompt Function

The central prompt function processes a list of questions sequentially, collecting answers and managing the interaction flow with comprehensive error handling.

```python { .api }
def prompt(
    questions: list,
    render=None,
    answers: dict | None = None,
    theme=themes.Default(),
    raise_keyboard_interrupt: bool = False
) -> dict:
    """
    Process a list of questions and collect user answers.
    
    Args:
        questions: List of Question instances to process
        render: Custom render engine (defaults to ConsoleRender)
        answers: Pre-existing answers dict to extend
        theme: Theme for visual styling (defaults to themes.Default())
        raise_keyboard_interrupt: Whether to raise KeyboardInterrupt or handle gracefully
        
    Returns:
        Dictionary mapping question names to user answers
        
    Raises:
        KeyboardInterrupt: If raise_keyboard_interrupt=True and user cancels
    """
```

**Usage Examples:**

```python
import inquirer

# Basic usage
questions = [
    inquirer.Text('name', message="Your name?"),
    inquirer.List('color', message="Favorite color?", choices=['Red', 'Blue', 'Green'])
]
answers = inquirer.prompt(questions)

# With pre-existing answers and custom theme
from inquirer.themes import GreenPassion

initial_answers = {'user_type': 'admin'}
answers = inquirer.prompt(
    questions,
    answers=initial_answers,
    theme=GreenPassion(),
    raise_keyboard_interrupt=True
)

# With validation and dynamic messages
questions = [
    inquirer.Text('first_name', message="First name?"),
    inquirer.Text('last_name', message="Last name?"),
    inquirer.Confirm(
        'confirm_name',
        message="Is your name {first_name} {last_name}?",  # Dynamic message
        default=True
    )
]
```

### Question Loading from Dictionary

Load individual questions from dictionary configurations, enabling dynamic question creation and serialization support.

```python { .api }
def load_from_dict(question_dict: dict) -> Question:
    """
    Load a single question from dictionary configuration.
    
    Args:
        question_dict: Dictionary with question configuration.
                      Must include 'name' and 'kind' keys.
                      Additional keys depend on question type.
                      
    Returns:
        Question instance of the appropriate type
        
    Raises:
        UnknownQuestionTypeError: If 'kind' value is not recognized
        KeyError: If required keys ('name', 'kind') are missing
    """
```

**Usage Example:**

```python
question_config = {
    'kind': 'text',
    'name': 'username', 
    'message': 'Enter username',
    'default': 'admin',
    'validate': lambda _, x: len(x) >= 3
}

question = inquirer.load_from_dict(question_config)
```

### Question Loading from List

Load multiple questions from a list of dictionary configurations, useful for configuration-driven question generation.

```python { .api }
def load_from_list(question_list: list[dict]) -> list[Question]:
    """
    Load multiple questions from list of dictionary configurations.
    
    Args:
        question_list: List of dictionaries, each with question configuration.
                      Each dict must include 'name' and 'kind' keys.
        
    Returns:
        List of Question instances
        
    Raises:
        UnknownQuestionTypeError: If any 'kind' value is not recognized
        KeyError: If any required keys ('name', 'kind') are missing
    """
```

**Usage Example:**

```python
questions_config = [
    {
        'kind': 'text',
        'name': 'project_name',
        'message': 'Project name?'
    },
    {
        'kind': 'list',
        'name': 'project_type',
        'message': 'Project type?',
        'choices': ['web', 'api', 'cli']
    },
    {
        'kind': 'checkbox',
        'name': 'features',
        'message': 'Select features',
        'choices': ['auth', 'db', 'cache']
    }
]

questions = inquirer.load_from_list(questions_config)
answers = inquirer.prompt(questions)
```

### Question Loading from JSON

Load questions from JSON string, supporting both single question and question list formats for configuration file integration.

```python { .api }
def load_from_json(question_json: str) -> list[Question] | Question:
    """
    Load questions from JSON string.
    
    Args:
        question_json: JSON string containing question configuration(s).
                      Must be valid JSON containing dict or list.
        
    Returns:
        - List of Question instances if JSON contains an array
        - Single Question instance if JSON contains an object
        
    Raises:
        json.JSONDecodeError: If JSON is malformed
        TypeError: If JSON contains neither dict nor list
        UnknownQuestionTypeError: If any 'kind' value is not recognized
        KeyError: If any required keys ('name', 'kind') are missing
    """
```

**Usage Examples:**

```python
# Single question from JSON
json_config = '''
{
    "kind": "confirm",
    "name": "proceed",
    "message": "Continue with installation?",
    "default": true
}
'''
question = inquirer.load_from_json(json_config)

# Multiple questions from JSON
json_config = '''
[
    {
        "kind": "text",
        "name": "app_name",
        "message": "Application name?"
    },
    {
        "kind": "list", 
        "name": "environment",
        "message": "Target environment?",
        "choices": ["development", "staging", "production"]
    }
]
'''
questions = inquirer.load_from_json(json_config)
answers = inquirer.prompt(questions)

# Loading from file
with open('questions.json', 'r') as f:
    questions = inquirer.load_from_json(f.read())
```

## Dynamic Question Properties

Questions support dynamic properties that are resolved at runtime based on previous answers, enabling conditional logic and dynamic content.

### Dynamic Messages

Messages can include format strings that reference previous answers:

```python
questions = [
    inquirer.Text('first_name', message="First name?"),
    inquirer.Text('last_name', message="Last name?"),
    inquirer.Text(
        'email',
        message="Email for {first_name} {last_name}?",
        default="{first_name}.{last_name}@company.com"
    )
]
```

### Dynamic Defaults

Default values can be functions that receive the answers dictionary:

```python
def generate_username(answers):
    if 'first_name' in answers and 'last_name' in answers:
        return f"{answers['first_name']}.{answers['last_name']}".lower()
    return "user"

questions = [
    inquirer.Text('first_name', message="First name?"),
    inquirer.Text('last_name', message="Last name?"),
    inquirer.Text(
        'username',
        message="Username?",
        default=generate_username  # Function called with answers dict
    )
]
```

### Dynamic Choices

Choice lists can be functions that generate options based on previous answers:

```python
def get_available_roles(answers):
    if answers.get('user_type') == 'admin':
        return ['super_admin', 'admin', 'user']
    else:
        return ['user', 'guest']

questions = [
    inquirer.List('user_type', message="User type?", choices=['admin', 'regular']),
    inquirer.List(
        'role',
        message="Select role",
        choices=get_available_roles  # Dynamic choices based on user_type
    )
]
```

### Conditional Questions

Questions can be conditionally skipped using the `ignore` parameter:

```python
questions = [
    inquirer.Confirm('has_database', message="Does your app use a database?"),
    inquirer.List(
        'db_type',
        message="Database type?",
        choices=['postgresql', 'mysql', 'sqlite'],
        ignore=lambda answers: not answers.get('has_database', False)  # Skip if no DB
    )
]
```

## Validation System

The prompt system includes comprehensive validation support with custom error messages and validation functions.

### Custom Validation Functions

```python
def validate_email(answers, current):
    import re
    if not re.match(r'^[^@]+@[^@]+\.[^@]+$', current):
        raise inquirer.errors.ValidationError(
            current,
            reason="Please enter a valid email address"
        )
    return True

def validate_port(answers, current):
    try:
        port = int(current)
        if not (1 <= port <= 65535):
            raise inquirer.errors.ValidationError(
                current,
                reason="Port must be between 1 and 65535"
            )
        return True
    except ValueError:
        raise inquirer.errors.ValidationError(
            current,
            reason="Port must be a number"
        )

questions = [
    inquirer.Text('email', message="Email?", validate=validate_email),
    inquirer.Text('port', message="Port?", validate=validate_port, default="8080")
]
```