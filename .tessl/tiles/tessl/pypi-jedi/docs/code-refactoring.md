# Code Refactoring

Code refactoring operations including rename, extract variable, extract function, and inline operations. Provides safe refactoring with proper scope analysis, conflict detection, and cross-file changes.

## Capabilities

### Variable Renaming

Rename variables, functions, classes, and other symbols across their scope with conflict detection.

```python { .api }
def rename(self, line=None, column=None, *, new_name):
    """
    Rename the symbol at cursor position.
    
    Parameters:
    - line (int, optional): Line number (1-based).
    - column (int, optional): Column number (0-based).
    - new_name (str): New name for the symbol.
    
    Returns:
    Refactoring object with rename changes.
    
    Raises:
    RefactoringError: If rename is not possible or conflicts exist.
    """
```

**Usage Example:**
```python
import jedi
from jedi.api.exceptions import RefactoringError

code = '''
class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, value):
        self.result += value
        return self.result
    
    def get_result(self):
        return self.result

calc = Calculator()
calc.add(5)
print(calc.get_result())
'''

script = jedi.Script(code=code, path='calculator.py')

try:
    # Rename 'result' attribute to 'total'
    refactoring = script.rename(line=3, column=13, new_name='total')  # At 'result'
    
    # Get the changes
    changes = refactoring.get_changed_files()
    for file_path, changed_file in changes.items():
        print(f"Changes in {file_path}:")
        print(changed_file.get_diff())
    
    # Apply the changes
    refactoring.apply()
    print("Rename applied successfully")
    
except RefactoringError as e:
    print(f"Rename failed: {e}")
```

### Extract Variable

Extract expressions into new variables to improve code readability and reduce duplication.

```python { .api }
def extract_variable(self, line, column, *, new_name, until_line=None, until_column=None):
    """
    Extract expression to a new variable.
    
    Parameters:
    - line (int): Start line number (1-based).
    - column (int): Start column number (0-based).
    - new_name (str): Name for the extracted variable.
    - until_line (int, optional): End line number.
    - until_column (int, optional): End column number.
    
    Returns:
    Refactoring object with extraction changes.
    
    Raises:
    RefactoringError: If extraction is not possible.
    """
```

**Usage Example:**
```python
import jedi
from jedi.api.exceptions import RefactoringError

code = '''
def calculate_area(radius):
    return 3.14159 * radius * radius

def calculate_circumference(radius):
    return 2 * 3.14159 * radius

def calculate_volume(radius, height):
    base_area = 3.14159 * radius * radius
    return base_area * height
'''

script = jedi.Script(code=code, path='geometry.py')

try:
    # Extract pi constant from first function
    refactoring = script.extract_variable(
        line=2,
        column=11,  # Start of '3.14159'
        until_line=2,
        until_column=19,  # End of '3.14159'
        new_name='PI'
    )
    
    print("Extraction preview:")
    changes = refactoring.get_changed_files()
    for file_path, changed_file in changes.items():
        print(changed_file.get_new_code())
    
    # Apply the extraction
    refactoring.apply()
    
except RefactoringError as e:
    print(f"Extraction failed: {e}")

# Extract complex expression
complex_code = '''
def process_data(items):
    filtered_items = [item for item in items if len(item.strip()) > 0 and item.strip().startswith('data_')]
    return [item.strip().upper() for item in filtered_items]
'''

script = jedi.Script(code=complex_code, path='processor.py')

# Extract the filtering condition
refactoring = script.extract_variable(
    line=2,
    column=47,  # Start of condition
    until_line=2,
    until_column=95,  # End of condition
    new_name='is_valid_data_item'
)
```

### Extract Function

Extract code blocks into new functions to improve modularity and reusability.

```python { .api }
def extract_function(self, line, column, *, new_name, until_line=None, until_column=None):
    """
    Extract code block to a new function.
    
    Parameters:
    - line (int): Start line number (1-based).
    - column (int): Start column number (0-based).
    - new_name (str): Name for the extracted function.
    - until_line (int, optional): End line number.
    - until_column (int, optional): End column number.
    
    Returns:
    Refactoring object with extraction changes.
    
    Raises:
    RefactoringError: If extraction is not possible.
    """
```

**Usage Example:**
```python
import jedi
from jedi.api.exceptions import RefactoringError

code = '''
def process_user_data(users):
    results = []
    for user in users:
        # Complex validation logic
        if user.get('email') and '@' in user['email']:
            if user.get('age') and user['age'] >= 18:
                if user.get('name') and len(user['name'].strip()) > 0:
                    normalized_user = {
                        'name': user['name'].strip().title(),
                        'email': user['email'].lower(),
                        'age': user['age']
                    }
                    results.append(normalized_user)
    return results
'''

script = jedi.Script(code=code, path='user_processor.py')

try:
    # Extract validation and normalization logic
    refactoring = script.extract_function(
        line=4,
        column=8,   # Start of validation logic
        until_line=11,
        column=42,  # End of normalization
        new_name='validate_and_normalize_user'
    )
    
    print("Function extraction preview:")
    changes = refactoring.get_changed_files()
    for file_path, changed_file in changes.items():
        print(changed_file.get_new_code())
    
    # Apply the extraction
    refactoring.apply()
    
except RefactoringError as e:
    print(f"Function extraction failed: {e}")

# Extract with automatic parameter detection
calculation_code = '''
def calculate_compound_interest(principal, rate, time, frequency):
    annual_rate = rate / 100
    compound_frequency = frequency
    amount = principal * (1 + annual_rate / compound_frequency) ** (compound_frequency * time)
    interest = amount - principal
    return interest
'''

script = jedi.Script(code=calculation_code, path='finance.py')

# Extract compound calculation
refactoring = script.extract_function(
    line=3,
    column=4,   # Start of calculation
    until_line=4,
    column=30,  # End of calculation
    new_name='calculate_compound_amount'
)
```

### Inline Variable

Inline variables by replacing their usage with their values, removing unnecessary indirection.

```python { .api }
def inline(self, line=None, column=None):
    """
    Inline the variable at cursor position.
    
    Parameters:
    - line (int, optional): Line number (1-based).
    - column (int, optional): Column number (0-based).
    
    Returns:
    Refactoring object with inline changes.
    
    Raises:
    RefactoringError: If inlining is not possible.
    """
```

**Usage Example:**
```python
import jedi
from jedi.api.exceptions import RefactoringError

code = '''
def calculate_discount(price, discount_rate):
    discount_decimal = discount_rate / 100
    discount_amount = price * discount_decimal
    final_price = price - discount_amount
    return final_price
'''

script = jedi.Script(code=code, path='pricing.py')

try:
    # Inline 'discount_decimal' variable
    refactoring = script.inline(line=2, column=4)  # At 'discount_decimal'
    
    print("Inline preview:")
    changes = refactoring.get_changed_files()
    for file_path, changed_file in changes.items():
        print("Before:")
        print(code)
        print("\nAfter:")
        print(changed_file.get_new_code())
    
    # Apply the inline
    refactoring.apply()
    
except RefactoringError as e:
    print(f"Inline failed: {e}")

# Inline with multiple usages
multi_usage_code = '''
def format_message(user_name, message_type, content):
    prefix = f"[{message_type}]"
    timestamp = "2024-01-01 12:00:00"
    formatted_message = f"{timestamp} {prefix} {user_name}: {content}"
    log_entry = f"LOG: {formatted_message}"
    return log_entry
'''

script = jedi.Script(code=multi_usage_code, path='messaging.py')

# Inline 'prefix' variable (used once)
refactoring = script.inline(line=2, column=4)  # At 'prefix'
```

### Cross-File Refactoring

Perform refactoring operations across multiple files in a project.

**Usage Example:**
```python
import jedi
from jedi import Project

# Project with multiple files
project = Project("/path/to/project")

# File 1: models.py
models_code = '''
class UserModel:
    def __init__(self, name, email):
        self.user_name = name
        self.user_email = email
    
    def get_display_name(self):
        return f"{self.user_name} <{self.user_email}>"
'''

# File 2: services.py  
services_code = '''
from models import UserModel

class UserService:
    def create_user(self, name, email):
        user = UserModel(name, email)
        print(f"Created user: {user.get_display_name()}")
        return user
    
    def update_user_name(self, user, new_name):
        user.user_name = new_name
        return user
'''

# Rename 'user_name' across all files
script = jedi.Script(
    code=models_code,
    path="/path/to/project/models.py",
    project=project
)

try:
    # Rename user_name to username across project
    refactoring = script.rename(line=3, column=13, new_name='username')
    
    # Check all affected files
    changes = refactoring.get_changed_files()
    print("Files to be changed:")
    for file_path, changed_file in changes.items():
        print(f"\n{file_path}:")
        print(changed_file.get_diff())
    
    # Get file renames if any
    renames = refactoring.get_renames()
    if renames:
        print("File renames:")
        for old_path, new_path in renames.items():
            print(f"  {old_path} -> {new_path}")
    
    # Apply changes to all files
    refactoring.apply()
    
except RefactoringError as e:
    print(f"Cross-file refactoring failed: {e}")
```

### Refactoring Validation and Preview

Validate refactoring operations and preview changes before applying them.

**Usage Example:**
```python
import jedi

code = '''
def calculate_tax(income, tax_rate):
    base_tax = income * 0.1
    additional_tax = (income - 50000) * (tax_rate - 0.1) if income > 50000 else 0
    total_tax = base_tax + additional_tax
    return total_tax
'''

script = jedi.Script(code=code, path='tax_calculator.py')

# Preview variable extraction
try:
    refactoring = script.extract_variable(
        line=3,
        column=21,  # Start of '(income - 50000)'
        until_line=3,
        until_column=38,  # End of '(income - 50000)'
        new_name='excess_income'
    )
    
    # Preview changes without applying
    print("Refactoring preview:")
    changes = refactoring.get_changed_files()
    for file_path, changed_file in changes.items():
        print(f"\nFile: {file_path}")
        print("Diff:")
        print(changed_file.get_diff())
        
        print("\nNew content preview:")
        print(changed_file.get_new_code())
    
    # Get unified diff for the entire refactoring
    unified_diff = refactoring.get_diff()
    print("\nUnified diff:")
    print(unified_diff)
    
    # Only apply if preview looks good
    user_input = input("Apply refactoring? (y/n): ")
    if user_input.lower() == 'y':
        refactoring.apply()
        print("Refactoring applied successfully")
    
except RefactoringError as e:
    print(f"Refactoring validation failed: {e}")
```

## Refactoring Result Types

### Refactoring

```python { .api }
class Refactoring:
    def get_changed_files(self):
        """Get dictionary of Path -> ChangedFile for all changes."""
    
    def get_renames(self):
        """Get dictionary of Path -> Path for file renames."""
    
    def get_diff(self):
        """Get unified diff string for all changes."""
    
    def apply(self):
        """Apply all refactoring changes to files."""
```

### ChangedFile

```python { .api }
class ChangedFile:
    def get_diff(self):
        """Get diff string for this file."""
    
    def get_new_code(self):
        """Get new file content after changes."""
    
    def apply(self):
        """Apply changes to this specific file."""
```

## Error Handling

### RefactoringError

```python { .api }
class RefactoringError(Exception):
    """Raised when refactoring operations cannot be completed safely."""
```

**Common refactoring error scenarios:**
- Name conflicts (new name already exists in scope)
- Invalid syntax after refactoring
- Cross-file dependencies that would break
- Attempting to refactor built-in or imported symbols
- Extracting code with complex control flow dependencies