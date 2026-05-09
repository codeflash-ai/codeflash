# Project Configuration

Project-level configuration and management including sys.path customization, extension loading, and multi-file analysis scope. Enables project-aware analysis and search with proper dependency resolution and workspace understanding.

## Capabilities

### Project Creation and Configuration

Create and configure projects for comprehensive code analysis with custom settings.

```python { .api }
class Project:
    def __init__(self, path, *, environment_path=None, load_unsafe_extensions=False, 
                 sys_path=None, added_sys_path=(), smart_sys_path=True):
        """
        Create a project configuration.
        
        Parameters:
        - path (str or Path): Base project path.
        - environment_path (str, optional): Python environment path.
        - load_unsafe_extensions (bool): Load unsafe extensions. Default False.
        - sys_path (list, optional): Custom sys.path override.
        - added_sys_path (tuple): Additional sys.path entries. Default ().
        - smart_sys_path (bool): Calculate intelligent sys.path. Default True.
        """
```

**Usage Example:**
```python
import jedi
from jedi import Project
from pathlib import Path

# Basic project setup
project_path = Path("/path/to/my/project")
project = Project(project_path)

# Project with custom sys.path
custom_paths = ["/path/to/dependencies", "/opt/custom/libs"]
project = Project(
    project_path,
    added_sys_path=tuple(custom_paths),
    smart_sys_path=True
)

# Project with specific environment
project = Project(
    project_path,
    environment_path="/path/to/venv/bin/python"
)

# Use project with Script
code = '''
import my_local_module
from utils import helper_function
'''

script = jedi.Script(code=code, path=project_path / "main.py", project=project)
completions = script.complete(line=2, column=20)
```

### Project Properties and Settings

Access and configure project properties for analysis behavior.

```python { .api }
class Project:
    path: Path  # Base project path
    sys_path: list  # Custom sys.path
    smart_sys_path: bool  # Whether to calculate smart sys.path
    load_unsafe_extensions: bool  # Load unsafe extensions
    added_sys_path: tuple  # Additional sys.path entries
    
    def get_environment(self):
        """Get project environment."""
    
    def save(self):
        """Save project configuration to .jedi/project.json"""
```

**Usage Example:**
```python
import jedi
from jedi import Project

project = Project("/path/to/project")

# Check project settings
print(f"Project path: {project.path}")
print(f"Custom sys.path: {project.sys_path}")
print(f"Added paths: {project.added_sys_path}")
print(f"Smart sys.path: {project.smart_sys_path}")
print(f"Load unsafe extensions: {project.load_unsafe_extensions}")

# Get project environment
env = project.get_environment()
print(f"Project Python: {env.executable}")
print(f"Project Python version: {env.version_info}")

# Save configuration
project.save()  # Creates .jedi/project.json
```

### Project Loading and Persistence

Load and save project configurations for consistent analysis settings.

```python { .api }
@classmethod
def load(cls, path):
    """
    Load project configuration from path.
    
    Parameters:
    - path (str or Path): Project path containing .jedi/project.json.
    
    Returns:
    Project object with loaded configuration.
    """
```

**Usage Example:**
```python
import jedi
from jedi import Project
from pathlib import Path

# Create and configure project
project_path = Path("/path/to/project")
project = Project(
    project_path,
    added_sys_path=("/opt/custom/libs",),
    load_unsafe_extensions=True
)

# Save configuration
project.save()

# Later, load the same configuration
loaded_project = Project.load(project_path)
print(f"Loaded project: {loaded_project.path}")
print(f"Loaded settings: {loaded_project.added_sys_path}")

# Use loaded project
script = jedi.Script(
    path=project_path / "main.py",
    project=loaded_project
)
```

### Default Project Management

Get default project configurations based on file locations and context.

```python { .api }
def get_default_project(path=None):
    """
    Get default project configuration for a path.
    
    Parameters:
    - path (str or Path, optional): File or directory path.
    
    Returns:
    Project object with intelligent defaults.
    """
```

**Usage Example:**
```python
import jedi
from jedi import get_default_project

# Get default project for current directory
default_project = get_default_project()
print(f"Default project path: {default_project.path}")

# Get project for specific file
file_path = "/path/to/project/src/main.py"
file_project = get_default_project(file_path)
print(f"File project path: {file_project.path}")

# Use with Script
script = jedi.Script(
    code="import local_module",
    path=file_path,
    project=file_project
)

# The project helps resolve local imports
completions = script.complete(line=1, column=18)
```

### Project-Wide Search

Search for symbols across the entire project with scope control.

```python { .api }
def search(self, string, *, all_scopes=False):
    """
    Search for names across the entire project.
    
    Parameters:
    - string (str): Search string pattern.
    - all_scopes (bool): Search in all scopes. Default False.
    
    Returns:
    Generator of Name objects from across the project.
    """

def complete_search(self, string, **kwargs):
    """
    Search with completion-style matching across project.
    
    Parameters:
    - string (str): Search string pattern.
    - **kwargs: Additional search options.
    
    Returns:
    Generator of Completion objects from across the project.
    """
```

**Usage Example:**
```python
import jedi
from jedi import Project

# Set up project
project = Project("/path/to/large/project")

# Search for class definitions across the project
class_results = list(project.search("class:MyClass"))
for result in class_results:
    print(f"Found class: {result.name}")
    print(f"  File: {result.module_path}")
    print(f"  Line: {result.line}")

# Search for function definitions
function_results = list(project.search("function:process_data"))
for result in function_results:
    print(f"Found function: {result.name}")
    print(f"  Module: {result.module_name}")
    print(f"  Full name: {result.full_name}")

# Fuzzy search across project
fuzzy_results = list(project.complete_search("proc", fuzzy=True))
for result in fuzzy_results:
    print(f"Fuzzy match: {result.name} in {result.module_name}")

# Search in all scopes (including local variables)
all_scope_results = list(project.search("temp_var", all_scopes=True))
```

### Multi-File Analysis Context

Enable cross-file analysis and import resolution within project scope.

**Usage Example:**
```python
import jedi
from jedi import Project

# Project structure:
# /project/
#   ├── main.py
#   ├── utils/
#   │   ├── __init__.py
#   │   └── helpers.py
#   └── models/
#       ├── __init__.py
#       └── user.py

project = Project("/project")

# File: models/user.py
user_model_code = '''
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
    
    def get_display_name(self):
        return f"{self.name} <{self.email}>"
'''

# File: utils/helpers.py
helpers_code = '''
from models.user import User

def create_admin_user():
    return User("Admin", "admin@example.com")

def format_user_list(users):
    return [user.get_display_name() for user in users]
'''

# File: main.py
main_code = '''
from utils.helpers import create_admin_user, format_user_list
from models.user import User

# Create users
admin = create_admin_user()
regular_user = User("John", "john@example.com")

# Format for display
users = [admin, regular_user]
formatted = format_user_list(users)
'''

# Analyze main.py with project context
script = jedi.Script(
    code=main_code,
    path="/project/main.py",
    project=project
)

# Get completions for imported functions
completions = script.complete(line=5, column=15)  # At 'create_admin_user'
print("Available completions:")
for comp in completions:
    print(f"  {comp.name}: {comp.type}")

# Navigate to definition across files
definitions = script.goto(line=5, column=8, follow_imports=True)  # At 'create_admin_user'
for defn in definitions:
    print(f"Definition: {defn.name}")
    print(f"  File: {defn.module_path}")
    print(f"  Line: {defn.line}")

# Find references across the project
references = script.get_references(line=6, column=15)  # At 'User'
print("References across project:")
for ref in references:
    print(f"  {ref.module_path}:{ref.line} - {ref.get_line_code().strip()}")
```

### Sys.path Customization

Customize Python module search paths for dependency resolution.

**Usage Example:**
```python
import jedi
from jedi import Project

# Project with custom dependencies
project = Project(
    "/path/to/project",
    added_sys_path=(
        "/opt/custom/libraries",
        "/home/user/dev/shared-utils",
        "/path/to/vendor/packages"
    ),
    smart_sys_path=True  # Also include intelligent paths
)

# Code that imports from custom paths
code = '''
import custom_library
from shared_utils import common_functions
from vendor_package import specialized_tool

custom_library.'''

script = jedi.Script(code=code, project=project)

# Get completions from custom library
completions = script.complete(line=4, column=15)
print("Custom library completions:")
for comp in completions:
    print(f"  {comp.name}: {comp.type}")

# Override sys.path completely
project_override = Project(
    "/path/to/project",
    sys_path=[
        "/custom/path/1",
        "/custom/path/2",
        "/usr/lib/python3.9/site-packages"  # Keep standard library
    ],
    smart_sys_path=False  # Don't add intelligent paths
)
```

### Extension and Plugin Support

Configure loading of Jedi extensions and plugins for enhanced functionality.

**Usage Example:**
```python
import jedi
from jedi import Project

# Project with unsafe extensions enabled
project = Project(
    "/path/to/project",
    load_unsafe_extensions=True
)

# This enables additional analysis capabilities that might be unsafe
# in some environments but provide enhanced completions
script = jedi.Script(
    code="import numpy as np; np.array([1,2,3]).",
    project=project
)

# Extensions might provide better NumPy completions
completions = script.complete()
numpy_methods = [c for c in completions if 'array' in c.name.lower()]
print("Enhanced NumPy completions:")
for method in numpy_methods:
    print(f"  {method.name}: {method.description}")
```

### Project Configuration File

Manage project settings through configuration files.

**Configuration File Format (`.jedi/project.json`):**
```json
{
    "added_sys_path": [
        "/opt/custom/libs",
        "/path/to/dependencies"
    ],
    "load_unsafe_extensions": false,
    "smart_sys_path": true,
    "environment_path": "/path/to/venv/bin/python"
}
```

**Usage Example:**
```python
import jedi
from jedi import Project
import json

# Manually create configuration
config = {
    "added_sys_path": ["/opt/custom/libs"],
    "load_unsafe_extensions": False,
    "smart_sys_path": True
}

# Save configuration manually
project_path = "/path/to/project"
config_dir = f"{project_path}/.jedi"
import os
os.makedirs(config_dir, exist_ok=True)

with open(f"{config_dir}/project.json", "w") as f:
    json.dump(config, f, indent=2)

# Load project with configuration
project = Project.load(project_path)
print(f"Loaded config: {project.added_sys_path}")
```