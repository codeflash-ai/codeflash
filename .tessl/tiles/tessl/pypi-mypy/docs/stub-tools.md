# Stub Tools

Tools for generating and validating Python stub files (.pyi) for type checking without runtime dependencies. These tools enable type checking of untyped libraries and creating type-only interfaces.

## Capabilities

### Stub Generation (stubgen)

Automatically generate Python stub files from source code or runtime inspection.

```python { .api }
def main(args: list[str] | None = None) -> None:
    """
    Main entry point for stub generation.
    
    Parameters:
    - args: Command line arguments (uses sys.argv if None)
    
    Usage:
    stubgen.main(['mymodule'])  # Generate stubs for mymodule
    """
```

#### Generation Modes

**Source-based Generation**: Analyze Python source code to create stubs
```bash
stubgen mymodule.py           # Generate from source file
stubgen -p mypackage          # Generate from package source
```

**Runtime-based Generation**: Inspect imported modules at runtime
```bash
stubgen -m requests           # Generate from installed module
stubgen -m numpy pandas       # Generate multiple modules
```

#### Key Features

- **Preserve public API**: Only exports public names and interfaces
- **Type annotation extraction**: Extracts existing type hints from source
- **Docstring handling**: Can include or exclude docstrings
- **Private member control**: Options for including private definitions
- **Cross-references**: Maintains import relationships between modules

### Stub Testing (stubtest)

Validate stub files against runtime modules to ensure accuracy and completeness.

```python { .api }
def main(args: list[str] | None = None) -> None:
    """
    Main entry point for stub validation.
    
    Parameters:
    - args: Command line arguments (uses sys.argv if None)
    
    Usage:
    stubtest.main(['mymodule'])  # Test stubs for mymodule
    """
```

#### Validation Types

**Signature Validation**: Check function and method signatures match
```bash
stubtest mymodule             # Basic signature checking
stubtest --check-typeddict mymodule  # Include TypedDict validation
```

**Completeness Checking**: Ensure all public APIs are stubbed
```bash
stubtest --ignore-missing-stub mymodule  # Allow missing stubs
```

**Runtime Compatibility**: Verify stubs work with actual runtime behavior
```bash
stubtest --allowlist allowlist.txt mymodule  # Use allowlist for known issues
```

## Advanced Usage

### Programmatic Stub Generation

```python
import subprocess
import tempfile
import os
from pathlib import Path

class StubGenerator:
    """Programmatic interface for stub generation."""
    
    def __init__(self, output_dir: str = "stubs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_from_source(self, source_paths: list[str]) -> bool:
        """Generate stubs from source files."""
        cmd = [
            'stubgen',
            '-o', str(self.output_dir),
            '--include-private',
            '--no-import'
        ] + source_paths
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    
    def generate_from_modules(self, module_names: list[str]) -> bool:
        """Generate stubs from installed modules."""
        cmd = [
            'stubgen',
            '-o', str(self.output_dir),
            '-m'
        ] + module_names
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    
    def generate_package_stubs(self, package_name: str) -> bool:
        """Generate stubs for entire package."""
        cmd = [
            'stubgen',
            '-o', str(self.output_dir),
            '-p', package_name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

# Usage
generator = StubGenerator("my_stubs")
generator.generate_from_modules(['requests', 'numpy'])
generator.generate_package_stubs('mypackage')
```

### Programmatic Stub Testing

```python
import subprocess
import json
from pathlib import Path

class StubTester:
    """Programmatic interface for stub testing."""
    
    def __init__(self, allowlist_file: str | None = None):
        self.allowlist_file = allowlist_file
    
    def test_stubs(self, module_names: list[str]) -> dict:
        """Test stubs and return results."""
        cmd = ['stubtest'] + module_names
        
        if self.allowlist_file:
            cmd.extend(['--allowlist', self.allowlist_file])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'errors': self.parse_errors(result.stderr)
        }
    
    def parse_errors(self, stderr: str) -> list[dict]:
        """Parse stubtest error output."""
        errors = []
        for line in stderr.strip().split('\n'):
            if 'error:' in line:
                # Parse error format: module.function: error: message
                parts = line.split(':', 2)  
                if len(parts) >= 3:
                    errors.append({
                        'location': parts[0].strip(),
                        'message': parts[2].strip()
                    })
        return errors
    
    def generate_allowlist(self, module_names: list[str]) -> str:
        """Generate allowlist for current differences."""
        cmd = ['stubtest', '--generate-allowlist'] + module_names
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout

# Usage  
tester = StubTester('allowlist.txt')
results = tester.test_stubs(['mymodule'])

if not results['success']:
    print(f"Found {len(results['errors'])} stub errors")
    for error in results['errors']:
        print(f"  {error['location']}: {error['message']}")
```

### Integration with Build Systems

```python
# setup.py integration
from setuptools import setup
from setuptools.command.build import build
import subprocess
import os

class BuildWithStubs(build):
    """Custom build command that generates stubs."""
    
    def run(self):
        # Run normal build
        super().run()
        
        # Generate stubs for the package
        if self.should_generate_stubs():
            self.generate_stubs()
    
    def should_generate_stubs(self) -> bool:
        """Check if stubs should be generated."""
        return os.environ.get('GENERATE_STUBS', '').lower() == 'true'
    
    def generate_stubs(self):
        """Generate stubs for the package."""
        package_name = 'mypackage'  # Replace with actual package name
        
        print("Generating stub files...")
        result = subprocess.run([
            'stubgen',
            '-o', 'stubs',
            '-p', package_name
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Stub generation failed: {result.stderr}")
        else:
            print("Stub files generated successfully")

setup(
    name="mypackage",
    cmdclass={'build': BuildWithStubs},
    # ... other setup parameters
)
```

### CI/CD Integration

```yaml
# GitHub Actions workflow for stub validation
name: Validate Stubs

on: [push, pull_request]

jobs:
  test-stubs:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        pip install mypy
        pip install -e .  # Install the package
    
    - name: Generate stubs
      run: |
        stubgen -o stubs -p mypackage
    
    - name: Test stubs
      run: |
        stubtest --allowlist stubtest.allowlist mypackage
    
    - name: Upload stub artifacts
      uses: actions/upload-artifact@v2
      with:
        name: stubs
        path: stubs/
```

## Stub File Examples

### Basic Stub Structure

```python
# mymodule.pyi - Basic stub file
from typing import Any, Optional

def process_data(data: list[Any], options: Optional[dict[str, Any]] = ...) -> dict[str, Any]: ...

class DataProcessor:
    def __init__(self, config: dict[str, Any]) -> None: ...
    def process(self, data: Any) -> Any: ...
    @property
    def status(self) -> str: ...

ERROR_CODES: dict[str, int]
DEFAULT_CONFIG: dict[str, Any]
```

### Advanced Stub Features

```python
# advanced.pyi - Advanced stub features
from typing import Generic, TypeVar, Protocol, overload
from typing_extensions import ParamSpec, TypeVarTuple

T = TypeVar('T')
P = ParamSpec('P')
Ts = TypeVarTuple('Ts')

class Container(Generic[T]):
    def __init__(self, item: T) -> None: ...
    def get(self) -> T: ...
    def set(self, item: T) -> None: ...

class Callable(Protocol[P, T]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T: ...

@overload
def convert(value: int) -> str: ...
@overload  
def convert(value: str) -> int: ...
def convert(value: int | str) -> str | int: ...

# Variadic generic function
def process_multiple(*args: *Ts) -> tuple[*Ts]: ...
```

## Best Practices

### Stub Maintenance Workflow

```python
class StubMaintenanceWorkflow:
    """Automated workflow for stub maintenance."""
    
    def __init__(self, package_name: str):
        self.package_name = package_name
        self.stubs_dir = Path("stubs")
    
    def update_stubs(self) -> bool:
        """Update stubs and validate them."""
        # 1. Regenerate stubs
        if not self.regenerate_stubs():
            return False
        
        # 2. Test against runtime
        test_results = self.test_stubs()
        if not test_results['success']:
            # 3. Update allowlist if needed
            self.update_allowlist(test_results['errors'])
        
        # 4. Final validation
        return self.validate_stubs()
    
    def regenerate_stubs(self) -> bool:
        """Regenerate stub files."""
        cmd = [
            'stubgen',
            '-o', str(self.stubs_dir),
            '-p', self.package_name,
            '--include-private'
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0
    
    def test_stubs(self) -> dict:
        """Test stub accuracy."""
        tester = StubTester()
        return tester.test_stubs([self.package_name])
    
    def update_allowlist(self, errors: list[dict]):
        """Update allowlist based on test errors."""
        allowlist_entries = []
        
        for error in errors:
            if self.should_allow_error(error):
                allowlist_entries.append(error['location'])
        
        # Write updated allowlist
        with open('stubtest.allowlist', 'w') as f:
            f.write('\n'.join(allowlist_entries))
    
    def should_allow_error(self, error: dict) -> bool:
        """Determine if error should be allowlisted."""
        message = error['message'].lower()
        
        # Common patterns to allowlist
        allowlist_patterns = [
            'runtime argument',  # Runtime-only arguments
            'is not present at runtime',  # Stub-only definitions
            'incompatible default',  # Different default values
        ]
        
        return any(pattern in message for pattern in allowlist_patterns)
    
    def validate_stubs(self) -> bool:
        """Final validation of stub files."""
        # Run mypy on stub files themselves
        cmd = ['mypy', str(self.stubs_dir)]
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0

# Usage
workflow = StubMaintenanceWorkflow('mypackage')
success = workflow.update_stubs()
```

### Quality Assurance

```python
def validate_stub_quality(stub_file: Path) -> dict:
    """Validate stub file quality and completeness."""
    with open(stub_file) as f:
        content = f.read()
    
    issues = []
    
    # Check for common issues
    if 'Any' in content:
        any_count = content.count('Any')
        if any_count > 10:  # Threshold for too many Any types
            issues.append(f"Excessive use of Any ({any_count} occurrences)")
    
    if '...' not in content:
        issues.append("Missing ellipsis in function bodies")
    
    # Check for proper imports
    required_imports = []
    if 'Optional[' in content and 'from typing import' not in content:
        required_imports.append('Optional')
    
    if required_imports:
        issues.append(f"Missing imports: {', '.join(required_imports)}")
    
    return {
        'file': str(stub_file),
        'issues': issues,
        'quality_score': max(0, 100 - len(issues) * 10)
    }

# Usage
quality_report = validate_stub_quality(Path('stubs/mymodule.pyi'))
print(f"Quality score: {quality_report['quality_score']}/100")
```