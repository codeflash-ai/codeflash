# Error System

Error codes, error handling classes, and error reporting functionality for comprehensive type checking diagnostics. Mypy's error system provides detailed, categorized error messages with precise location information.

## Capabilities

### Error Code System

Structured error codes that categorize different types of type checking errors.

```python { .api }
class ErrorCode:
    """
    Represents specific error types with descriptions and categories.
    
    Each error code has a unique name, human-readable description,
    and belongs to a category for organizing related errors.
    
    Attributes:
    - code: str - Unique error code identifier
    - description: str - Human-readable description
    - category: str - Error category for grouping
    """
```

### Core Error Codes

Essential error codes for common type checking issues.

```python { .api }
# Attribute and name errors
ATTR_DEFINED: ErrorCode
"""Accessing undefined attributes on objects."""

NAME_DEFINED: ErrorCode  
"""Using undefined names or variables."""

# Function call errors
CALL_ARG: ErrorCode
"""Function call argument errors (wrong number, missing, etc.)."""

ARG_TYPE: ErrorCode
"""Argument type mismatches in function calls."""

# Return value errors
RETURN: ErrorCode
"""Return statement errors."""

RETURN_VALUE: ErrorCode
"""Return value type mismatches."""

# Assignment errors
ASSIGNMENT: ErrorCode
"""Assignment type compatibility errors."""

# Method and inheritance errors
OVERRIDE: ErrorCode
"""Method override signature mismatches."""

# Generic type errors
TYPE_ARG: ErrorCode
"""Generic type argument errors."""

TYPE_VAR: ErrorCode
"""Type variable constraint violations."""

# Union type errors
UNION_ATTR: ErrorCode
"""Attribute access on union types where not all members have the attribute."""

# Container operation errors
INDEX: ErrorCode
"""Indexing operation errors (wrong key type, etc.)."""

OPERATOR: ErrorCode
"""Operator usage errors (unsupported operations)."""

# Container operation errors
LIST_ITEM: ErrorCode
"""List item type errors."""

DICT_ITEM: ErrorCode
"""Dictionary item type errors."""

TYPEDDICT_ITEM: ErrorCode
"""TypedDict item access errors."""

TYPEDDICT_UNKNOWN_KEY: ErrorCode
"""Unknown key access in TypedDict."""

# Import and module errors
IMPORT: ErrorCode
"""Import-related errors."""

IMPORT_NOT_FOUND: ErrorCode
"""Module not found errors."""

IMPORT_UNTYPED: ErrorCode
"""Importing from untyped modules."""

# Definition and redefinition errors
NO_REDEF: ErrorCode
"""Name redefinition errors."""

VAR_ANNOTATED: ErrorCode
"""Variable annotation errors."""

FUNC_RETURNS_VALUE: ErrorCode
"""Function return value errors."""

# Abstract class and method errors
ABSTRACT: ErrorCode
"""Abstract class instantiation errors."""

TYPE_ABSTRACT: ErrorCode
"""Abstract type usage errors."""

# String and literal errors
STRING_FORMATTING: ErrorCode
"""String formatting type errors."""

LITERAL_REQ: ErrorCode
"""Literal value requirement errors."""

STR_BYTES_PY3: ErrorCode
"""String/bytes compatibility errors in Python 3."""

# Async/await errors
UNUSED_COROUTINE: ErrorCode
"""Unused coroutine warnings."""

TOP_LEVEL_AWAIT: ErrorCode
"""Top-level await usage errors."""

AWAIT_NOT_ASYNC: ErrorCode
"""Await used outside async context."""

# Type checking strictness errors
NO_UNTYPED_CALL: ErrorCode
"""Calling untyped functions."""

NO_ANY_UNIMPORTED: ErrorCode
"""Any from unimported modules."""

NO_ANY_RETURN: ErrorCode
"""Functions returning Any."""

# Code quality and style errors
REDUNDANT_CAST: ErrorCode
"""Redundant type cast warnings."""

REDUNDANT_EXPR: ErrorCode
"""Redundant expression warnings."""

COMPARISON_OVERLAP: ErrorCode
"""Overlapping comparison warnings."""

UNUSED_IGNORE: ErrorCode
"""Unused type: ignore comments."""

UNUSED_AWAITABLE: ErrorCode
"""Unused awaitable expressions."""

# Advanced features
ASSERT_TYPE: ErrorCode
"""Assert type failures."""

SAFE_SUPER: ErrorCode
"""Super call safety checks."""

EXHAUSTIVE_MATCH: ErrorCode
"""Non-exhaustive match statements."""

EXPLICIT_OVERRIDE_REQUIRED: ErrorCode
"""Missing explicit override decorators."""

# Type annotation errors
ANNOTATION_UNCHECKED: ErrorCode
"""Issues with type annotations that can't be checked."""

VALID_TYPE: ErrorCode
"""Invalid type expressions in annotations."""

VALID_NEWTYPE: ErrorCode
"""NewType definition errors."""

HAS_TYPE: ErrorCode
"""Type existence checks."""

# Exit and control flow
EXIT_RETURN: ErrorCode
"""Exit function return type errors."""

# Miscellaneous errors
MISC: ErrorCode
"""Miscellaneous type checking errors."""

SYNTAX: ErrorCode
"""Python syntax errors."""

NO_RETURN: ErrorCode
"""Functions that should return but don't."""

UNREACHABLE: ErrorCode
"""Unreachable code detection."""

# Internal and debugging
FILE: ErrorCode
"""Internal marker for whole file ignoring."""

EXPLICIT_ANY: ErrorCode
"""Explicit Any usage warnings."""

UNIMPORTED_REVEAL: ErrorCode
"""reveal_type/reveal_locals usage warnings."""
```

### Error Handling Classes

Core classes for collecting, formatting, and reporting type checking errors.

```python { .api }
class Errors:
    """
    Collects and formats error messages during type checking.
    
    Central error reporting system that accumulates errors from
    various phases of analysis and formats them for output.
    
    Methods:
    - report(line: int, column: int, message: str, code: ErrorCode | None)
    - num_messages() -> int
    - is_errors() -> bool  
    - format_messages() -> list[str]
    """

class CompileError(Exception):
    """
    Exception raised when type checking fails with critical errors.
    
    Used for errors that prevent continuation of analysis,
    such as syntax errors or critical import failures.
    
    Attributes:
    - messages: list[str] - Error messages
    - use_stdout: bool - Whether to print to stdout
    """

class ErrorInfo:
    """
    Contains information about individual errors.
    
    Structured error data with precise location information
    and categorization for tools and IDEs.
    
    Attributes:
    - file: str - Source file path
    - line: int - Line number (1-based)
    - column: int - Column number (1-based)  
    - message: str - Error message text
    - severity: str - Error severity level
    - error_code: ErrorCode | None - Associated error code
    """
```

## Error Code Categories

### Type Compatibility Errors

```python
# Examples of common type compatibility errors

# ASSIGNMENT - Variable assignment type mismatch
x: int = "hello"  # Error: Incompatible types (expression has type "str", variable has type "int")

# ARG_TYPE - Function argument type mismatch  
def greet(name: str) -> str:
    return f"Hello, {name}!"

greet(42)  # Error: Argument 1 has incompatible type "int"; expected "str"

# RETURN_VALUE - Return value type mismatch
def get_number() -> int:
    return "not a number"  # Error: Incompatible return value type (got "str", expected "int")
```

### Attribute and Name Errors

```python
# NAME_DEFINED - Undefined variable
print(undefined_variable)  # Error: Name 'undefined_variable' is not defined

# ATTR_DEFINED - Undefined attribute
class Person:
    def __init__(self, name: str):
        self.name = name

person = Person("Alice")
print(person.age)  # Error: "Person" has no attribute "age"

# UNION_ATTR - Union attribute access
from typing import Union

def process(value: Union[str, int]) -> str:
    return value.upper()  # Error: Item "int" of "Union[str, int]" has no attribute "upper"
```

### Function Call Errors

```python
# CALL_ARG - Wrong number of arguments
def add(x: int, y: int) -> int:
    return x + y

add(1)  # Error: Missing positional argument "y" in call to "add"
add(1, 2, 3)  # Error: Too many positional arguments for "add"

# ARG_TYPE - Argument type mismatch
add("1", "2")  # Error: Argument 1 has incompatible type "str"; expected "int"
```

### Generic Type Errors

```python
from typing import List, TypeVar

# TYPE_ARG - Generic type argument errors
numbers: List = [1, 2, 3]  # Error: Missing type parameters for generic type "List"

# TYPE_VAR - Type variable constraint violations
T = TypeVar('T', int, str)  # T can only be int or str

def process(value: T) -> T:
    return value

process(3.14)  # Error: Value of type variable "T" cannot be "float"
```

## Error Reporting and Formatting

### Error Message Structure

```python
def parse_error_message(message: str) -> dict:
    """Parse mypy error message into components."""
    # Format: filename:line:column: level: message [error-code]
    import re
    
    pattern = r'([^:]+):(\d+):(\d+): (error|warning|note): (.+?)(?:\s+\[([^\]]+)\])?$'
    match = re.match(pattern, message)
    
    if match:
        filename, line, column, level, text, error_code = match.groups()
        return {
            'file': filename,
            'line': int(line),
            'column': int(column),
            'level': level,
            'message': text,
            'error_code': error_code
        }
    
    return {'raw_message': message}

# Example error message parsing
error_msg = "myfile.py:10:5: error: Incompatible types [assignment]"
parsed = parse_error_message(error_msg)
# {'file': 'myfile.py', 'line': 10, 'column': 5, 'level': 'error', 
#  'message': 'Incompatible types', 'error_code': 'assignment'}
```

### Custom Error Handling

```python
from mypy.errors import Errors
from mypy.errorcodes import ErrorCode

class CustomErrorReporter:
    """Custom error reporter for specialized error handling."""
    
    def __init__(self):
        self.errors = Errors()
        self.error_counts = {}
    
    def report_error(self, line: int, column: int, message: str, 
                    code: ErrorCode | None = None):
        """Report an error with custom processing."""
        self.errors.report(line, column, message, code)
        
        # Track error frequency
        if code:
            self.error_counts[code.code] = self.error_counts.get(code.code, 0) + 1
    
    def get_error_summary(self) -> dict:
        """Get summary of error types and counts."""
        return {
            'total_errors': self.errors.num_messages(),
            'error_breakdown': self.error_counts.copy(),
            'has_errors': self.errors.is_errors()
        }
    
    def format_errors_json(self) -> str:
        """Format errors as JSON for tool integration."""
        import json
        
        errors = []
        for msg in self.errors.format_messages():
            parsed = parse_error_message(msg)
            errors.append(parsed)
        
        return json.dumps({
            'errors': errors,
            'summary': self.get_error_summary()
        }, indent=2)

# Usage
reporter = CustomErrorReporter()
reporter.report_error(10, 5, "Type mismatch", ARG_TYPE)

summary = reporter.get_error_summary()
json_output = reporter.format_errors_json()
```

### Error Filtering and Processing

```python
from mypy.errorcodes import ErrorCode

class ErrorFilter:
    """Filter and categorize mypy errors."""
    
    def __init__(self):
        self.ignored_codes = set()
        self.severity_levels = {
            'critical': {SYNTAX, IMPORT},
            'high': {ARG_TYPE, RETURN_VALUE, ASSIGNMENT},
            'medium': {ATTR_DEFINED, NAME_DEFINED},
            'low': {MISC, ANNOTATION_UNCHECKED}
        }
    
    def ignore_error_code(self, code: ErrorCode):
        """Ignore specific error code."""
        self.ignored_codes.add(code)
    
    def should_report(self, code: ErrorCode | None) -> bool:
        """Check if error should be reported."""
        return code not in self.ignored_codes
    
    def get_severity(self, code: ErrorCode | None) -> str:
        """Get severity level for error code."""
        if not code:
            return 'unknown'
        
        for level, codes in self.severity_levels.items():
            if code in codes:
                return level
        
        return 'medium'  # Default severity
    
    def filter_errors(self, error_messages: list[str]) -> list[dict]:
        """Filter and categorize error messages."""
        filtered = []
        
        for msg in error_messages:
            parsed = parse_error_message(msg)
            
            if 'error_code' in parsed:
                code_name = parsed['error_code']
                # Find ErrorCode object by name
                code = next((c for c in globals().values() 
                           if isinstance(c, ErrorCode) and c.code == code_name), None)
                
                if self.should_report(code):
                    parsed['severity'] = self.get_severity(code)
                    filtered.append(parsed)
        
        return filtered

# Usage
error_filter = ErrorFilter()
error_filter.ignore_error_code(MISC)  # Ignore miscellaneous errors

errors = [
    "myfile.py:10:5: error: Incompatible types [assignment]",
    "myfile.py:15:2: error: Missing import [misc]"
]

filtered = error_filter.filter_errors(errors)
# Only assignment error will be included, misc error filtered out
```

## Integration with Development Tools

### IDE Integration

```python
class IDEErrorReporter:
    """Error reporter optimized for IDE integration."""
    
    def __init__(self):
        self.diagnostics = []
    
    def process_mypy_output(self, output: str) -> list[dict]:
        """Convert mypy output to IDE diagnostic format."""
        diagnostics = []
        
        for line in output.strip().split('\n'):
            if not line:
                continue
                
            parsed = parse_error_message(line)
            if 'file' in parsed:
                diagnostic = {
                    'range': {
                        'start': {
                            'line': parsed['line'] - 1,  # 0-based for LSP
                            'character': parsed['column'] - 1
                        },
                        'end': {
                            'line': parsed['line'] - 1,
                            'character': parsed['column'] + 10  # Approximate end
                        }
                    },
                    'severity': 1 if parsed['level'] == 'error' else 2,  # LSP severity
                    'message': parsed['message'],
                    'source': 'mypy',
                    'code': parsed.get('error_code', 'unknown')
                }
                diagnostics.append(diagnostic)
        
        return diagnostics

# Usage in LSP server
ide_reporter = IDEErrorReporter()
diagnostics = ide_reporter.process_mypy_output(mypy_output)
# Send diagnostics to IDE client
```

### CI/CD Integration

```python
import subprocess
import json

class CIErrorReporter:
    """Error reporter for CI/CD pipelines."""
    
    def run_mypy_with_json_output(self, files: list[str]) -> dict:
        """Run mypy and return structured error data."""
        try:
            result = subprocess.run(
                ['mypy', '--show-error-codes'] + files,
                capture_output=True,
                text=True
            )
            
            errors = []
            if result.stderr:
                for line in result.stderr.strip().split('\n'):
                    parsed = parse_error_message(line)
                    if 'file' in parsed:
                        errors.append(parsed)
            
            return {
                'success': result.returncode == 0,
                'exit_code': result.returncode,
                'errors': errors,
                'error_count': len(errors),
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except FileNotFoundError:
            return {
                'success': False,
                'error': 'mypy not found - please install mypy'
            }
    
    def generate_junit_xml(self, results: dict, output_file: str):
        """Generate JUnit XML report for CI systems."""
        import xml.etree.ElementTree as ET
        
        testsuite = ET.Element('testsuite', {
            'name': 'mypy',
            'tests': str(len(results.get('errors', []))),
            'failures': str(len([e for e in results.get('errors', []) 
                                if e.get('level') == 'error'])),
            'errors': '0'
        })
        
        for error in results.get('errors', []):
            testcase = ET.SubElement(testsuite, 'testcase', {
                'name': f"{error['file']}:{error['line']}",
                'classname': 'mypy'
            })
            
            if error.get('level') == 'error':
                failure = ET.SubElement(testcase, 'failure', {
                    'message': error['message'],
                    'type': error.get('error_code', 'unknown')
                })
                failure.text = f"Line {error['line']}: {error['message']}"
        
        tree = ET.ElementTree(testsuite)
        tree.write(output_file, encoding='utf-8', xml_declaration=True)

# Usage in CI pipeline
ci_reporter = CIErrorReporter()
results = ci_reporter.run_mypy_with_json_output(['src/'])

if not results['success']:
    print(f"Type checking failed with {results['error_count']} errors")
    ci_reporter.generate_junit_xml(results, 'mypy-results.xml')
    exit(1)
```

## Error Code Reference

### Complete Error Code List

```python
# Core type errors
ASSIGNMENT = ErrorCode("assignment", "Assignment type mismatch", "types")
ARG_TYPE = ErrorCode("arg-type", "Argument type mismatch", "types")
RETURN_VALUE = ErrorCode("return-value", "Return value type mismatch", "types")

# Attribute and name errors
ATTR_DEFINED = ErrorCode("attr-defined", "Accessing undefined attributes", "names")
NAME_DEFINED = ErrorCode("name-defined", "Using undefined names", "names")

# Function call errors
CALL_ARG = ErrorCode("call-arg", "Function call argument errors", "calls")
TYPE_ARG = ErrorCode("type-arg", "Generic type argument errors", "generics")

# Advanced type system errors
UNION_ATTR = ErrorCode("union-attr", "Union attribute access errors", "unions")
OVERRIDE = ErrorCode("override", "Method override errors", "inheritance")
INDEX = ErrorCode("index", "Indexing operation errors", "operators")
OPERATOR = ErrorCode("operator", "Operator usage errors", "operators")

# Import and module errors
IMPORT = ErrorCode("import", "Import-related errors", "imports")
IMPORT_UNTYPED = ErrorCode("import-untyped", "Untyped module imports", "imports")

# Annotation and syntax errors
VALID_TYPE = ErrorCode("valid-type", "Invalid type expressions", "annotations")
ANNOTATION_UNCHECKED = ErrorCode("annotation-unchecked", "Unchecked annotations", "annotations")
SYNTAX = ErrorCode("syntax", "Python syntax errors", "syntax")

# Control flow errors
RETURN = ErrorCode("return", "Return statement errors", "control-flow")
NO_RETURN = ErrorCode("no-return", "Missing return statements", "control-flow")
UNREACHABLE = ErrorCode("unreachable", "Unreachable code", "control-flow")

# Miscellaneous
MISC = ErrorCode("misc", "Miscellaneous errors", "general")
```