import tempfile
from argparse import Namespace
from pathlib import Path

import libcst as cst

from codeflash.context.code_context_extractor import get_code_optimization_context
from codeflash.context.unused_definition_remover import remove_unused_definitions_by_function_names
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import FunctionParent
from codeflash.optimization.optimizer import Optimizer


def test_variable_removal_only() -> None:
    """Test that only variables not used by specified functions are removed, not functions."""
    code = """
def main_function():
    return USED_CONSTANT + 10

def helper_function():
    return 42

USED_CONSTANT = 42
UNUSED_CONSTANT = 123

def another_function():
    return UNUSED_CONSTANT
"""

    expected = """
def main_function():
    return USED_CONSTANT + 10

def helper_function():
    return 42

USED_CONSTANT = 42

def another_function():
    return UNUSED_CONSTANT
"""

    qualified_functions = {"main_function"}
    result = remove_unused_definitions_by_function_names(code, qualified_functions)
    # Normalize whitespace for comparison
    assert result.strip() == expected.strip()


def test_class_variable_removal() -> None:
    """Test that only class variables not used by specified functions are removed, not methods."""
    code = """
class MyClass:
    CLASS_USED = "used value"
    CLASS_UNUSED = "unused value"

    def __init__(self):
        self.value = self.CLASS_USED
        self.other = self.CLASS_UNUSED

    def used_method(self):
        return self.value

    def unused_method(self):
        return "Not used but not removed"

GLOBAL_USED = "global used"
GLOBAL_UNUSED = "global unused"

def helper_function():
    return MyClass().used_method() + GLOBAL_USED
"""

    expected = """
class MyClass:
    CLASS_USED = "used value"
    CLASS_UNUSED = "unused value"

    def __init__(self):
        self.value = self.CLASS_USED
        self.other = self.CLASS_UNUSED

    def used_method(self):
        return self.value

    def unused_method(self):
        return "Not used but not removed"

GLOBAL_USED = "global used"

def helper_function():
    return MyClass().used_method() + GLOBAL_USED
"""

    qualified_functions = {"helper_function"}
    result = remove_unused_definitions_by_function_names(code, qualified_functions)
    # Normalize whitespace for comparison
    assert result.strip() == expected.strip()


def test_complex_variable_dependencies() -> None:
    """Test that only variables with complex dependencies are properly handled."""
    code = """
def main_function():
    return DIRECT_DEPENDENCY

def unused_function():
    return "Not used but not removed"

DIRECT_DEPENDENCY = INDIRECT_DEPENDENCY + "_suffix"
INDIRECT_DEPENDENCY = "base value"
UNUSED_VARIABLE = "This should be removed"

TUPLE_USED, TUPLE_UNUSED = ("used", "unused")

def tuple_user():
    return TUPLE_USED
"""

    expected = """
def main_function():
    return DIRECT_DEPENDENCY

def unused_function():
    return "Not used but not removed"

DIRECT_DEPENDENCY = INDIRECT_DEPENDENCY + "_suffix"
INDIRECT_DEPENDENCY = "base value"

def tuple_user():
    return TUPLE_USED
"""

    qualified_functions = {"main_function"}
    result = remove_unused_definitions_by_function_names(code, qualified_functions)
    assert result.strip() == expected.strip()


def test_type_annotation_usage() -> None:
    """Test that variables used in type annotations are considered used."""
    code = """
# Type definition
CustomType = int
UnusedType = str

def main_function(param: CustomType) -> CustomType:
    return param + 10

def unused_function(param: UnusedType) -> UnusedType:
    return param + " suffix"

UNUSED_CONSTANT = 123
"""

    expected = """
# Type definition
CustomType = int

def main_function(param: CustomType) -> CustomType:
    return param + 10

def unused_function(param: UnusedType) -> UnusedType:
    return param + " suffix"

"""

    qualified_functions = {"main_function"}
    result = remove_unused_definitions_by_function_names(code, qualified_functions)
    # Normalize whitespace for comparison
    assert result.strip() == expected.strip()


def test_class_method_with_dunder_methods() -> None:
    """Test that when a class method is used, dunder methods of that class are preserved."""
    code = """
class MyClass:
    CLASS_VAR = "class variable"
    UNUSED_VAR = GLOBAL_VAR_2

    def __init__(self, value):
        self.value = GLOBAL_VAR

    def __str__(self):
        return f"MyClass({self.value})"

    def target_method(self):
        return self.value * 2

    def unused_method(self):
        return "Not used"

GLOBAL_VAR = "global"
GLOBAL_VAR_2 = "global"
UNUSED_GLOBAL = "unused global"

def helper_function():
    obj = MyClass(5)
    return obj.target_method()
"""

    expected = """
class MyClass:
    CLASS_VAR = "class variable"
    UNUSED_VAR = GLOBAL_VAR_2

    def __init__(self, value):
        self.value = GLOBAL_VAR

    def __str__(self):
        return f"MyClass({self.value})"

    def target_method(self):
        return self.value * 2

    def unused_method(self):
        return "Not used"

GLOBAL_VAR = "global"
GLOBAL_VAR_2 = "global"

def helper_function():
    obj = MyClass(5)
    return obj.target_method()
"""

    qualified_functions = {"MyClass.target_method"}
    result = remove_unused_definitions_by_function_names(code, qualified_functions)
    # Normalize whitespace for comparison
    assert result.strip() == expected.strip()


def test_complex_type_annotations() -> None:
    """Test complex type annotations with nested types."""
    code = """
from typing import List, Dict, Optional

# Type aliases
ItemType = Dict[str, int]
ResultType = List[ItemType]
UnusedType = Optional[str]

def process_data(items: ResultType) -> int:
    total = 0
    for item in items:
        for key, value in item.items():
            total += value
    return total

def unused_function(param: UnusedType) -> None:
    pass

# Variables
SAMPLE_DATA: ResultType = [{"a": 1, "b": 2}]
UNUSED_DATA: UnusedType = None
"""

    expected = """
from typing import List, Dict, Optional

# Type aliases
ItemType = Dict[str, int]
ResultType = List[ItemType]

def process_data(items: ResultType) -> int:
    total = 0
    for item in items:
        for key, value in item.items():
            total += value
    return total

def unused_function(param: UnusedType) -> None:
    pass
"""

    qualified_functions = {"process_data"}
    result = remove_unused_definitions_by_function_names(code, qualified_functions)
    assert result.strip() == expected.strip()


def test_try_except_finally_variables() -> None:
    """Test handling of variables defined in try-except-finally blocks."""
    code = """
import math
import os

# Top-level try-except that defines variables
try:
    MATH_CONSTANT = math.pi
    USED_ERROR_MSG = "An error occurred"
    UNUSED_CONST = 42
except ImportError:
    MATH_CONSTANT = 3.14
    USED_ERROR_MSG = "Math module not available"
    UNUSED_CONST = 0
finally:
    CLEANUP_FLAG = True
    UNUSED_CLEANUP = "Not used"

def use_constants():
    return f"Pi is approximately {MATH_CONSTANT}, message: {USED_ERROR_MSG}"

def use_cleanup():
    if CLEANUP_FLAG:
        return "Cleanup performed"
    return "No cleanup"

def unused_function():
    return UNUSED_CONST
"""

    expected = """
import math
import os

# Top-level try-except that defines variables
try:
    MATH_CONSTANT = math.pi
    USED_ERROR_MSG = "An error occurred"
except ImportError:
    MATH_CONSTANT = 3.14
    USED_ERROR_MSG = "Math module not available"
finally:
    CLEANUP_FLAG = True

def use_constants():
    return f"Pi is approximately {MATH_CONSTANT}, message: {USED_ERROR_MSG}"

def use_cleanup():
    if CLEANUP_FLAG:
        return "Cleanup performed"
    return "No cleanup"

def unused_function():
    return UNUSED_CONST
"""

    qualified_functions = {"use_constants", "use_cleanup"}
    result = remove_unused_definitions_by_function_names(code, qualified_functions)
    assert result.strip() == expected.strip()

def test_conditional_and_loop_variables() -> None:
    """Test handling of variables defined in if-else and while loops."""
    code = """
import sys
import platform

# Top-level if-else block defining variables
if sys.platform.startswith('win'):
    OS_TYPE = "Windows"
    OS_SEP = ""
    UNUSED_WIN_VAR = "Unused Windows variable"
elif sys.platform.startswith('linux'):
    OS_TYPE = "Linux"
    OS_SEP = "/"
    UNUSED_LINUX_VAR = "Unused Linux variable"
else:
    OS_TYPE = "Other"
    OS_SEP = "/"
    UNUSED_OTHER_VAR = "Unused other variable"

# While loop with variable definitions
counter = 0
while counter < 5:
    LOOP_RESULT = "Iteration " + str(counter)
    UNUSED_LOOP_VAR = "Unused loop " + str(counter)
    counter += 1

def get_platform_info():
    return "OS: " + OS_TYPE + ", Separator: " + OS_SEP

def get_loop_result():
    return LOOP_RESULT

def unused_function():
    result = ""
    if sys.platform.startswith('win'):
        result = UNUSED_WIN_VAR
    elif sys.platform.startswith('linux'):
        result = UNUSED_LINUX_VAR
    else:
        result = UNUSED_OTHER_VAR
    return result
"""

    expected = """
import sys
import platform

# Top-level if-else block defining variables
if sys.platform.startswith('win'):
    OS_TYPE = "Windows"
    OS_SEP = ""
elif sys.platform.startswith('linux'):
    OS_TYPE = "Linux"
    OS_SEP = "/"
else:
    OS_TYPE = "Other"
    OS_SEP = "/"

# While loop with variable definitions
counter = 0
while counter < 5:
    LOOP_RESULT = "Iteration " + str(counter)
    counter += 1

def get_platform_info():
    return "OS: " + OS_TYPE + ", Separator: " + OS_SEP

def get_loop_result():
    return LOOP_RESULT

def unused_function():
    result = ""
    if sys.platform.startswith('win'):
        result = UNUSED_WIN_VAR
    elif sys.platform.startswith('linux'):
        result = UNUSED_LINUX_VAR
    else:
        result = UNUSED_OTHER_VAR
    return result
"""

    qualified_functions = {"get_platform_info", "get_loop_result"}
    result = remove_unused_definitions_by_function_names(code, qualified_functions)
    assert result.strip() == expected.strip()
