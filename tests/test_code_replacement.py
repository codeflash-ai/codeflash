import os

from codeflash.code_utils.code_replacer import replace_functions_in_file

os.environ["CODEFLASH_API_KEY"] = "cf-test-key"


def test_test_libcst_code_replacement():
    optim_code = """import libcst as cst
from typing import Optional

def totally_new_function(value):
    return value

class NewClass:
    def __init__(self, name):
        self.name = name
    def new_function(self, value):
        return self.name
    def new_function2(value):
        return value
    """

    original_code = """class NewClass:
    def __init__(self, name):
        self.name = name
    def new_function(self, value):
        return "I am still old"

print("Hello world")
"""
    expected = """import libcst as cst
from typing import Optional
class NewClass:
    def __init__(self, name):
        self.name = name
    def new_function(self, value):
        return self.name
    def new_function2(value):
        return value

def totally_new_function(value):
    return value

print("Hello world")
"""

    function_name: str = "NewClass.new_function"
    preexisting_functions: list[str] = ["NewClass.new_function"]
    new_code: str = replace_functions_in_file(
        original_code, [function_name], optim_code, preexisting_functions
    )
    assert new_code == expected


def test_test_libcst_code_replacement2():
    optim_code = """import libcst as cst
from typing import Optional

def totally_new_function(value):
    return value

def other_function(st):
    return(st * 2)
    
class NewClass:
    def __init__(self, name):
        self.name = name
    def new_function(self, value):
        return other_function(self.name)
    def new_function2(value):
        return value
    """

    original_code = """from OtherModule import other_function

class NewClass:
    def __init__(self, name):
        self.name = name
    def new_function(self, value):
        return other_function("I am still old")

print("Hello world")
"""
    expected = """import libcst as cst
from typing import Optional
from OtherModule import other_function

class NewClass:
    def __init__(self, name):
        self.name = name
    def new_function(self, value):
        return other_function(self.name)
    def new_function2(value):
        return value

def totally_new_function(value):
    return value

print("Hello world")
"""

    function_name: str = "NewClass.new_function"
    preexisting_functions: list[str] = ["NewClass.new_function", "other_function"]
    new_code: str = replace_functions_in_file(
        original_code, [function_name], optim_code, preexisting_functions
    )
    assert new_code == expected


def test_test_libcst_code_replacement3():
    optim_code = """import libcst as cst
from typing import Optional

def totally_new_function(value):
    return value

def other_function(st):
    return(st * 2)

class NewClass:
    def __init__(self, name):
        self.name = name
    def new_function(self, value):
        return other_function(self.name)
    def new_function2(value):
        return value
    """

    original_code = """import libcst as cst
from typing import Mandatory

print("Au revoir")

def yet_another_function(values):
    return len(values)

def other_function(st):
    return(st + st)

print("Salut monde")
"""
    expected = """import libcst as cst
from typing import Optional
import libcst as cst
from typing import Mandatory

print("Au revoir")

def yet_another_function(values):
    return len(values)

def other_function(st):
    return(st * 2)

print("Salut monde")
"""

    function_names: list[str] = ["other_function"]
    preexisting_functions: list[str] = []
    new_code: str = replace_functions_in_file(
        original_code, function_names, optim_code, preexisting_functions
    )
    assert new_code == expected


def test_test_libcst_code_replacement4():
    optim_code = """import libcst as cst
from typing import Optional

def totally_new_function(value):
    return value
    
def yet_another_function(values):
    return len(values) + 2

def other_function(st):
    return(st * 2)

class NewClass:
    def __init__(self, name):
        self.name = name
    def new_function(self, value):
        return other_function(self.name)
    def new_function2(value):
        return value
    """

    original_code = """import libcst as cst
from typing import Mandatory

print("Au revoir")

def yet_another_function(values):
    return len(values)

def other_function(st):
    return(st + st)

print("Salut monde")
"""
    expected = """import libcst as cst
from typing import Optional
import libcst as cst
from typing import Mandatory

print("Au revoir")
def yet_another_function(values):
    return len(values) + 2

def other_function(st):
    return(st * 2)

print("Salut monde")
"""

    function_names: list[str] = ["yet_another_function", "other_function"]
    preexisting_functions: list[str] = []
    new_code: str = replace_functions_in_file(
        original_code, function_names, optim_code, preexisting_functions
    )
    assert new_code == expected
