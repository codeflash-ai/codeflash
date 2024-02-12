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
        original_code, [function_name], optim_code, preexisting_functions,
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

    function_names: list[str] = ["module.other_function"]
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

    function_names: list[str] = ["module.yet_another_function", "module.other_function"]
    preexisting_functions: list[str] = []
    new_code: str = replace_functions_in_file(
        original_code, function_names, optim_code, preexisting_functions
    )
    assert new_code == expected


def test_test_libcst_code_replacement5():
    optim_code = """def sorter_deps(arr):
    supersort(badsort(arr))
    return arr

def badsort(ploc):
    donothing(ploc)
    
def supersort(doink):
    for i in range(len(doink)):
        fix(doink, i)
"""

    original_code = """from code_to_optimize.bubble_sort_dep1_helper import dep1_comparer
from code_to_optimize.bubble_sort_dep2_swap import dep2_swap

def sorter_deps(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if dep1_comparer(arr, j):
                dep2_swap(arr, j)
    return arr
"""
    expected = """from code_to_optimize.bubble_sort_dep1_helper import dep1_comparer
from code_to_optimize.bubble_sort_dep2_swap import dep2_swap
def sorter_deps(arr):
    supersort(badsort(arr))
    return arr

def badsort(ploc):
    donothing(ploc)
    
def supersort(doink):
    for i in range(len(doink)):
        fix(doink, i)
"""

    function_names: list[str] = ["sorter_deps"]
    preexisting_functions: list[str] = ["sorter_deps"]
    new_code: str = replace_functions_in_file(
        original_code, function_names, optim_code, preexisting_functions
    )
    assert new_code == expected


def test_test_libcst_code_replacement6():
    optim_code = """import libcst as cst
from typing import Optional

def other_function(st):
    return(st * blob(st))

def blob(st):
    return(st * 2)
"""
    original_code_main = """import libcst as cst
from typing import Mandatory
from dependent import blob

print("Au revoir")

def yet_another_function(values):
    return len(values)

def other_function(st):
    return(st + blob(st))

print("Salut monde")
"""

    original_code_dependent = """import numpy as np

print("Cool")

def blob(values):
    return len(values)

def blab(st):
    return(st + st)

print("Not cool")
"""
    expected_main = """import libcst as cst
from typing import Optional
import libcst as cst
from typing import Mandatory
from dependent import blob

print("Au revoir")

def yet_another_function(values):
    return len(values)

def other_function(st):
    return(st * blob(st))

print("Salut monde")
"""

    expected_dependent = """import libcst as cst
from typing import Optional
import numpy as np

print("Cool")

def blob(st):
    return(st * 2)

def blab(st):
    return(st + st)

print("Not cool")
"""
    new_main_code: str = replace_functions_in_file(
        original_code_main, ["other_function"], optim_code, ["other_function", "yet_another_function", "blob"]
    )
    assert new_main_code == expected_main

    new_dependent_code: str = replace_functions_in_file(
        original_code_dependent, ["blob"], optim_code, []
    )
    assert new_dependent_code == expected_dependent
