import os

from codeflash.code_utils.code_replacer import replace_function_in_file

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
    new_code: str = replace_function_in_file(
        original_code, function_name, optim_code, preexisting_functions
    )
    assert new_code == expected
