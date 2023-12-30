import os
from tempfile import NamedTemporaryFile

os.environ["CODEFLASH_API_KEY"] = "test-key"
from codeflash.code_utils.code_replacer import replace_function_in_file


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

    with NamedTemporaryFile(mode="w") as f:
        f.write(original_code)
        f.flush()
        path = f.name
        function_name = "NewClass.new_function"
        preexisting_functions = ["NewClass.new_function"]
        optimized_code = optim_code

        new_code = replace_function_in_file(
            path, function_name, optimized_code, preexisting_functions
        )
        assert new_code == expected
