import argparse
import os
import tempfile
from pathlib import Path

import pytest

from codeflash.code_utils.config_parser import parse_config_file
from codeflash.code_utils.formatter import format_code, sort_imports

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.verification_utils import TestConfig

def test_remove_duplicate_imports():
    """Test that duplicate imports are removed when should_sort_imports is True."""
    original_code = "import os\nimport os\n"
    new_code = sort_imports(original_code)
    assert new_code == "import os\n"


def test_remove_multiple_duplicate_imports():
    """Test that multiple duplicate imports are removed when should_sort_imports is True."""
    original_code = "import sys\nimport os\nimport sys\n"

    new_code = sort_imports(original_code)
    assert new_code == "import os\nimport sys\n"


def test_sorting_imports():
    """Test that imports are sorted when should_sort_imports is True."""
    original_code = "import sys\nimport unittest\nimport os\n"

    new_code = sort_imports(original_code)
    assert new_code == "import os\nimport sys\nimport unittest\n"


def test_sort_imports_without_formatting():
    """Test that imports are sorted when formatting is disabled and should_sort_imports is True."""
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(b"import sys\nimport unittest\nimport os\n")
        tmp.flush()
        tmp_path = Path(tmp.name)

        new_code = format_code(formatter_cmds=["disabled"], path=tmp_path)
        assert new_code is not None
        new_code = sort_imports(new_code)
        assert new_code == "import os\nimport sys\nimport unittest\n"


def test_dedup_and_sort_imports_deduplicates():
    original_code = """
import os
import sys


def foo():
    return os.path.join(sys.path[0], 'bar')
"""

    expected = """
import os
import sys


def foo():
    return os.path.join(sys.path[0], 'bar')
"""

    actual = sort_imports(original_code)

    assert actual == expected


def test_dedup_and_sort_imports_sorts_and_deduplicates():
    original_code = """
import os
import sys
import json
import os


def foo():
    return os.path.join(sys.path[0], 'bar')
"""

    expected = """
import json
import os
import sys


def foo():
    return os.path.join(sys.path[0], 'bar')
"""

    actual = sort_imports(original_code)

    assert actual == expected


def test_formatter_cmds_non_existent():
    """Test that default formatter-cmds is used when it doesn't exist in the toml."""
    config_data = """
[tool.codeflash]
module-root = "src"
tests-root = "tests"
test-framework = "pytest"
ignore-paths = []
"""

    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as tmp:
        tmp.write(config_data.encode())
        tmp.flush()
        tmp_path = Path(tmp.name)

    try:
        config, _ = parse_config_file(tmp_path)
        assert config["formatter_cmds"] == ["black $file"]
    finally:
        os.remove(tmp_path)

    try:
        import black
    except ImportError:
        pytest.skip("black is not installed")

    original_code = b"""
import os
import sys
def foo():
    return os.path.join(sys.path[0], 'bar')"""
    expected = """import os
import sys


def foo():
    return os.path.join(sys.path[0], "bar")
"""
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(original_code)
        tmp.flush()
        tmp_path = tmp.name

        actual = format_code(formatter_cmds=["black $file"], path=Path(tmp_path))
        assert actual == expected


def test_formatter_black():
    try:
        import black
    except ImportError:
        pytest.skip("black is not installed")
    original_code = b"""
import os
import sys    
def foo():
    return os.path.join(sys.path[0], 'bar')"""
    expected = """import os
import sys


def foo():
    return os.path.join(sys.path[0], "bar")
"""
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(original_code)
        tmp.flush()
        tmp_path = tmp.name

        actual = format_code(formatter_cmds=["black $file"], path=Path(tmp_path))
        assert actual == expected


def test_formatter_ruff():
    try:
        import ruff  # type: ignore
    except ImportError:
        pytest.skip("ruff is not installed")
    original_code = b"""
import os
import sys    
def foo():
    return os.path.join(sys.path[0], 'bar')"""
    expected = """import os
import sys


def foo():
    return os.path.join(sys.path[0], "bar")
"""
    with tempfile.NamedTemporaryFile(suffix=".py") as tmp:
        tmp.write(original_code)
        tmp.flush()
        tmp_path = tmp.name

        actual = format_code(
            formatter_cmds=["ruff check --exit-zero --fix $file", "ruff format $file"], path=Path(tmp_path)
        )
        assert actual == expected


def test_formatter_error():
    original_code = """
import os
import sys
def foo():
    return os.path.join(sys.path[0], 'bar')"""
    expected = original_code
    with tempfile.NamedTemporaryFile("w") as tmp:
        tmp.write(original_code)
        tmp.flush()
        tmp_path = tmp.name
        with pytest.raises(FileNotFoundError):
            format_code(formatter_cmds=["exit 1"], path=Path(tmp_path))

############################################################
################ CST based formatting tests ################
############################################################
@pytest.fixture
def setup_cst_formatter_args():
    """Common setup for reformat_code_and_helpers tests."""
    def _setup(unformatted_code, function_name):
        test_dir = Path(tempfile.mkdtemp())
        target_path = test_dir / "target.py"
        target_path.write_text(unformatted_code, encoding="utf-8")
        
        function_to_optimize = FunctionToOptimize(
            function_name=function_name, 
            parents=[], 
            file_path=target_path
        )
        
        test_cfg = TestConfig(
            tests_root=test_dir,
            project_root_path=test_dir,
            test_framework="pytest",
            tests_project_rootdir=test_dir,
        )
        
        args = argparse.Namespace(
            disable_imports_sorting=False,
            formatter_cmds=[
                "ruff check --exit-zero --fix $file",
                "ruff format $file"
            ],
        )
        
        optimizer = FunctionOptimizer(
            function_to_optimize=function_to_optimize,
            test_cfg=test_cfg,
            args=args,
        )
        
        return optimizer, target_path, function_to_optimize
    
    yield _setup


def test_reformat_code_and_helpers(setup_cst_formatter_args):
    """
    reformat_code_and_helpers should only format the code that is optimized not the whole file, to avoid large diffing
    """
    unformatted_code = """import sys


def lol():
    print(       "lol" )




class MyClass:
    def __init__(self, x=0):
        self.x = x

    def lol(self):
        print(       "lol" )

    def lol2    (self):
        print(       " lol2" )"""
    
    expected_code = """import sys


def lol():
    print(       "lol" )




class MyClass:
    def __init__(self, x=0):
        self.x = x

    def lol(self):
        print(       "lol" )

    def lol2(self):
        print(" lol2")
"""
    
    optimizer, target_path, function_to_optimize = setup_cst_formatter_args(
        unformatted_code, "MyClass.lol2"
    )
    
    formatted_code, _ = optimizer.reformat_code_and_helpers(
        helper_functions=[],
        path=target_path,
        original_code=optimizer.function_to_optimize_source_code,
        opt_func_name=function_to_optimize.function_name
    )
    
    assert formatted_code == expected_code


def test_reformat_code_and_helpers_with_duplicated_target_function_names(setup_cst_formatter_args):
    unformatted_code = """import sys
def lol():
    print(       "lol" )

class MyClass:
    def __init__(self, x=0):
        self.x = x

    def lol(self):
        print(       "lol" )"""
    
    expected_code = """import sys
def lol():
    print(       "lol" )

class MyClass:
    def __init__(self, x=0):
        self.x = x

    def lol(self):
        print("lol")
"""
    
    optimizer, target_path, function_to_optimize = setup_cst_formatter_args(
        unformatted_code, "MyClass.lol"
    )
    
    formatted_code, _ = optimizer.reformat_code_and_helpers(
        helper_functions=[],
        path=target_path,
        original_code=optimizer.function_to_optimize_source_code,
        opt_func_name=function_to_optimize.function_name
    )
    
    assert formatted_code == expected_code



def test_formatting_nested_functions(setup_cst_formatter_args):
    unformatted_code = """def hello():
    print("Hello")
    def nested_function()   :
        print       ("This is a nested function")
    def another_nested_function():
        print   ("This is another nested function")"""
    
    expected_code = """def hello():
    print("Hello")
    def nested_function():
        print("This is a nested function")
    def another_nested_function():
        print   ("This is another nested function")"""
    
    optimizer, target_path, function_to_optimize = setup_cst_formatter_args(
        unformatted_code, "hello.nested_function"
    )
    
    formatted_code, _ = optimizer.reformat_code_and_helpers(
        helper_functions=[],
        path=target_path,
        original_code=optimizer.function_to_optimize_source_code,
        opt_func_name=function_to_optimize.function_name
    )
    
    assert formatted_code == expected_code


def test_formatting_standalone_functions(setup_cst_formatter_args):
    unformatted_code = """def func1   ():
    print(      "This is a function with bad formatting")
def func2() :
    print   (   "This is another function with bad formatting"  )
"""
    
    expected_code = """def func1   ():
    print(      "This is a function with bad formatting")
def func2():
    print("This is another function with bad formatting")
"""
    
    optimizer, target_path, function_to_optimize = setup_cst_formatter_args(
        unformatted_code, "func2"
    )
    
    formatted_code, _ = optimizer.reformat_code_and_helpers(
        helper_functions=[],
        path=target_path,
        original_code=optimizer.function_to_optimize_source_code,
        opt_func_name=function_to_optimize.function_name
    )
    
    assert formatted_code == expected_code


def test_formatting_function_with_decorators(setup_cst_formatter_args):
   unformatted_code = """@decorator1
@decorator2(   arg1 , arg2   )
def func1   ():
   print(      "This is a function with bad formatting")

@another_decorator(     arg)
def func2   (  x,y  ):
   print   (   "This is another function with bad formatting"  )"""
   
   expected_code = """@decorator1
@decorator2(   arg1 , arg2   )
def func1   ():
   print(      "This is a function with bad formatting")

@another_decorator(arg)
def func2(x, y):
    print("This is another function with bad formatting")
"""
   
   optimizer, target_path, function_to_optimize = setup_cst_formatter_args(
       unformatted_code, "func2"
   )
   
   formatted_code, _ = optimizer.reformat_code_and_helpers(
       helper_functions=[],
       path=target_path,
       original_code=optimizer.function_to_optimize_source_code,
       opt_func_name=function_to_optimize.function_name
   )
   
   assert formatted_code == expected_code


def test_formatting_function_with_syntax_error(setup_cst_formatter_args):
    """shouldn't happen anyway, but just in case"""
    unformatted_code = """def func1():
    print("This is a function with a syntax error"
def func2():
    print("This is another function with a syntax error")
"""
    
    expected_code = unformatted_code  # No formatting should be applied due to syntax error
    
    optimizer, target_path, function_to_optimize = setup_cst_formatter_args(
        unformatted_code, "func2"
    )
    
    formatted_code, _ = optimizer.reformat_code_and_helpers(
        helper_functions=[],
        path=target_path,
        original_code=optimizer.function_to_optimize_source_code,
        opt_func_name=function_to_optimize.function_name
    )
    
    assert formatted_code == expected_code
