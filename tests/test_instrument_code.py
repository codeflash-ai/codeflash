from pathlib import Path

from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import FunctionParent
from codeflash.verification.instrument_code import instrument_code


def test_add_codeflash_capture():
    # Test input code
    original_code = """
class MyClass:
    def __init__(self):
        self.x = 1

    def target_function(self):
        return self.x + 1
"""

    expected = f"""
from codeflash.verification.codeflash_capture import codeflash_capture

class MyClass:

    @codeflash_capture(function_name='target_function', tmp_dir_path='{get_run_tmp_file(Path("test_return_values"))!s}')
    def __init__(self):
        self.x = 1

    def target_function(self):
        return self.x + 1
"""

    # Create and modify test file
    test_file = Path("test_file.py")
    test_file.write_text(original_code)

    function = FunctionToOptimize(
        function_name="target_function",
        file_path=Path("test_file.py"),
        parents=[FunctionParent(type="ClassDef", name="MyClass")],
    )

    try:
        # Run the instrumentation
        instrument_code(function)

        # Check the result
        modified_code = test_file.read_text()
        assert modified_code.strip() == expected.strip()

    finally:
        # Cleanup
        test_file.unlink(missing_ok=True)


def test_add_codeflash_capture_no_parent():
    # Test input code
    original_code = """
class MyClass:

    def target_function(self):
        return self.x + 1
"""

    expected = """
class MyClass:

    def target_function(self):
        return self.x + 1
"""

    # Create and modify test file
    test_file = Path("test_file.py")
    test_file.write_text(original_code)

    function = FunctionToOptimize(function_name="target_function", file_path=Path("test_file.py"), parents=[])

    try:
        # Run the instrumentation
        instrument_code(function)

        # Check the result
        modified_code = test_file.read_text()
        assert modified_code.strip() == expected.strip()

    finally:
        # Cleanup
        test_file.unlink(missing_ok=True)


def test_add_codeflash_capture_no_init():
    # Test input code
    original_code = """
class MyClass(ParentClass):

    def target_function(self):
        return self.x + 1
"""

    expected = f"""
from codeflash.verification.codeflash_capture import codeflash_capture

class MyClass(ParentClass):

    @codeflash_capture(function_name='target_function', tmp_dir_path='{get_run_tmp_file(Path("test_return_values"))!s}')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def target_function(self):
        return self.x + 1
"""

    # Create and modify test file
    test_file = Path("test_file.py")
    test_file.write_text(original_code)

    function = FunctionToOptimize(
        function_name="target_function",
        file_path=Path("test_file.py"),
        parents=[FunctionParent(type="ClassDef", name="MyClass")],
    )

    try:
        # Run the instrumentation
        instrument_code(function)

        # Check the result
        modified_code = test_file.read_text()
        assert modified_code.strip() == expected.strip()

    finally:
        # Cleanup
        test_file.unlink(missing_ok=True)
