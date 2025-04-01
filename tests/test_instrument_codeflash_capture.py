from pathlib import Path

from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import FunctionParent
from codeflash.verification.instrument_codeflash_capture import instrument_codeflash_capture


def test_add_codeflash_capture():
    original_code = """
class MyClass:
    def __init__(self):
        self.x = 1

    def target_function(self):
        return self.x + 1
"""
    test_path = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_file.py").resolve()
    expected = f"""
from codeflash.verification.codeflash_capture import codeflash_capture


class MyClass:

    @codeflash_capture(function_name='MyClass.__init__', tmp_dir_path='{get_run_tmp_file("test_return_values").as_posix()}', tests_root='{test_path.parent.as_posix()}', is_fto=True)
    def __init__(self):
        self.x = 1

    def target_function(self):
        return self.x + 1
"""
    test_path.write_text(original_code)

    function = FunctionToOptimize(
        function_name="target_function", file_path=test_path, parents=[FunctionParent(type="ClassDef", name="MyClass")]
    )

    try:
        instrument_codeflash_capture(function, {}, test_path.parent)
        modified_code = test_path.read_text()
        assert modified_code.strip() == expected.strip()

    finally:
        test_path.unlink(missing_ok=True)


def test_add_codeflash_capture_no_parent():
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
    test_path = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_file.py").resolve()
    test_path.write_text(original_code)

    function = FunctionToOptimize(function_name="target_function", file_path=test_path, parents=[])

    try:
        instrument_codeflash_capture(function, {}, test_path.parent)
        modified_code = test_path.read_text()
        assert modified_code.strip() == expected.strip()
    finally:
        test_path.unlink(missing_ok=True)


def test_add_codeflash_capture_no_init():
    # Test input code
    original_code = """
class MyClass(ParentClass):

    def target_function(self):
        return self.x + 1
"""
    test_path = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_file.py").resolve()
    expected = f"""
from codeflash.verification.codeflash_capture import codeflash_capture


class MyClass(ParentClass):

    @codeflash_capture(function_name='MyClass.__init__', tmp_dir_path='{get_run_tmp_file(Path("test_return_values")).as_posix()}', tests_root='{test_path.parent.as_posix()}', is_fto=True)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def target_function(self):
        return self.x + 1
"""
    test_path.write_text(original_code)

    function = FunctionToOptimize(
        function_name="target_function", file_path=test_path, parents=[FunctionParent(type="ClassDef", name="MyClass")]
    )

    try:
        instrument_codeflash_capture(function, {}, test_path.parent)
        modified_code = test_path.read_text()
        assert modified_code.strip() == expected.strip()

    finally:
        test_path.unlink(missing_ok=True)


def test_add_codeflash_capture_with_helpers():
    # Test input code
    original_code = """
class MyClass:
    def __init__(self):
        self.x = 1

    def target_function(self):
        return helper() + 1

    def helper(self):
        return self.x
"""
    test_path = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_file.py").resolve()
    expected = f"""
from codeflash.verification.codeflash_capture import codeflash_capture


class MyClass:

    @codeflash_capture(function_name='MyClass.__init__', tmp_dir_path='{get_run_tmp_file(Path("test_return_values")).as_posix()}', tests_root='{test_path.parent.as_posix()}', is_fto=True)
    def __init__(self):
        self.x = 1

    def target_function(self):
        return helper() + 1

    def helper(self):
        return self.x
"""

    test_path.write_text(original_code)

    function = FunctionToOptimize(
        function_name="target_function", file_path=test_path, parents=[FunctionParent(type="ClassDef", name="MyClass")]
    )

    try:
        instrument_codeflash_capture(
            function, {test_path: {"MyClass"}}, test_path.parent
        )  # MyClass was removed from the file_path_to_helper_class as it shares class with FTO
        modified_code = test_path.read_text()
        assert modified_code.strip() == expected.strip()

    finally:
        test_path.unlink(missing_ok=True)


def test_add_codeflash_capture_with_helpers_2():
    # Test input code
    original_code = """
from test_helper_file import HelperClass

class MyClass:
    def __init__(self):
        self.x = 1

    def target_function(self):
        return HelperClass().helper() + 1
"""
    original_helper = """
class HelperClass:
    def __init__(self):
        self.y = 1
    def helper(self):
        return 1
"""
    test_path = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_file.py").resolve()
    expected = f"""
from test_helper_file import HelperClass

from codeflash.verification.codeflash_capture import codeflash_capture


class MyClass:

    @codeflash_capture(function_name='MyClass.__init__', tmp_dir_path='{get_run_tmp_file(Path("test_return_values")).as_posix()}', tests_root='{test_path.parent.as_posix()}', is_fto=True)
    def __init__(self):
        self.x = 1

    def target_function(self):
        return HelperClass().helper() + 1
"""
    expected_helper = f"""
from codeflash.verification.codeflash_capture import codeflash_capture


class HelperClass:

    @codeflash_capture(function_name='HelperClass.__init__', tmp_dir_path='{get_run_tmp_file(Path("test_return_values")).as_posix()}', tests_root='{test_path.parent.as_posix()}', is_fto=False)
    def __init__(self):
        self.y = 1

    def helper(self):
        return 1
"""

    test_path.write_text(original_code)
    helper_path = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_helper_file.py").resolve()
    helper_path.write_text(original_helper)

    function = FunctionToOptimize(
        function_name="target_function", file_path=test_path, parents=[FunctionParent(type="ClassDef", name="MyClass")]
    )

    try:
        instrument_codeflash_capture(function, {helper_path: {"HelperClass"}}, test_path.parent)
        modified_code = test_path.read_text()
        assert modified_code.strip() == expected.strip()
        assert helper_path.read_text().strip() == expected_helper.strip()
    finally:
        test_path.unlink(missing_ok=True)
        helper_path.unlink(missing_ok=True)


def test_add_codeflash_capture_with_multiple_helpers():
    # Test input code with imports from two helper files
    original_code = """
from helper_file_1 import HelperClass1
from helper_file_2 import HelperClass2, AnotherHelperClass

class MyClass:
    def __init__(self):
        self.x = 1

    def target_function(self):
        helper1 = HelperClass1().helper1()
        helper2 = HelperClass2().helper2()
        another = AnotherHelperClass().another_helper()
        return helper1 + helper2 + another
"""

    # First helper file content
    original_helper1 = """
class HelperClass1:
    def __init__(self):
        self.y = 1
    def helper1(self):
        return 1
"""

    # Second helper file content
    original_helper2 = """
class HelperClass2:
    def __init__(self):
        self.z = 2
    def helper2(self):
        return 2

class AnotherHelperClass:
    def another_helper(self):
        return 3
"""
    test_path = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_file.py").resolve()
    expected = f"""
from helper_file_1 import HelperClass1
from helper_file_2 import AnotherHelperClass, HelperClass2

from codeflash.verification.codeflash_capture import codeflash_capture


class MyClass:

    @codeflash_capture(function_name='MyClass.__init__', tmp_dir_path='{get_run_tmp_file(Path("test_return_values")).as_posix()}', tests_root='{test_path.parent.as_posix()}', is_fto=True)
    def __init__(self):
        self.x = 1

    def target_function(self):
        helper1 = HelperClass1().helper1()
        helper2 = HelperClass2().helper2()
        another = AnotherHelperClass().another_helper()
        return helper1 + helper2 + another
"""

    # Expected output for first helper file
    expected_helper1 = f"""
from codeflash.verification.codeflash_capture import codeflash_capture


class HelperClass1:

    @codeflash_capture(function_name='HelperClass1.__init__', tmp_dir_path='{get_run_tmp_file(Path("test_return_values")).as_posix()}', tests_root='{test_path.parent.as_posix()}', is_fto=False)
    def __init__(self):
        self.y = 1

    def helper1(self):
        return 1
"""

    # Expected output for second helper file
    expected_helper2 = f"""
from codeflash.verification.codeflash_capture import codeflash_capture


class HelperClass2:

    @codeflash_capture(function_name='HelperClass2.__init__', tmp_dir_path='{get_run_tmp_file(Path("test_return_values")).as_posix()}', tests_root='{test_path.parent.as_posix()}', is_fto=False)
    def __init__(self):
        self.z = 2

    def helper2(self):
        return 2

class AnotherHelperClass:

    @codeflash_capture(function_name='AnotherHelperClass.__init__', tmp_dir_path='{get_run_tmp_file(Path("test_return_values")).as_posix()}', tests_root='{test_path.parent.as_posix()}', is_fto=False)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def another_helper(self):
        return 3
"""

    # Set up test files
    helper1_path = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/helper_file_1.py").resolve()
    helper2_path = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/helper_file_2.py").resolve()

    # Write original content to files
    test_path.write_text(original_code)
    helper1_path.write_text(original_helper1)
    helper2_path.write_text(original_helper2)

    # Create FunctionToOptimize instance
    function = FunctionToOptimize(
        function_name="target_function", file_path=test_path, parents=[FunctionParent(type="ClassDef", name="MyClass")]
    )

    try:
        # Instrument code with multiple helper files
        helper_classes = {helper1_path: {"HelperClass1"}, helper2_path: {"HelperClass2", "AnotherHelperClass"}}
        instrument_codeflash_capture(function, helper_classes, test_path.parent)

        # Verify the modifications
        modified_code = test_path.read_text()
        modified_helper1 = helper1_path.read_text()
        modified_helper2 = helper2_path.read_text()

        assert modified_code.strip() == expected.strip()
        assert modified_helper1.strip() == expected_helper1.strip()
        assert modified_helper2.strip() == expected_helper2.strip()

    finally:
        # Clean up test files
        test_path.unlink(missing_ok=True)
        helper1_path.unlink(missing_ok=True)
        helper2_path.unlink(missing_ok=True)
