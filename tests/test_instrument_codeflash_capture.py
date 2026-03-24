from pathlib import Path

from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash_core.models import FunctionToOptimize
from codeflash.languages.python.instrument_codeflash_capture import instrument_codeflash_capture
from codeflash.models.models import FunctionParent


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

    @codeflash_capture(function_name='MyClass.__init__', tmp_dir_path='{get_run_tmp_file(Path("test_return_values")).as_posix()}', tests_root='{test_path.parent.as_posix()}', is_fto=True)
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


def test_dataclass_no_init_skipped():
    """Dataclasses have auto-generated __init__ not visible in AST. Instrumentation should skip them."""
    original_code = """
from dataclasses import dataclass

@dataclass
class MyDataClass:
    x: int
    y: str

    def target_function(self):
        return self.x + len(self.y)
"""
    test_path = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_file.py").resolve()
    test_path.write_text(original_code)

    function = FunctionToOptimize(
        function_name="target_function", file_path=test_path, parents=[FunctionParent(type="ClassDef", name="MyDataClass")]
    )

    try:
        instrument_codeflash_capture(function, {}, test_path.parent)
        modified_code = test_path.read_text()
        # Dataclass should NOT get a synthetic __init__ injected
        assert "super().__init__" not in modified_code
        assert "codeflash_capture" not in modified_code
    finally:
        test_path.unlink(missing_ok=True)


def test_dataclass_with_call_syntax_skipped():
    """@dataclass(frozen=True) should also be skipped."""
    original_code = """
from dataclasses import dataclass

@dataclass(frozen=True)
class FrozenData:
    value: int

    def compute(self):
        return self.value * 2
"""
    test_path = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_file.py").resolve()
    test_path.write_text(original_code)

    function = FunctionToOptimize(
        function_name="compute", file_path=test_path, parents=[FunctionParent(type="ClassDef", name="FrozenData")]
    )

    try:
        instrument_codeflash_capture(function, {}, test_path.parent)
        modified_code = test_path.read_text()
        assert "super().__init__" not in modified_code
        assert "codeflash_capture" not in modified_code
    finally:
        test_path.unlink(missing_ok=True)


def test_namedtuple_no_init_skipped():
    """NamedTuples have synthesized __init__ that cannot be overwritten. Instrumentation should skip them."""
    original_code = """
from typing import NamedTuple

class MyTuple(NamedTuple):
    x: int
    y: str

    def display(self):
        return f"{self.x}: {self.y}"
"""
    test_path = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_file.py").resolve()
    test_path.write_text(original_code)

    function = FunctionToOptimize(
        function_name="display", file_path=test_path, parents=[FunctionParent(type="ClassDef", name="MyTuple")]
    )

    try:
        instrument_codeflash_capture(function, {}, test_path.parent)
        modified_code = test_path.read_text()
        assert "super().__init__" not in modified_code
        assert "codeflash_capture" not in modified_code
    finally:
        test_path.unlink(missing_ok=True)


def test_module_qualified_dataclass_with_call_syntax_skipped():
    """@dataclasses.dataclass(frozen=True) — module-qualified call-style decorator — should be skipped."""
    original_code = """
import dataclasses

@dataclasses.dataclass(frozen=True)
class FrozenPoint:
    x: int
    y: int

    def magnitude(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5
"""
    test_path = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_file.py").resolve()
    test_path.write_text(original_code)

    function = FunctionToOptimize(
        function_name="magnitude", file_path=test_path, parents=[FunctionParent(type="ClassDef", name="FrozenPoint")]
    )

    try:
        instrument_codeflash_capture(function, {}, test_path.parent)
        modified_code = test_path.read_text()
        assert "super().__init__" not in modified_code
        assert "codeflash_capture" not in modified_code
    finally:
        test_path.unlink(missing_ok=True)


def test_module_qualified_namedtuple_skipped():
    """typing.NamedTuple — module-qualified base class — should be skipped."""
    original_code = """
import typing

class MyTuple(typing.NamedTuple):
    x: int
    y: str

    def display(self):
        return f"{self.x}: {self.y}"
"""
    test_path = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_file.py").resolve()
    test_path.write_text(original_code)

    function = FunctionToOptimize(
        function_name="display", file_path=test_path, parents=[FunctionParent(type="ClassDef", name="MyTuple")]
    )

    try:
        instrument_codeflash_capture(function, {}, test_path.parent)
        modified_code = test_path.read_text()
        assert "super().__init__" not in modified_code
        assert "codeflash_capture" not in modified_code
    finally:
        test_path.unlink(missing_ok=True)


def test_attrs_define_patched_via_module_wrapper():
    """@attrs.define classes must NOT get a synthetic body __init__; instead a module-level
    monkey-patch block is emitted after the class to avoid the __class__ cell TypeError
    that arises when attrs.define(slots=True) replaces the original class object.
    """
    original_code = """
import attrs
from attrs.validators import instance_of

@attrs.define
class MyAttrsClass:
    x: int = attrs.field(validator=[instance_of(int)])
    y: str = attrs.field(default="hello")

    def compute(self):
        return self.x
"""
    test_path = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_file.py").resolve()
    expected = f"""import attrs
from attrs.validators import instance_of

from codeflash.verification.codeflash_capture import codeflash_capture


@attrs.define
class MyAttrsClass:
    x: int = attrs.field(validator=[instance_of(int)])
    y: str = attrs.field(default='hello')

    def compute(self):
        return self.x
_codeflash_orig_MyAttrsClass_init = MyAttrsClass.__init__

def _codeflash_patched_MyAttrsClass_init(self, *args, **kwargs):
    return _codeflash_orig_MyAttrsClass_init(self, *args, **kwargs)
MyAttrsClass.__init__ = codeflash_capture(function_name='MyAttrsClass.__init__', tmp_dir_path='{get_run_tmp_file(Path("test_return_values")).as_posix()}', tests_root='{test_path.parent.as_posix()}', is_fto=True)(_codeflash_patched_MyAttrsClass_init)
"""
    test_path.write_text(original_code)

    function = FunctionToOptimize(
        function_name="compute", file_path=test_path, parents=[FunctionParent(type="ClassDef", name="MyAttrsClass")]
    )

    try:
        instrument_codeflash_capture(function, {}, test_path.parent)
        modified_code = test_path.read_text()
        assert modified_code.strip() == expected.strip()
    finally:
        test_path.unlink(missing_ok=True)


def test_attrs_define_frozen_patched_via_module_wrapper():
    """@attrs.define(frozen=True) should also be monkey-patched at module level."""
    original_code = """
import attrs

@attrs.define(frozen=True)
class FrozenPoint:
    x: float = attrs.field()
    y: float = attrs.field()

    def distance(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5
"""
    test_path = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_file.py").resolve()
    expected = f"""import attrs

from codeflash.verification.codeflash_capture import codeflash_capture


@attrs.define(frozen=True)
class FrozenPoint:
    x: float = attrs.field()
    y: float = attrs.field()

    def distance(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5
_codeflash_orig_FrozenPoint_init = FrozenPoint.__init__

def _codeflash_patched_FrozenPoint_init(self, *args, **kwargs):
    return _codeflash_orig_FrozenPoint_init(self, *args, **kwargs)
FrozenPoint.__init__ = codeflash_capture(function_name='FrozenPoint.__init__', tmp_dir_path='{get_run_tmp_file(Path("test_return_values")).as_posix()}', tests_root='{test_path.parent.as_posix()}', is_fto=True)(_codeflash_patched_FrozenPoint_init)
"""
    test_path.write_text(original_code)

    function = FunctionToOptimize(
        function_name="distance", file_path=test_path, parents=[FunctionParent(type="ClassDef", name="FrozenPoint")]
    )

    try:
        instrument_codeflash_capture(function, {}, test_path.parent)
        modified_code = test_path.read_text()
        assert modified_code.strip() == expected.strip()
    finally:
        test_path.unlink(missing_ok=True)


def test_attr_s_patched_via_module_wrapper():
    """@attr.s classes should also be monkey-patched at module level."""
    original_code = """
import attr

@attr.s
class MyAttrClass:
    x: int = attr.ib()

    def display(self):
        return self.x
"""
    test_path = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_file.py").resolve()
    expected = f"""import attr

from codeflash.verification.codeflash_capture import codeflash_capture


@attr.s
class MyAttrClass:
    x: int = attr.ib()

    def display(self):
        return self.x
_codeflash_orig_MyAttrClass_init = MyAttrClass.__init__

def _codeflash_patched_MyAttrClass_init(self, *args, **kwargs):
    return _codeflash_orig_MyAttrClass_init(self, *args, **kwargs)
MyAttrClass.__init__ = codeflash_capture(function_name='MyAttrClass.__init__', tmp_dir_path='{get_run_tmp_file(Path("test_return_values")).as_posix()}', tests_root='{test_path.parent.as_posix()}', is_fto=True)(_codeflash_patched_MyAttrClass_init)
"""
    test_path.write_text(original_code)

    function = FunctionToOptimize(
        function_name="display", file_path=test_path, parents=[FunctionParent(type="ClassDef", name="MyAttrClass")]
    )

    try:
        instrument_codeflash_capture(function, {}, test_path.parent)
        modified_code = test_path.read_text()
        assert modified_code.strip() == expected.strip()
    finally:
        test_path.unlink(missing_ok=True)


def test_attrs_define_init_false_skipped():
    """@attrs.define(init=False) should NOT be monkey-patched because attrs won't generate an __init__."""
    original_code = """
import attrs

@attrs.define(init=False)
class ManualInit:
    x: int = attrs.field()

    def compute(self):
        return self.x
"""
    expected = """import attrs


@attrs.define(init=False)
class ManualInit:
    x: int = attrs.field()

    def compute(self):
        return self.x
"""
    test_path = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_file.py").resolve()
    test_path.write_text(original_code)

    function = FunctionToOptimize(
        function_name="compute", file_path=test_path, parents=[FunctionParent(type="ClassDef", name="ManualInit")]
    )

    try:
        instrument_codeflash_capture(function, {}, test_path.parent)
        modified_code = test_path.read_text()
        assert modified_code.strip() == expected.strip()
    finally:
        test_path.unlink(missing_ok=True)


def test_dataclass_with_explicit_init_still_instrumented():
    """A dataclass that defines its own __init__ should still be instrumented normally."""
    original_code = """
from dataclasses import dataclass

@dataclass
class CustomInit:
    x: int

    def __init__(self, x: int):
        self.x = x * 2

    def target(self):
        return self.x
"""
    test_path = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_file.py").resolve()
    test_path.write_text(original_code)

    function = FunctionToOptimize(
        function_name="target", file_path=test_path, parents=[FunctionParent(type="ClassDef", name="CustomInit")]
    )

    try:
        instrument_codeflash_capture(function, {}, test_path.parent)
        modified_code = test_path.read_text()
        # Should be instrumented because it has an explicit __init__
        assert "codeflash_capture" in modified_code
        # Should NOT have super().__init__ injected (it has its own __init__)
        assert "super().__init__" not in modified_code
    finally:
        test_path.unlink(missing_ok=True)
