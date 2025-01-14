from textwrap import dedent

import libcst as cst
from codeflash.code_utils.code_replacer import merge_init_functions, replace_functions_in_file
from codeflash.models.models import FunctionParent


def test_basic_merge() -> None:
    original = """
    class MyClass:
        def __init__(self, a, b):
            self.a = a
            self.b = b
    """
    new = """
    class MyClass:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    """
    result = merge_init_functions(
        cst.parse_module(dedent(original)).body[0].body.body[0], cst.parse_module(dedent(new)).body[0].body.body[0]
    )

    expected = """
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.x = x
        self.y = y
    """
    assert cst.Module([result]).code.strip() == dedent(expected).strip()


def test_prevent_duplication() -> None:
    original = """
    class MyClass:
        def __init__(self):
            self.x = 1
            print("init")
            self.setup()
    """
    new = """
    class MyClass:
        def __init__(self):
            print("init")
            self.setup()
            self.y = 2
    """
    result = merge_init_functions(
        cst.parse_module(dedent(original)).body[0].body.body[0], cst.parse_module(dedent(new)).body[0].body.body[0]
    )

    expected = """
    def __init__(self):
        self.x = 1
        print("init")
        self.setup()
        self.y = 2
    """
    assert cst.Module([result]).code.strip() == dedent(expected).strip()


def test_prevent_overwrite() -> None:
    original = """
    class MyClass:
        def __init__(self):
            self.x = 1
            self.y = 2 
    """
    new = """
    class MyClass:
        def __init__(self):
            self.x = 2
    """
    result = merge_init_functions(
        cst.parse_module(dedent(original)).body[0].body.body[0], cst.parse_module(dedent(new)).body[0].body.body[0]
    )

    expected = """
    def __init__(self):
        self.x = 1
        self.y = 2
    """
    assert cst.Module([result]).code.strip() == dedent(expected).strip()


def test_complex_control_flow() -> None:
    original = """
    class MyClass:
        def __init__(self):
            with self.lock:
                self.setup()
            if self.debug:
                self.enable_logging()
    """
    new = """
    class MyClass:
        def __init__(self):
            try:
                self.connect()
            except ConnectionError:
                self.fallback()
    """
    result = merge_init_functions(
        cst.parse_module(dedent(original)).body[0].body.body[0], cst.parse_module(dedent(new)).body[0].body.body[0]
    )

    expected = """
    def __init__(self):
        with self.lock:
            self.setup()
        if self.debug:
            self.enable_logging()
        try:
            self.connect()
        except ConnectionError:
            self.fallback()
    """
    assert cst.Module([result]).code.strip() == dedent(expected).strip()


def test_docstrings_and_comments() -> None:
    original = """
    class MyClass:
        def __init__(self):
            # Setup configuration
            self.config = {}  # Empty config
    """
    new = """
    class MyClass:
        def __init__(self):
            \"\"\"New docstring.\"\"\"
            # Initialize database
            self.db = None  # Database connection
    """
    result = merge_init_functions(
        cst.parse_module(dedent(original)).body[0].body.body[0], cst.parse_module(dedent(new)).body[0].body.body[0]
    )
    expected = """
    def __init__(self):
        # Setup configuration
        self.config = {}  # Empty config
        # Initialize database
        self.db = None  # Database connection
    """
    assert cst.Module([result]).code.strip() == dedent(expected).strip()


def test_type_annotations() -> None:
    original = """
    class MyClass:
        def __init__(self) -> None:
            self.x: int = 1
            self.y: str = "hello"
    """
    new = """
    class MyClass:
        def __init__(self):
            self.y: str = "new hello"
            self.z: float = 2.0
    """
    result = merge_init_functions(
        cst.parse_module(dedent(original)).body[0].body.body[0], cst.parse_module(dedent(new)).body[0].body.body[0]
    )

    expected = """
    def __init__(self) -> None:
        self.x: int = 1
        self.y: str = "hello"
        self.z: float = 2.0
    """
    assert cst.Module([result]).code.strip() == dedent(expected).strip()


# Tests for code replacement with init
def test_merge_init_methods() -> None:
    optim_code = """class MyClass:
    def __init__(self):
        self.y = 2
        self.z = 3
"""

    original_code = """class MyClass:
    def __init__(self):
        self.y = 1
        self.setup()
"""

    expected = """class MyClass:
    def __init__(self):
        self.y = 1
        self.setup()
        self.z = 3
"""

    result = replace_functions_in_file(
        source_code=original_code,
        original_function_names=[],
        optimized_code=optim_code,
        preexisting_objects=[("__init__", [FunctionParent(name="MyClass", type="ClassDef")])],
    )
    assert result == expected


def test_init_is_function_to_optimize() -> None:
    optim_code = """class MyClass:
    def __init__(self):
        self.y = 2
        self.z = 3
"""

    original_code = """class MyClass:
    def __init__(self):
        self.y = 1
        self.setup()
"""

    expected = """class MyClass:
    def __init__(self):
        self.y = 2
        self.z = 3
"""
    # In this scenario, we leave the mutation check to the usual FTO behaviour check.
    result = replace_functions_in_file(
        source_code=original_code,
        original_function_names=["MyClass.__init__"],
        optimized_code=optim_code,
        preexisting_objects=[("__init__", [FunctionParent(name="MyClass", type="ClassDef")])],
    )
    assert result == expected
