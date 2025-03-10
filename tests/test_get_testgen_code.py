from textwrap import dedent

import pytest

from codeflash.models.models import CodeContextType
from codeflash.context.code_context_extractor import parse_code_and_prune_cst

def test_simple_function() -> None:
    code = """
    def target_function():
        x = 1
        y = 2
        return x + y
    """
    result = parse_code_and_prune_cst(dedent(code), CodeContextType.TESTGEN, {"target_function"}, set())

    expected = """
    def target_function():
        x = 1
        y = 2
        return x + y
    """
    assert dedent(expected).strip() == result.strip()

def test_basic_class() -> None:
    code = """
    class TestClass:
        class_var = "value"

        def target_method(self):
            print("This should be included")

        def other_method(self):
            print("This too")
    """

    expected = """
    class TestClass:
        class_var = "value"

        def target_method(self):
            print("This should be included")
    """

    output = parse_code_and_prune_cst(dedent(code), CodeContextType.TESTGEN, {"TestClass.target_method"}, set())
    assert dedent(expected).strip() == output.strip()

def test_dunder_methods() -> None:
    code = """
    class TestClass:
        def __init__(self):
            self.x = 42

        def __str__(self):
            return f"Value: {self.x}"

        def target_method(self):
            print("include me")
    """

    expected = """
    class TestClass:
        def __init__(self):
            self.x = 42

        def __str__(self):
            return f"Value: {self.x}"

        def target_method(self):
            print("include me")
    """

    output = parse_code_and_prune_cst(dedent(code), CodeContextType.TESTGEN, {"TestClass.target_method"}, set())
    assert dedent(expected).strip() == output.strip()


def test_dunder_methods_remove_docstring() -> None:
    code = """
    class TestClass:
        def __init__(self):
            \"\"\"Constructor for TestClass.\"\"\"
            self.x = 42

        def __str__(self):
            \"\"\"String representation of TestClass.\"\"\"
            return f"Value: {self.x}"

        def target_method(self):
            \"\"\"Target method docstring.\"\"\"
            print("include me")
    """

    expected = """
    class TestClass:
        def __init__(self):
            self.x = 42

        def __str__(self):
            return f"Value: {self.x}"

        def target_method(self):
            print("include me")
    """

    output = parse_code_and_prune_cst(dedent(code), CodeContextType.TESTGEN, {"TestClass.target_method"}, set(), remove_docstrings=True)
    assert dedent(expected).strip() == output.strip()


def test_class_remove_docstring() -> None:
    code = """
    class TestClass:
        \"\"\"Class docstring.\"\"\"
        def __init__(self):
            self.x = 42

        def __str__(self):
            return f"Value: {self.x}"

        def target_method(self):
            print("include me")
    """

    expected = """
    class TestClass:
        def __init__(self):
            self.x = 42

        def __str__(self):
            return f"Value: {self.x}"

        def target_method(self):
            print("include me")
    """

    output = parse_code_and_prune_cst(dedent(code), CodeContextType.TESTGEN, {"TestClass.target_method"}, set(), remove_docstrings=True)
    assert dedent(expected).strip() == output.strip()


def test_target_in_nested_class() -> None:
    """Test that attempting to find a target in a nested class raises an error."""
    code = """
    class Outer:
        outer_var = 1

        class Inner:
            inner_var = 2

            def target_method(self):
                print("include this")
    """

    with pytest.raises(ValueError, match="No target functions found in the provided code"):
        parse_code_and_prune_cst(dedent(code), CodeContextType.TESTGEN, {"Outer.Inner.target_method"}, set())

def test_method_signatures() -> None:
    code = """
    class TestClass:
        @property
        def target_method(self) -> str:
            \"\"\"Property docstring.\"\"\"
            return "value"

        @classmethod
        def class_method(cls, param: int = 42) -> None:
            print("class method")
    """

    expected = """
    class TestClass:
        @property
        def target_method(self) -> str:
            \"\"\"Property docstring.\"\"\"
            return "value"
    """

    output = parse_code_and_prune_cst(dedent(code), CodeContextType.TESTGEN, {"TestClass.target_method"}, set())
    assert dedent(expected).strip() == output.strip()
def test_multiple_top_level_targets() -> None:
    code = """
    class TestClass:
        def target1(self):
            print("include 1")

        def target2(self):
            print("include 2")

        def __init__(self):
            self.x = 42

        def other_method(self):
            print("include other")
    """

    expected = """
    class TestClass:
        def target1(self):
            print("include 1")

        def target2(self):
            print("include 2")

        def __init__(self):
            self.x = 42
    """

    output = parse_code_and_prune_cst(dedent(code), CodeContextType.TESTGEN, {"TestClass.target1", "TestClass.target2"}, set())
    assert dedent(expected).strip() == output.strip()


def test_class_annotations() -> None:
    code = """
    class TestClass:
        var1: int = 42
        var2: str

        def target_method(self) -> None:
            self.var2 = "test"
    """

    expected = """
    class TestClass:
        var1: int = 42
        var2: str

        def target_method(self) -> None:
            self.var2 = "test"
    """

    output = parse_code_and_prune_cst(dedent(code), CodeContextType.TESTGEN, {"TestClass.target_method"}, set())
    assert dedent(expected).strip() == output.strip()

def test_class_annotations_if() -> None:
    code = """
    if True:
        class TestClass:
            var1: int = 42
            var2: str

            def target_method(self) -> None:
                self.var2 = "test"
    """

    expected = """
    if True:
        class TestClass:
            var1: int = 42
            var2: str

            def target_method(self) -> None:
                self.var2 = "test"
    """

    output = parse_code_and_prune_cst(dedent(code), CodeContextType.TESTGEN, {"TestClass.target_method"}, set())
    assert dedent(expected).strip() == output.strip()


def test_conditional_class_definitions() -> None:
    code = """
    if PLATFORM == "linux":
        class PlatformClass:
            platform = "linux"
            def target_method(self):
                print("linux")
    elif PLATFORM == "windows":
        class PlatformClass:
            platform = "windows"
            def target_method(self):
                print("windows")
    else:
        class PlatformClass:
            platform = "other"
            def target_method(self):
                print("other")
    """

    expected = """
    if PLATFORM == "linux":
        class PlatformClass:
            platform = "linux"
            def target_method(self):
                print("linux")
    elif PLATFORM == "windows":
        class PlatformClass:
            platform = "windows"
            def target_method(self):
                print("windows")
    else:
        class PlatformClass:
            platform = "other"
            def target_method(self):
                print("other")
    """

    output = parse_code_and_prune_cst(dedent(code), CodeContextType.TESTGEN, {"PlatformClass.target_method"}, set())
    assert dedent(expected).strip() == output.strip()


def test_try_except_structure() -> None:
    code = """
    try:
        class TargetClass:
            attr = "value"
            def target_method(self):
                return 42
    except ValueError:
        class ErrorClass:
            def handle_error(self):
                print("error")
    """

    expected = """
    try:
        class TargetClass:
            attr = "value"
            def target_method(self):
                return 42
    except ValueError:
        class ErrorClass:
            def handle_error(self):
                print("error")
    """

    output = parse_code_and_prune_cst(dedent(code), CodeContextType.TESTGEN, {"TargetClass.target_method"}, set())
    assert dedent(expected).strip() == output.strip()


def test_module_var() -> None:
    code = """
    def target_function(self) -> None:
        self.var2 = "test"

    x = 5

    def some_function():
        print("wow")
    """

    expected = """
    def target_function(self) -> None:
        self.var2 = "test"

    x = 5
    """

    output = parse_code_and_prune_cst(dedent(code), CodeContextType.TESTGEN, {"target_function"}, set())
    assert dedent(expected).strip() == output.strip()

def test_module_var_if() -> None:
    code = """
    def target_function(self) -> None:
        var2 = "test"

    if y:
        x = 5
    else:
        z = 10
        def some_function():
            print("wow")

    def some_function():
        print("wow")
    """

    expected = """
    def target_function(self) -> None:
        var2 = "test"

    if y:
        x = 5
    else:
        z = 10
    """

    output = parse_code_and_prune_cst(dedent(code), CodeContextType.TESTGEN, {"target_function"}, set())
    assert dedent(expected).strip() == output.strip()

def test_multiple_classes() -> None:
    code = """
    class ClassA:
        def process(self):
            return "A"

    class ClassB:
        def process(self):
            return "B"

    class ClassC:
        def process(self):
            return "C"
    """

    expected = """
    class ClassA:
        def process(self):
            return "A"

    class ClassC:
        def process(self):
            return "C"
    """

    output = parse_code_and_prune_cst(dedent(code), CodeContextType.TESTGEN, {"ClassA.process", "ClassC.process"}, set())
    assert dedent(expected).strip() == output.strip()


def test_with_statement_and_loops() -> None:
    code = """
    with context_manager() as ctx:
        while attempt_count < max_attempts:
            try:
                for item in items:
                    if item.ready:
                        class TestClass:
                            context = "ready"
                            def target_method(self):
                                print("ready")
                    else:
                        class TestClass:
                            context = "not_ready"
                            def target_method(self):
                                print("not ready")
            except ConnectionError:
                class TestClass:
                    context = "connection_error"
                    def target_method(self):
                        print("connection error")
                continue
            finally:
                class TestClass:
                    context = "cleanup"
                    def target_method(self):
                        print("cleanup")
    """

    expected = """
    with context_manager() as ctx:
        while attempt_count < max_attempts:
            try:
                for item in items:
                    if item.ready:
                        class TestClass:
                            context = "ready"
                            def target_method(self):
                                print("ready")
                    else:
                        class TestClass:
                            context = "not_ready"
                            def target_method(self):
                                print("not ready")
            except ConnectionError:
                class TestClass:
                    context = "connection_error"
                    def target_method(self):
                        print("connection error")
                continue
            finally:
                class TestClass:
                    context = "cleanup"
                    def target_method(self):
                        print("cleanup")
    """

    output = parse_code_and_prune_cst(dedent(code), CodeContextType.TESTGEN, {"TestClass.target_method"}, set())
    assert dedent(expected).strip() == output.strip()


def test_async_with_try_except() -> None:
    code = """
    async with async_context() as ctx:
        try:
            async for item in items:
                if await item.is_valid():
                    class TestClass:
                        status = "valid"
                        async def target_method(self):
                            await self.process()
                elif await item.can_retry():
                    continue
                else:
                    break
        except AsyncIOError:
            class TestClass:
                status = "io_error"
                async def target_method(self):
                    await self.handle_error()
        except CancelledError:
            class TestClass:
                status = "cancelled"
                async def target_method(self):
                    await self.cleanup()
    """

    expected = """
    async with async_context() as ctx:
        try:
            async for item in items:
                if await item.is_valid():
                    class TestClass:
                        status = "valid"
                        async def target_method(self):
                            await self.process()
                elif await item.can_retry():
                    continue
                else:
                    break
        except AsyncIOError:
            class TestClass:
                status = "io_error"
                async def target_method(self):
                    await self.handle_error()
        except CancelledError:
            class TestClass:
                status = "cancelled"
                async def target_method(self):
                    await self.cleanup()
    """

    output = parse_code_and_prune_cst(dedent(code), CodeContextType.TESTGEN, {"TestClass.target_method"}, set())
    assert dedent(expected).strip() == output.strip()

def test_simplified_complete_implementation() -> None:
    code = """
    class DataProcessor:
        \"\"\"A simple data processing class.\"\"\"

        def __init__(self, data: Dict[str, Any]) -> None:
            self.data = data
            self._processed = False
            self.result = None

        def __repr__(self) -> str:
            return f"DataProcessor(processed={self._processed})"

        def target_method(self, key: str) -> Optional[Any]:
            \"\"\"Process and retrieve a specific key from the data.\"\"\"
            if not self._processed:
                self._process_data()
            return self.result.get(key) if self.result else None

        def _process_data(self) -> None:
            \"\"\"Internal method to process the data.\"\"\"
            processed = {}
            for key, value in self.data.items():
                if isinstance(value, (int, float)):
                    processed[key] = value * 2
                elif isinstance(value, str):
                    processed[key] = value.upper()
                else:
                    processed[key] = value
            self.result = processed
            self._processed = True

        def to_json(self) -> str:
            \"\"\"Convert the processed data to JSON string.\"\"\"
            if not self._processed:
                self._process_data()
            return json.dumps(self.result)

    try:
        sample_data = {"number": 42, "text": "hello"}
        processor = DataProcessor(sample_data)

        class ResultHandler:
            def __init__(self, processor: DataProcessor):
                self.processor = processor
                self.cache = {}

            def __str__(self) -> str:
                return f"ResultHandler(cache_size={len(self.cache)})"

            def target_method(self, key: str) -> Optional[Any]:
                \"\"\"Retrieve and cache results for a key.\"\"\"
                if key not in self.cache:
                    self.cache[key] = self.processor.target_method(key)
                return self.cache[key]

            def clear_cache(self) -> None:
                \"\"\"Clear the internal cache.\"\"\"
                self.cache.clear()

            def get_stats(self) -> Dict[str, int]:
                \"\"\"Get cache statistics.\"\"\"
                return {
                    "cache_size": len(self.cache),
                    "hits": sum(1 for v in self.cache.values() if v is not None)
                }

    except Exception as e:
        class ResultHandler:
            def __init__(self):
                self.error = str(e)

            def target_method(self, key: str) -> None:
                raise RuntimeError(f"Failed to initialize: {self.error}")
    """

    expected = """
    class DataProcessor:
        \"\"\"A simple data processing class.\"\"\"

        def __init__(self, data: Dict[str, Any]) -> None:
            self.data = data
            self._processed = False
            self.result = None

        def __repr__(self) -> str:
            return f"DataProcessor(processed={self._processed})"

        def target_method(self, key: str) -> Optional[Any]:
            \"\"\"Process and retrieve a specific key from the data.\"\"\"
            if not self._processed:
                self._process_data()
            return self.result.get(key) if self.result else None

    try:
        sample_data = {"number": 42, "text": "hello"}
        processor = DataProcessor(sample_data)

        class ResultHandler:
            def __init__(self, processor: DataProcessor):
                self.processor = processor
                self.cache = {}

            def __str__(self) -> str:
                return f"ResultHandler(cache_size={len(self.cache)})"

            def target_method(self, key: str) -> Optional[Any]:
                \"\"\"Retrieve and cache results for a key.\"\"\"
                if key not in self.cache:
                    self.cache[key] = self.processor.target_method(key)
                return self.cache[key]

    except Exception as e:
        class ResultHandler:
            def __init__(self):
                self.error = str(e)

            def target_method(self, key: str) -> None:
                raise RuntimeError(f"Failed to initialize: {self.error}")
    """

    output = parse_code_and_prune_cst(dedent(code), CodeContextType.TESTGEN, {"DataProcessor.target_method", "ResultHandler.target_method"}, set())
    assert dedent(expected).strip() == output.strip()


def test_simplified_complete_implementation_no_docstring() -> None:
    code = """
    class DataProcessor:
        \"\"\"A simple data processing class.\"\"\"
        def __repr__(self) -> str:
            return f"DataProcessor(processed={self._processed})"

        def target_method(self, key: str) -> Optional[Any]:
            \"\"\"Process and retrieve a specific key from the data.\"\"\"
            if not self._processed:
                self._process_data()
            return self.result.get(key) if self.result else None

        def _process_data(self) -> None:
            \"\"\"Internal method to process the data.\"\"\"
            processed = {}
            for key, value in self.data.items():
                if isinstance(value, (int, float)):
                    processed[key] = value * 2
                elif isinstance(value, str):
                    processed[key] = value.upper()
                else:
                    processed[key] = value
            self.result = processed
            self._processed = True

        def to_json(self) -> str:
            \"\"\"Convert the processed data to JSON string.\"\"\"
            if not self._processed:
                self._process_data()
            return json.dumps(self.result)

    try:
        sample_data = {"number": 42, "text": "hello"}
        processor = DataProcessor(sample_data)

        class ResultHandler:

            def __str__(self) -> str:
                return f"ResultHandler(cache_size={len(self.cache)})"

            def target_method(self, key: str) -> Optional[Any]:
                \"\"\"Retrieve and cache results for a key.\"\"\"
                if key not in self.cache:
                    self.cache[key] = self.processor.target_method(key)
                return self.cache[key]

            def clear_cache(self) -> None:
                \"\"\"Clear the internal cache.\"\"\"
                self.cache.clear()

            def get_stats(self) -> Dict[str, int]:
                \"\"\"Get cache statistics.\"\"\"
                return {
                    "cache_size": len(self.cache),
                    "hits": sum(1 for v in self.cache.values() if v is not None)
                }

    except Exception as e:
        class ResultHandler:

            def target_method(self, key: str) -> None:
                raise RuntimeError(f"Failed to initialize: {self.error}")
    """

    expected = """
    class DataProcessor:
        def __repr__(self) -> str:
            return f"DataProcessor(processed={self._processed})"

        def target_method(self, key: str) -> Optional[Any]:
            if not self._processed:
                self._process_data()
            return self.result.get(key) if self.result else None

    try:
        sample_data = {"number": 42, "text": "hello"}
        processor = DataProcessor(sample_data)

        class ResultHandler:

            def __str__(self) -> str:
                return f"ResultHandler(cache_size={len(self.cache)})"

            def target_method(self, key: str) -> Optional[Any]:
                if key not in self.cache:
                    self.cache[key] = self.processor.target_method(key)
                return self.cache[key]

    except Exception as e:
        class ResultHandler:

            def target_method(self, key: str) -> None:
                raise RuntimeError(f"Failed to initialize: {self.error}")
    """

    output = parse_code_and_prune_cst(
        dedent(code), CodeContextType.TESTGEN, {"DataProcessor.target_method", "ResultHandler.target_method"}, set(), remove_docstrings=True
    )
    assert dedent(expected).strip() == output.strip()
