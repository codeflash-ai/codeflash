from textwrap import dedent

import pytest
from codeflash.optimization.cst_manipulator import get_read_only_code


def test_basic_class() -> None:
    code = """
    class TestClass:
        class_var = "value"

        def target_method(self):
            print("This should be stubbed")

        def other_method(self):
            print("This too")
    """

    expected = """
    class TestClass:
        class_var = "value"
    """

    output = get_read_only_code(dedent(code), {"TestClass.target_method"})
    assert dedent(expected).strip() == output.strip()


def test_dunder_methods() -> None:
    code = """
    class TestClass:
        def __init__(self):
            self.x = 42

        def __str__(self):
            return f"Value: {self.x}"

        def target_method(self):
            print("stub me")
    """

    expected = """
    class TestClass:
        def __init__(self):
            self.x = 42

        def __str__(self):
            return f"Value: {self.x}"
    """

    output = get_read_only_code(dedent(code), {"TestClass.target_method"})
    assert dedent(expected).strip() == output.strip()


def test_target_in_nested_class() -> None:
    """Test that attempting to find a target in a nested class raises an error."""
    code = """
    class Outer:
        outer_var = 1

        class Inner:
            inner_var = 2

            def target_method(self):
                print("stub this")
    """

    with pytest.raises(ValueError, match="No target functions found in the provided code"):
        get_read_only_code(dedent(code), {"Outer.Inner.target_method"})


def test_docstrings() -> None:
    code = """
    class TestClass:
        \"\"\"Class docstring.\"\"\"

        def target_method(self):
            \"\"\"Method docstring.\"\"\"
            print("stub this")

        def other_method(self):
            \"\"\"Other docstring.\"\"\"
            print("stub this too")
    """

    expected = """
    class TestClass:
        \"\"\"Class docstring.\"\"\"
    """

    output = get_read_only_code(dedent(code), {"TestClass.target_method"})
    assert dedent(expected).strip() == output.strip()


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

    expected = """"""

    output = get_read_only_code(dedent(code), {"TestClass.target_method"})
    assert dedent(expected).strip() == output.strip()


def test_multiple_top_level_targets() -> None:
    code = """
    class TestClass:
        def target1(self):
            print("stub 1")

        def target2(self):
            print("stub 2")

        def __init__(self):
            self.x = 42
    """

    expected = """
    class TestClass:

        def __init__(self):
            self.x = 42
    """

    output = get_read_only_code(dedent(code), {"TestClass.target1", "TestClass.target2"})
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
    """

    output = get_read_only_code(dedent(code), {"TestClass.target_method"})
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
    """

    output = get_read_only_code(dedent(code), {"TestClass.target_method"})
    assert dedent(expected).strip() == output.strip()


def test_class_annotations_try() -> None:
    code = """
    try:
        class TestClass:
            var1: int = 42
            var2: str

            def target_method(self) -> None:
                self.var2 = "test"
    except Exception:
        continue
    """

    expected = """
    try:
        class TestClass:
            var1: int = 42
            var2: str
    except Exception:
        continue
    """

    output = get_read_only_code(dedent(code), {"TestClass.target_method"})
    assert dedent(expected).strip() == output.strip()


def test_class_annotations_else() -> None:
    code = """
    if x is True:
        class TestClass:
            var1: int = 42
            var2: str

            def wrong_method(self) -> None:
                print("wrong")
    else:
        class TestClass:
            var1: int = 42
            var2: str

            def target_method(self) -> None:
                self.var2 = "test"
    """

    expected = """
    if x is True:
        class TestClass:
            var1: int = 42
            var2: str

            def wrong_method(self) -> None:
                print("wrong")
    else:
        class TestClass:
            var1: int = 42
            var2: str
    """

    output = get_read_only_code(dedent(code), {"TestClass.target_method"})
    assert dedent(expected).strip() == output.strip()


def test_top_level_functions() -> None:
    code = """
    def target_function(self) -> None:
        self.var2 = "test"

    def some_function():
        print("wow")
    """

    expected = """"""

    output = get_read_only_code(dedent(code), {"target_function"})
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
    x = 5
    """

    output = get_read_only_code(dedent(code), {"target_function"})
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
    if y:
        x = 5
    else: 
        z = 10
    """

    output = get_read_only_code(dedent(code), {"target_function"})
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
    elif PLATFORM == "windows":
        class PlatformClass:
            platform = "windows"
    else:
        class PlatformClass:
            platform = "other"
    """

    output = get_read_only_code(dedent(code), {"PlatformClass.target_method"})
    assert dedent(expected).strip() == output.strip()


def test_multiple_except_clauses() -> None:
    code = """
    try:
        class TestClass:
            error_type = None
            def target_method(self):
                print("main")
    except ValueError:
        class TestClass:
            error_type = "value_error"
            def target_method(self):
                print("value error")
    except TypeError:
        class TestClass:
            error_type = "type_error"
            def target_method(self):
                print("type error")
    except Exception as e:
        class TestClass:
            error_type = "generic_error"
            def target_method(self):
                print("generic error")
    else:
        class TestClass:
            error_type = "no_error"
            def target_method(self):
                print("no error")
    finally:
        class TestClass:
            error_type = "cleanup"
            def target_method(self):
                print("cleanup")
    """

    expected = """
    try:
        class TestClass:
            error_type = None
    except ValueError:
        class TestClass:
            error_type = "value_error"
    except TypeError:
        class TestClass:
            error_type = "type_error"
    except Exception as e:
        class TestClass:
            error_type = "generic_error"
    else:
        class TestClass:
            error_type = "no_error"
    finally:
        class TestClass:
            error_type = "cleanup"
    """

    output = get_read_only_code(dedent(code), {"TestClass.target_method"})
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
                    else:
                        class TestClass:
                            context = "not_ready"
            except ConnectionError:
                class TestClass:
                    context = "connection_error"
                continue
            finally:
                class TestClass:
                    context = "cleanup"
    """

    output = get_read_only_code(dedent(code), {"TestClass.target_method"})
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
                elif await item.can_retry():
                    continue
                else:
                    break
        except AsyncIOError:
            class TestClass:
                status = "io_error"
        except CancelledError:
            class TestClass:
                status = "cancelled"
    """

    output = get_read_only_code(dedent(code), {"TestClass.target_method"})
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

    try:
        sample_data = {"number": 42, "text": "hello"}
        processor = DataProcessor(sample_data)

        class ResultHandler:
            def __init__(self, processor: DataProcessor):
                self.processor = processor
                self.cache = {}

            def __str__(self) -> str:
                return f"ResultHandler(cache_size={len(self.cache)})"

    except Exception as e:
        class ResultHandler:
            def __init__(self):
                self.error = str(e)
    """

    output = get_read_only_code(dedent(code), {"DataProcessor.target_method", "ResultHandler.target_method"})
    assert dedent(expected).strip() == output.strip()
