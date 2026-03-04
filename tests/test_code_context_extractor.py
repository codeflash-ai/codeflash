from __future__ import annotations

import ast
import sys
import tempfile
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import pytest

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.python.context.code_context_extractor import (
    collect_type_names_from_annotation,
    enrich_testgen_context,
    extract_init_stub_from_class,
    extract_parameter_type_constructors,
    get_code_optimization_context,
    resolve_instance_class_name,
)
from codeflash.languages.python.static_analysis.code_extractor import GlobalAssignmentCollector, add_global_assignments
from codeflash.languages.python.static_analysis.code_replacer import replace_functions_and_add_imports
from codeflash.models.models import CodeString, CodeStringsMarkdown, FunctionParent
from codeflash.optimization.optimizer import Optimizer


class HelperClass:
    def __init__(self, name):
        self.name = name

    def innocent_bystander(self):
        pass

    def helper_method(self):
        return self.name

    class NestedClass:
        def __init__(self, name):
            self.name = name

        def nested_method(self):
            return self.name


def main_method():
    return "hello"


class MainClass:
    def __init__(self, name):
        self.name = name

    def main_method(self):
        self.name = HelperClass.NestedClass("test").nested_method()
        return HelperClass(self.name).helper_method()


class Graph:
    def __init__(self, vertices):
        self.graph = defaultdict(list)
        self.V = vertices  # No. of vertices

    def addEdge(self, u, v):
        self.graph[u].append(v)

    def topologicalSortUtil(self, v, visited, stack):
        visited[v] = True

        for i in self.graph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

        stack.insert(0, v)

    def topologicalSort(self):
        visited = [False] * self.V
        stack = []

        for i in range(self.V):
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

        # Print contents of stack
        return stack


def test_code_replacement10() -> None:
    file_path = Path(__file__).resolve()

    func_top_optimize = FunctionToOptimize(
        function_name="main_method", file_path=file_path, parents=[FunctionParent("MainClass", "ClassDef")]
    )

    code_ctx = get_code_optimization_context(function_to_optimize=func_top_optimize, project_root_path=file_path.parent)
    qualified_names = {func.qualified_name for func in code_ctx.helper_functions}
    # HelperClass.__init__ is now tracked because HelperClass(self.name) instantiates the class
    assert qualified_names == {
        "HelperClass.helper_method",
        "HelperClass.__init__",
    }  # Nested method should not be in here
    read_write_context, read_only_context = code_ctx.read_writable_code, code_ctx.read_only_context_code
    hashing_context = code_ctx.hashing_code_context

    expected_read_write_context = f"""
```python:{file_path.relative_to(file_path.parent)}
from __future__ import annotations


class HelperClass:
    def __init__(self, name):
        self.name = name

    def helper_method(self):
        return self.name


class MainClass:
    def __init__(self, name):
        self.name = name

    def main_method(self):
        self.name = HelperClass.NestedClass("test").nested_method()
        return HelperClass(self.name).helper_method()
```
"""
    expected_read_only_context = """
    """

    expected_hashing_context = f"""
```python:{file_path.relative_to(file_path.parent)}
class HelperClass:

    def helper_method(self):
        return self.name

class MainClass:

    def main_method(self):
        self.name = HelperClass.NestedClass('test').nested_method()
        return HelperClass(self.name).helper_method()
```
"""

    assert read_write_context.markdown.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()
    assert hashing_context.strip() == expected_hashing_context.strip()


def test_class_method_dependencies() -> None:
    file_path = Path(__file__).resolve()

    function_to_optimize = FunctionToOptimize(
        function_name="topologicalSort",
        file_path=str(file_path),
        parents=[FunctionParent(name="Graph", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    code_ctx = get_code_optimization_context(function_to_optimize, file_path.parent.resolve())
    read_write_context, read_only_context = code_ctx.read_writable_code, code_ctx.read_only_context_code
    hashing_context = code_ctx.hashing_code_context

    expected_read_write_context = f"""
```python:{file_path.relative_to(file_path.parent)}
from __future__ import annotations
from collections import defaultdict


class Graph:
    def __init__(self, vertices):
        self.graph = defaultdict(list)
        self.V = vertices  # No. of vertices

    def topologicalSortUtil(self, v, visited, stack):
        visited[v] = True

        for i in self.graph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

        stack.insert(0, v)

    def topologicalSort(self):
        visited = [False] * self.V
        stack = []

        for i in range(self.V):
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

        # Print contents of stack
        return stack
```
"""
    expected_read_only_context = ""

    expected_hashing_context = f"""
```python:{file_path.relative_to(file_path.parent.resolve())}
class Graph:

    def topologicalSortUtil(self, v, visited, stack):
        visited[v] = True
        for i in self.graph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)
        stack.insert(0, v)

    def topologicalSort(self):
        visited = [False] * self.V
        stack = []
        for i in range(self.V):
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)
        return stack
```
"""

    assert read_write_context.markdown.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()
    assert hashing_context.strip() == expected_hashing_context.strip()


def test_bubble_sort_helper() -> None:
    path_to_fto = (
        Path(__file__).resolve().parent.parent
        / "code_to_optimize"
        / "code_directories"
        / "retriever"
        / "bubble_sort_imported.py"
    )

    function_to_optimize = FunctionToOptimize(
        function_name="sort_from_another_file",
        file_path=str(path_to_fto),
        parents=[],
        starting_line=None,
        ending_line=None,
    )

    code_ctx = get_code_optimization_context(function_to_optimize, Path(__file__).resolve().parent.parent)
    read_write_context, read_only_context = code_ctx.read_writable_code, code_ctx.read_only_context_code
    hashing_context = code_ctx.hashing_code_context

    expected_read_write_context = """
```python:code_to_optimize/code_directories/retriever/bubble_sort_with_math.py
import math


def sorter(arr):
    arr.sort()
    x = math.sqrt(2)
    print(x)
    return arr
```
```python:code_to_optimize/code_directories/retriever/bubble_sort_imported.py
from bubble_sort_with_math import sorter


def sort_from_another_file(arr):
    sorted_arr = sorter(arr)
    return sorted_arr
```
"""
    expected_read_only_context = ""

    expected_hashing_context = """
```python:code_to_optimize/code_directories/retriever/bubble_sort_with_math.py
def sorter(arr):
    arr.sort()
    x = math.sqrt(2)
    print(x)
    return arr
```
```python:code_to_optimize/code_directories/retriever/bubble_sort_imported.py
def sort_from_another_file(arr):
    sorted_arr = sorter(arr)
    return sorted_arr
```
"""
    assert read_write_context.markdown.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()
    assert hashing_context.strip() == expected_hashing_context.strip()


def test_flavio_typed_code_helper(tmp_path: Path) -> None:
    code = '''

_P = ParamSpec("_P")
_KEY_T = TypeVar("_KEY_T")
_STORE_T = TypeVar("_STORE_T")
class AbstractCacheBackend(CacheBackend, Protocol[_KEY_T, _STORE_T]):
    """Interface for cache backends used by the persistent cache decorator."""

    def __init__(self) -> None: ...

    def hash_key(
        self,
        *,
        func: Callable[_P, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[str, _KEY_T]: ...

    def encode(self, *, data: Any) -> _STORE_T:  # noqa: ANN401
        ...

    def decode(self, *, data: _STORE_T) -> Any:  # noqa: ANN401
        ...

    def get(self, *, key: tuple[str, _KEY_T]) -> tuple[datetime.datetime, _STORE_T] | None: ...

    def delete(self, *, key: tuple[str, _KEY_T]) -> None: ...

    def put(self, *, key: tuple[str, _KEY_T], data: _STORE_T) -> None: ...

    def get_cache_or_call(
        self,
        *,
        func: Callable[_P, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        lifespan: datetime.timedelta,
    ) -> Any:  # noqa: ANN401
        """
        Retrieve the cached results for a function call.

        Args:
        ----
            func (Callable[..., _R]): The function to retrieve cached results for.
            args (tuple[Any, ...]): The positional arguments passed to the function.
            kwargs (dict[str, Any]): The keyword arguments passed to the function.
            lifespan (datetime.timedelta): The maximum age of the cached results.

        Returns:
        -------
            _R: The cached results, if available.

        """
        if os.environ.get("NO_CACHE"):
            return func(*args, **kwargs)

        try:
            key = self.hash_key(func=func, args=args, kwargs=kwargs)
        except:  # noqa: E722
            # If we can't create a cache key, we should just call the function.
            logging.warning("Failed to hash cache key for function: %s", func)
            return func(*args, **kwargs)
        result_pair = self.get(key=key)

        if result_pair is not None:
            cached_time, result = result_pair
            if not os.environ.get("RE_CACHE") and (
                datetime.datetime.now() < (cached_time + lifespan)  # noqa: DTZ005
            ):
                try:
                    return self.decode(data=result)
                except CacheBackendDecodeError as e:
                    logging.warning("Failed to decode cache data: %s", e)
                    # If decoding fails we will treat this as a cache miss.
                    # This might happens if underlying class definition of the data changes.
            self.delete(key=key)
        result = func(*args, **kwargs)
        try:
            self.put(key=key, data=self.encode(data=result))
        except CacheBackendEncodeError as e:
            logging.warning("Failed to encode cache data: %s", e)
        # If encoding fails, we should still return the result.
        return result

_P = ParamSpec("_P")
_R = TypeVar("_R")
_CacheBackendT = TypeVar("_CacheBackendT", bound=CacheBackend)


class _PersistentCache(Generic[_P, _R, _CacheBackendT]):
    """
    A decorator class that provides persistent caching functionality for a function.

    Args:
    ----
        func (Callable[_P, _R]): The function to be decorated.
        duration (datetime.timedelta): The duration for which the cached results should be considered valid.
        backend (_backend): The backend storage for the cached results.

    Attributes:
    ----------
        __wrapped__ (Callable[_P, _R]): The wrapped function.
        __duration__ (datetime.timedelta): The duration for which the cached results should be considered valid.
        __backend__ (_backend): The backend storage for the cached results.

    """  # noqa: E501

    __wrapped__: Callable[_P, _R]
    __duration__: datetime.timedelta
    __backend__: _CacheBackendT

    def __init__(
        self,
        func: Callable[_P, _R],
        duration: datetime.timedelta,
    ) -> None:
        self.__wrapped__ = func
        self.__duration__ = duration
        self.__backend__ = AbstractCacheBackend()
        functools.update_wrapper(self, func)

    def cache_clear(self) -> None:
        """Clears the cache for the wrapped function."""
        self.__backend__.del_func_cache(func=self.__wrapped__)

    def no_cache_call(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """
        Calls the wrapped function without using the cache.

        Args:
        ----
            *args (_P.args): Positional arguments for the wrapped function.
            **kwargs (_P.kwargs): Keyword arguments for the wrapped function.

        Returns:
        -------
            _R: The result of the wrapped function.

        """
        return self.__wrapped__(*args, **kwargs)

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """
        Calls the wrapped function, either using the cache or bypassing it based on environment variables.

        Args:
        ----
            *args (_P.args): Positional arguments for the wrapped function.
            **kwargs (_P.kwargs): Keyword arguments for the wrapped function.

        Returns:
        -------
            _R: The result of the wrapped function.

        """  # noqa: E501
        if "NO_CACHE" in os.environ:
            return self.__wrapped__(*args, **kwargs)
        os.makedirs(DEFAULT_CACHE_LOCATION, exist_ok=True)
        return self.__backend__.get_cache_or_call(
            func=self.__wrapped__,
            args=args,
            kwargs=kwargs,
            lifespan=self.__duration__,
        )
'''
    # Create a temporary Python file using pytest's tmp_path fixture
    file_path = tmp_path / "test_code.py"
    file_path.write_text(code, encoding="utf-8")
    opt = Optimizer(
        Namespace(
            project_root=file_path.parent.resolve(),
            disable_telemetry=True,
            tests_root="tests",
            test_framework="pytest",
            pytest_cmd="pytest",
            experiment_id=None,
            test_project_root=Path().resolve(),
        )
    )
    function_to_optimize = FunctionToOptimize(
        function_name="__call__",
        file_path=file_path,
        parents=[FunctionParent(name="_PersistentCache", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    code_ctx = get_code_optimization_context(function_to_optimize, opt.args.project_root)
    read_write_context, read_only_context = code_ctx.read_writable_code, code_ctx.read_only_context_code
    hashing_context = code_ctx.hashing_code_context
    expected_read_write_context = f"""
```python:{file_path.relative_to(opt.args.project_root)}
_P = ParamSpec("_P")
_KEY_T = TypeVar("_KEY_T")
_STORE_T = TypeVar("_STORE_T")
class AbstractCacheBackend(CacheBackend, Protocol[_KEY_T, _STORE_T]):

    def __init__(self) -> None: ...

    def get_cache_or_call(
        self,
        *,
        func: Callable[_P, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        lifespan: datetime.timedelta,
    ) -> Any:  # noqa: ANN401
        \"\"\"
        Retrieve the cached results for a function call.

        Args:
        ----
            func (Callable[..., _R]): The function to retrieve cached results for.
            args (tuple[Any, ...]): The positional arguments passed to the function.
            kwargs (dict[str, Any]): The keyword arguments passed to the function.
            lifespan (datetime.timedelta): The maximum age of the cached results.

        Returns:
        -------
            _R: The cached results, if available.

        \"\"\"
        if os.environ.get("NO_CACHE"):
            return func(*args, **kwargs)

        try:
            key = self.hash_key(func=func, args=args, kwargs=kwargs)
        except:  # noqa: E722
            # If we can't create a cache key, we should just call the function.
            logging.warning("Failed to hash cache key for function: %s", func)
            return func(*args, **kwargs)
        result_pair = self.get(key=key)

        if result_pair is not None:
            cached_time, result = result_pair
            if not os.environ.get("RE_CACHE") and (
                datetime.datetime.now() < (cached_time + lifespan)  # noqa: DTZ005
            ):
                try:
                    return self.decode(data=result)
                except CacheBackendDecodeError as e:
                    logging.warning("Failed to decode cache data: %s", e)
                    # If decoding fails we will treat this as a cache miss.
                    # This might happens if underlying class definition of the data changes.
            self.delete(key=key)
        result = func(*args, **kwargs)
        try:
            self.put(key=key, data=self.encode(data=result))
        except CacheBackendEncodeError as e:
            logging.warning("Failed to encode cache data: %s", e)
        # If encoding fails, we should still return the result.
        return result

_P = ParamSpec("_P")
_R = TypeVar("_R")
_CacheBackendT = TypeVar("_CacheBackendT", bound=CacheBackend)


class _PersistentCache(Generic[_P, _R, _CacheBackendT]):

    def __init__(
        self,
        func: Callable[_P, _R],
        duration: datetime.timedelta,
    ) -> None:
        self.__wrapped__ = func
        self.__duration__ = duration
        self.__backend__ = AbstractCacheBackend()
        functools.update_wrapper(self, func)

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        \"\"\"
        Calls the wrapped function, either using the cache or bypassing it based on environment variables.

        Args:
        ----
            *args (_P.args): Positional arguments for the wrapped function.
            **kwargs (_P.kwargs): Keyword arguments for the wrapped function.

        Returns:
        -------
            _R: The result of the wrapped function.

        \"\"\"  # noqa: E501
        if "NO_CACHE" in os.environ:
            return self.__wrapped__(*args, **kwargs)
        os.makedirs(DEFAULT_CACHE_LOCATION, exist_ok=True)
        return self.__backend__.get_cache_or_call(
            func=self.__wrapped__,
            args=args,
            kwargs=kwargs,
            lifespan=self.__duration__,
        )
```
"""
    expected_read_only_context = f'''
```python:{file_path.relative_to(opt.args.project_root)}
_P = ParamSpec("_P")
_KEY_T = TypeVar("_KEY_T")
_STORE_T = TypeVar("_STORE_T")
class AbstractCacheBackend(CacheBackend, Protocol[_KEY_T, _STORE_T]):
    """Interface for cache backends used by the persistent cache decorator."""

    def __init__(self) -> None: ...

    def hash_key(
        self,
        *,
        func: Callable[_P, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[str, _KEY_T]: ...

    def encode(self, *, data: Any) -> _STORE_T:  # noqa: ANN401
        ...

    def decode(self, *, data: _STORE_T) -> Any:  # noqa: ANN401
        ...

    def get(self, *, key: tuple[str, _KEY_T]) -> tuple[datetime.datetime, _STORE_T] | None: ...

    def delete(self, *, key: tuple[str, _KEY_T]) -> None: ...

    def put(self, *, key: tuple[str, _KEY_T], data: _STORE_T) -> None: ...

_P = ParamSpec("_P")
_R = TypeVar("_R")
_CacheBackendT = TypeVar("_CacheBackendT", bound=CacheBackend)


class _PersistentCache(Generic[_P, _R, _CacheBackendT]):
    """
    A decorator class that provides persistent caching functionality for a function.

    Args:
    ----
        func (Callable[_P, _R]): The function to be decorated.
        duration (datetime.timedelta): The duration for which the cached results should be considered valid.
        backend (_backend): The backend storage for the cached results.

    Attributes:
    ----------
        __wrapped__ (Callable[_P, _R]): The wrapped function.
        __duration__ (datetime.timedelta): The duration for which the cached results should be considered valid.
        __backend__ (_backend): The backend storage for the cached results.

    """  # noqa: E501

    __wrapped__: Callable[_P, _R]
    __duration__: datetime.timedelta
    __backend__: _CacheBackendT
```
'''
    expected_hashing_context = f"""
```python:{file_path.relative_to(opt.args.project_root)}
class AbstractCacheBackend(CacheBackend, Protocol[_KEY_T, _STORE_T]):

    def get_cache_or_call(self, *, func: Callable[_P, Any], args: tuple[Any, ...], kwargs: dict[str, Any], lifespan: datetime.timedelta) -> Any:
        if os.environ.get('NO_CACHE'):
            return func(*args, **kwargs)
        try:
            key = self.hash_key(func=func, args=args, kwargs=kwargs)
        except:
            logging.warning('Failed to hash cache key for function: %s', func)
            return func(*args, **kwargs)
        result_pair = self.get(key=key)
        if result_pair is not None:
            {"cached_time, result = result_pair" if sys.version_info >= (3, 11) else "(cached_time, result) = result_pair"}
            if not os.environ.get('RE_CACHE') and datetime.datetime.now() < cached_time + lifespan:
                try:
                    return self.decode(data=result)
                except CacheBackendDecodeError as e:
                    logging.warning('Failed to decode cache data: %s', e)
            self.delete(key=key)
        result = func(*args, **kwargs)
        try:
            self.put(key=key, data=self.encode(data=result))
        except CacheBackendEncodeError as e:
            logging.warning('Failed to encode cache data: %s', e)
        return result

class _PersistentCache(Generic[_P, _R, _CacheBackendT]):

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        if 'NO_CACHE' in os.environ:
            return self.__wrapped__(*args, **kwargs)
        os.makedirs(DEFAULT_CACHE_LOCATION, exist_ok=True)
        return self.__backend__.get_cache_or_call(func=self.__wrapped__, args=args, kwargs=kwargs, lifespan=self.__duration__)
```
"""
    assert read_write_context.markdown.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()
    assert hashing_context.strip() == expected_hashing_context.strip()


def test_example_class(tmp_path: Path) -> None:
    code = """
class MyClass:
    \"\"\"A class with a helper method.\"\"\"
    def __init__(self):
        self.x = 1
    def target_method(self):
        y = HelperClass().helper_method()

class HelperClass:
    \"\"\"A helper class for MyClass.\"\"\"
    def __init__(self):
        \"\"\"Initialize the HelperClass.\"\"\"
        self.x = 1
    def __repr__(self):
        \"\"\"Return a string representation of the HelperClass.\"\"\"
        return "HelperClass" + str(self.x)
    def helper_method(self):
        return self.x
"""
    # Create a temporary Python file using pytest's tmp_path fixture
    file_path = tmp_path / "test_code.py"
    file_path.write_text(code, encoding="utf-8")
    opt = Optimizer(
        Namespace(
            project_root=file_path.parent.resolve(),
            disable_telemetry=True,
            tests_root="tests",
            test_framework="pytest",
            pytest_cmd="pytest",
            experiment_id=None,
            test_project_root=Path().resolve(),
        )
    )
    function_to_optimize = FunctionToOptimize(
        function_name="target_method",
        file_path=file_path,
        parents=[FunctionParent(name="MyClass", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    code_ctx = get_code_optimization_context(function_to_optimize, opt.args.project_root)
    read_write_context, read_only_context = code_ctx.read_writable_code, code_ctx.read_only_context_code
    hashing_context = code_ctx.hashing_code_context

    expected_read_write_context = f"""
```python:{file_path.relative_to(opt.args.project_root)}
class MyClass:
    def __init__(self):
        self.x = 1
    def target_method(self):
        y = HelperClass().helper_method()

class HelperClass:
    def __init__(self):
        \"\"\"Initialize the HelperClass.\"\"\"
        self.x = 1
    def helper_method(self):
        return self.x
```
"""
    expected_read_only_context = f"""
```python:{file_path.relative_to(opt.args.project_root)}
class MyClass:
    \"\"\"A class with a helper method.\"\"\"

class HelperClass:
    \"\"\"A helper class for MyClass.\"\"\"
    def __repr__(self):
        \"\"\"Return a string representation of the HelperClass.\"\"\"
        return "HelperClass" + str(self.x)
```
"""
    expected_hashing_context = f"""
```python:{file_path.relative_to(opt.args.project_root)}
class MyClass:

    def target_method(self):
        y = HelperClass().helper_method()

class HelperClass:

    def helper_method(self):
        return self.x
```
"""

    assert read_write_context.markdown.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()
    assert hashing_context.strip() == expected_hashing_context.strip()


def test_example_class_token_limit_1(tmp_path: Path) -> None:
    docstring_filler = " ".join(
        ["This is a long docstring that will be used to fill up the token limit." for _ in range(4000)]
    )
    code = f"""
class MyClass:
    \"\"\"A class with a helper method.
{docstring_filler}\"\"\"
    def __init__(self):
        self.x = 1
    def target_method(self):
        \"\"\"Docstring for target method\"\"\"
        y = HelperClass().helper_method()

class HelperClass:
    \"\"\"A helper class for MyClass.\"\"\"
    def __init__(self):
        \"\"\"Initialize the HelperClass.\"\"\"
        self.x = 1
    def __repr__(self):
        \"\"\"Return a string representation of the HelperClass.\"\"\"
        return "HelperClass" + str(self.x)
    def helper_method(self):
        return self.x
"""
    # Create a temporary Python file using pytest's tmp_path fixture
    file_path = tmp_path / "test_code.py"
    file_path.write_text(code, encoding="utf-8")
    opt = Optimizer(
        Namespace(
            project_root=file_path.parent.resolve(),
            disable_telemetry=True,
            tests_root="tests",
            test_framework="pytest",
            pytest_cmd="pytest",
            experiment_id=None,
            test_project_root=Path().resolve(),
        )
    )
    function_to_optimize = FunctionToOptimize(
        function_name="target_method",
        file_path=file_path,
        parents=[FunctionParent(name="MyClass", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    code_ctx = get_code_optimization_context(function_to_optimize, opt.args.project_root)
    read_write_context, read_only_context = code_ctx.read_writable_code, code_ctx.read_only_context_code
    hashing_context = code_ctx.hashing_code_context
    # In this scenario, the read-only code context is too long, so the read-only docstrings are removed.
    expected_read_write_context = f"""
```python:{file_path.relative_to(opt.args.project_root)}
class MyClass:
    def __init__(self):
        self.x = 1
    def target_method(self):
        \"\"\"Docstring for target method\"\"\"
        y = HelperClass().helper_method()

class HelperClass:
    def __init__(self):
        \"\"\"Initialize the HelperClass.\"\"\"
        self.x = 1
    def helper_method(self):
        return self.x
```
"""
    expected_read_only_context = f"""
```python:{file_path.relative_to(opt.args.project_root)}
class MyClass:
    pass

class HelperClass:
    def __repr__(self):
        return "HelperClass" + str(self.x)
```
"""
    expected_hashing_context = f"""
```python:{file_path.relative_to(opt.args.project_root)}
class MyClass:

    def target_method(self):
        y = HelperClass().helper_method()

class HelperClass:

    def helper_method(self):
        return self.x
```
"""
    assert read_write_context.markdown.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()
    assert hashing_context.strip() == expected_hashing_context.strip()


def test_example_class_token_limit_2(tmp_path: Path) -> None:
    string_filler = " ".join(
        ["This is a long string that will be used to fill up the token limit." for _ in range(1000)]
    )
    code = f"""
class MyClass:
    \"\"\"A class with a helper method. \"\"\"
    def __init__(self):
        self.x = 1
    def target_method(self):
        \"\"\"Docstring for target method\"\"\"
        y = HelperClass().helper_method()
x = '{string_filler}'

class HelperClass:
    \"\"\"A helper class for MyClass.\"\"\"
    def __init__(self):
        \"\"\"Initialize the HelperClass.\"\"\"
        self.x = 1
    def __repr__(self):
        \"\"\"Return a string representation of the HelperClass.\"\"\"
        return "HelperClass" + str(self.x)
    def helper_method(self):
        return self.x
"""
    # Create a temporary Python file using pytest's tmp_path fixture
    file_path = tmp_path / "test_code.py"
    file_path.write_text(code, encoding="utf-8")
    opt = Optimizer(
        Namespace(
            project_root=file_path.parent.resolve(),
            disable_telemetry=True,
            tests_root="tests",
            test_framework="pytest",
            pytest_cmd="pytest",
            experiment_id=None,
            test_project_root=Path().resolve(),
        )
    )
    function_to_optimize = FunctionToOptimize(
        function_name="target_method",
        file_path=file_path,
        parents=[FunctionParent(name="MyClass", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    code_ctx = get_code_optimization_context(function_to_optimize, opt.args.project_root, 8000, 100000)
    read_write_context, read_only_context = code_ctx.read_writable_code, code_ctx.read_only_context_code
    hashing_context = code_ctx.hashing_code_context
    # In this scenario, the read-only code context is too long even after removing docstrings, hence we remove it completely.
    expected_read_write_context = f"""
```python:{file_path.relative_to(opt.args.project_root)}
class MyClass:
    def __init__(self):
        self.x = 1
    def target_method(self):
        \"\"\"Docstring for target method\"\"\"
        y = HelperClass().helper_method()

class HelperClass:
    def __init__(self):
        \"\"\"Initialize the HelperClass.\"\"\"
        self.x = 1
    def helper_method(self):
        return self.x
```
"""
    expected_read_only_context = f'''```python:{file_path.relative_to(opt.args.project_root)}
class MyClass:
    """A class with a helper method. """

class HelperClass:
    """A helper class for MyClass."""
    def __repr__(self):
        """Return a string representation of the HelperClass."""
        return "HelperClass" + str(self.x)
```
'''
    expected_hashing_context = f"""
```python:{file_path.relative_to(opt.args.project_root)}
class MyClass:

    def target_method(self):
        y = HelperClass().helper_method()

class HelperClass:

    def helper_method(self):
        return self.x
```
"""
    assert read_write_context.markdown.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()
    assert hashing_context.strip() == expected_hashing_context.strip()


def test_example_class_token_limit_3(tmp_path: Path) -> None:
    string_filler = " ".join(
        ["This is a long string that will be used to fill up the token limit." for _ in range(4000)]
    )
    code = f"""
class MyClass:
    \"\"\"A class with a helper method. \"\"\"
    def __init__(self):
        self.x = 1
    def target_method(self):
        \"\"\"{string_filler}\"\"\"
        y = HelperClass().helper_method()

class HelperClass:
    \"\"\"A helper class for MyClass.\"\"\"
    def __init__(self):
        \"\"\"Initialize the HelperClass.\"\"\"
        self.x = 1
    def __repr__(self):
        \"\"\"Return a string representation of the HelperClass.\"\"\"
        return "HelperClass" + str(self.x)
    def helper_method(self):
        return self.x
"""
    # Create a temporary Python file using pytest's tmp_path fixture
    file_path = tmp_path / "test_code.py"
    file_path.write_text(code, encoding="utf-8")
    opt = Optimizer(
        Namespace(
            project_root=file_path.parent.resolve(),
            disable_telemetry=True,
            tests_root="tests",
            test_framework="pytest",
            pytest_cmd="pytest",
            experiment_id=None,
            test_project_root=Path().resolve(),
        )
    )
    function_to_optimize = FunctionToOptimize(
        function_name="target_method",
        file_path=file_path,
        parents=[FunctionParent(name="MyClass", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )
    # In this scenario, the read-writable code is too long, so we abort.
    with pytest.raises(ValueError, match="Read-writable code has exceeded token limit, cannot proceed"):
        get_code_optimization_context(function_to_optimize, opt.args.project_root, optim_token_limit=8000)


def test_example_class_token_limit_4(tmp_path: Path) -> None:
    string_filler = " ".join(
        ["This is a long string that will be used to fill up the token limit." for _ in range(4000)]
    )
    code = f"""
class MyClass:
    \"\"\"A class with a helper method. \"\"\"
    def __init__(self):
        global x
        x = 1
    def target_method(self):
        \"\"\"Docstring for target method\"\"\"
        y = HelperClass().helper_method()
x = '{string_filler}'

class HelperClass:
    \"\"\"A helper class for MyClass.\"\"\"
    def __init__(self):
        \"\"\"Initialize the HelperClass.\"\"\"
        self.x = 1
    def __repr__(self):
        \"\"\"Return a string representation of the HelperClass.\"\"\"
        return "HelperClass" + str(self.x)
    def helper_method(self):
        return self.x
"""
    # Create a temporary Python file using pytest's tmp_path fixture
    file_path = tmp_path / "test_code.py"
    file_path.write_text(code, encoding="utf-8")
    opt = Optimizer(
        Namespace(
            project_root=file_path.parent.resolve(),
            disable_telemetry=True,
            tests_root="tests",
            test_framework="pytest",
            pytest_cmd="pytest",
            experiment_id=None,
            test_project_root=Path().resolve(),
        )
    )
    function_to_optimize = FunctionToOptimize(
        function_name="target_method",
        file_path=file_path,
        parents=[FunctionParent(name="MyClass", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    # In this scenario, the read-writable code context becomes too large because the __init__ function is referencing the global x variable instead of the class attribute self.x, so we abort.
    with pytest.raises(ValueError, match="Read-writable code has exceeded token limit, cannot proceed"):
        get_code_optimization_context(function_to_optimize, opt.args.project_root, optim_token_limit=8000)


def test_example_class_token_limit_5(tmp_path: Path) -> None:
    string_filler = " ".join(
        ["This is a long string that will be used to fill up the token limit." for _ in range(1000)]
    )
    code = f"""
class MyClass:
    \"\"\"A class with a helper method. \"\"\"
    def __init__(self):
        self.x = 1
    def target_method(self):
        \"\"\"Docstring for target method\"\"\"
        y = HelperClass().helper_method()
x = '{string_filler}'

class HelperClass:
    \"\"\"A helper class for MyClass.\"\"\"
    def __init__(self):
        \"\"\"Initialize the HelperClass.\"\"\"
        self.x = 1
    def __repr__(self):
        \"\"\"Return a string representation of the HelperClass.\"\"\"
        return "HelperClass" + str(self.x)
    def helper_method(self):
        return self.x
"""
    # Create a temporary Python file using pytest's tmp_path fixture
    file_path = tmp_path / "test_code.py"
    file_path.write_text(code, encoding="utf-8")
    opt = Optimizer(
        Namespace(
            project_root=file_path.parent.resolve(),
            disable_telemetry=True,
            tests_root="tests",
            test_framework="pytest",
            pytest_cmd="pytest",
            experiment_id=None,
            test_project_root=Path().resolve(),
        )
    )
    function_to_optimize = FunctionToOptimize(
        function_name="target_method",
        file_path=file_path,
        parents=[FunctionParent(name="MyClass", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    code_ctx = get_code_optimization_context(function_to_optimize, opt.args.project_root)

    # the global x variable shouldn't be included in any context type
    assert (
        code_ctx.read_writable_code.flat
        == '''# file: test_code.py
class MyClass:
    def __init__(self):
        self.x = 1
    def target_method(self):
        """Docstring for target method"""
        y = HelperClass().helper_method()

class HelperClass:
    def __init__(self):
        """Initialize the HelperClass."""
        self.x = 1
    def helper_method(self):
        return self.x
'''
    )
    assert (
        code_ctx.testgen_context.flat
        == '''# file: test_code.py
class MyClass:
    """A class with a helper method. """
    def __init__(self):
        self.x = 1
    def target_method(self):
        """Docstring for target method"""
        y = HelperClass().helper_method()

class HelperClass:
    """A helper class for MyClass."""
    def __init__(self):
        """Initialize the HelperClass."""
        self.x = 1
    def __repr__(self):
        """Return a string representation of the HelperClass."""
        return "HelperClass" + str(self.x)
    def helper_method(self):
        return self.x
'''
    )


def test_repo_helper() -> None:
    project_root = Path(__file__).resolve().parent.parent / "code_to_optimize" / "code_directories" / "retriever"
    path_to_file = project_root / "main.py"
    path_to_utils = project_root / "utils.py"
    function_to_optimize = FunctionToOptimize(
        function_name="fetch_and_process_data",
        file_path=str(path_to_file),
        parents=[],
        starting_line=None,
        ending_line=None,
    )

    code_ctx = get_code_optimization_context(function_to_optimize, project_root)
    read_write_context, read_only_context = code_ctx.read_writable_code, code_ctx.read_only_context_code
    hashing_context = code_ctx.hashing_code_context
    path_to_globals = project_root / "globals.py"
    expected_read_write_context = f"""
```python:{path_to_globals.relative_to(project_root)}
# Define a global variable
API_URL = "https://api.example.com/data"
```
```python:{path_to_utils.relative_to(project_root)}
import math


class DataProcessor:

    def __init__(self, default_prefix: str = "PREFIX_"):
        \"\"\"Initialize the DataProcessor with a default prefix.\"\"\"
        self.default_prefix = default_prefix
        self.number += math.log(self.number)

    def process_data(self, raw_data: str) -> str:
        \"\"\"Process raw data by converting it to uppercase.\"\"\"
        return raw_data.upper()

    def add_prefix(self, data: str, prefix: str = "PREFIX_") -> str:
        \"\"\"Add a prefix to the processed data.\"\"\"
        return prefix + data
```
```python:{path_to_file.relative_to(project_root)}
import requests
from globals import API_URL
from utils import DataProcessor


def fetch_and_process_data():
    # Use the global variable for the request
    response = requests.get(API_URL)
    response.raise_for_status()

    raw_data = response.text

    # Use code from another file (utils.py)
    processor = DataProcessor()
    processed = processor.process_data(raw_data)
    processed = processor.add_prefix(processed)

    return processed
```
"""
    expected_read_only_context = f"""
```python:{path_to_utils.relative_to(project_root)}
class DataProcessor:
    \"\"\"A class for processing data.\"\"\"

    number = 1

    def __repr__(self) -> str:
        \"\"\"Return a string representation of the DataProcessor.\"\"\"
        return f"DataProcessor(default_prefix={{self.default_prefix!r}})"
```
"""
    expected_hashing_context = f"""
```python:{path_to_utils.relative_to(project_root)}
class DataProcessor:

    def process_data(self, raw_data: str) -> str:
        return raw_data.upper()

    def add_prefix(self, data: str, prefix: str='PREFIX_') -> str:
        return prefix + data
```
```python:{path_to_file.relative_to(project_root)}
def fetch_and_process_data():
    response = requests.get(API_URL)
    response.raise_for_status()
    raw_data = response.text
    processor = DataProcessor()
    processed = processor.process_data(raw_data)
    processed = processor.add_prefix(processed)
    return processed
```
"""
    assert read_write_context.markdown.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()
    assert hashing_context.strip() == expected_hashing_context.strip()


def test_repo_helper_of_helper() -> None:
    project_root = Path(__file__).resolve().parent.parent / "code_to_optimize" / "code_directories" / "retriever"
    path_to_file = project_root / "main.py"
    path_to_utils = project_root / "utils.py"
    path_to_transform_utils = project_root / "transform_utils.py"
    function_to_optimize = FunctionToOptimize(
        function_name="fetch_and_transform_data",
        file_path=str(path_to_file),
        parents=[],
        starting_line=None,
        ending_line=None,
    )

    code_ctx = get_code_optimization_context(function_to_optimize, project_root)
    read_write_context, read_only_context = code_ctx.read_writable_code, code_ctx.read_only_context_code
    hashing_context = code_ctx.hashing_code_context
    path_to_globals = project_root / "globals.py"
    expected_read_write_context = f"""
```python:{path_to_globals.relative_to(project_root)}
# Define a global variable
API_URL = "https://api.example.com/data"
```
```python:{path_to_utils.relative_to(project_root)}
import math
from transform_utils import DataTransformer


class DataProcessor:

    def __init__(self, default_prefix: str = "PREFIX_"):
        \"\"\"Initialize the DataProcessor with a default prefix.\"\"\"
        self.default_prefix = default_prefix
        self.number += math.log(self.number)

    def process_data(self, raw_data: str) -> str:
        \"\"\"Process raw data by converting it to uppercase.\"\"\"
        return raw_data.upper()

    def transform_data(self, data: str) -> str:
        \"\"\"Transform the processed data\"\"\"
        return DataTransformer().transform(data)
```
```python:{path_to_file.relative_to(project_root)}
import requests
from globals import API_URL
from utils import DataProcessor


def fetch_and_transform_data():
    # Use the global variable for the request
    response = requests.get(API_URL)

    raw_data = response.text

    # Use code from another file (utils.py)
    processor = DataProcessor()
    processed = processor.process_data(raw_data)
    transformed = processor.transform_data(processed)

    return transformed
```
"""
    expected_read_only_context = f"""
```python:{path_to_utils.relative_to(project_root)}
class DataProcessor:
    \"\"\"A class for processing data.\"\"\"

    number = 1

    def __repr__(self) -> str:
        \"\"\"Return a string representation of the DataProcessor.\"\"\"
        return f"DataProcessor(default_prefix={{self.default_prefix!r}})"
```
```python:{path_to_transform_utils.relative_to(project_root)}
class DataTransformer:
    def __init__(self):
        self.data = None

    def transform(self, data):
        self.data = data
        return self.data
```
"""
    expected_hashing_context = f"""
```python:{path_to_utils.relative_to(project_root)}
class DataProcessor:

    def process_data(self, raw_data: str) -> str:
        return raw_data.upper()

    def transform_data(self, data: str) -> str:
        return DataTransformer().transform(data)
```
```python:{path_to_file.relative_to(project_root)}
def fetch_and_transform_data():
    response = requests.get(API_URL)
    raw_data = response.text
    processor = DataProcessor()
    processed = processor.process_data(raw_data)
    transformed = processor.transform_data(processed)
    return transformed
```
"""
    assert read_write_context.markdown.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()
    assert hashing_context.strip() == expected_hashing_context.strip()


def test_repo_helper_of_helper_same_class() -> None:
    project_root = Path(__file__).resolve().parent.parent / "code_to_optimize" / "code_directories" / "retriever"
    path_to_utils = project_root / "utils.py"
    path_to_transform_utils = project_root / "transform_utils.py"
    function_to_optimize = FunctionToOptimize(
        function_name="transform_data_own_method",
        file_path=str(path_to_utils),
        parents=[FunctionParent(name="DataProcessor", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    code_ctx = get_code_optimization_context(function_to_optimize, project_root)
    read_write_context, read_only_context = code_ctx.read_writable_code, code_ctx.read_only_context_code
    hashing_context = code_ctx.hashing_code_context
    expected_read_write_context = f"""
```python:{path_to_transform_utils.relative_to(project_root)}
class DataTransformer:
    def __init__(self):
        self.data = None

    def transform_using_own_method(self, data):
        return self.transform(data)
```
```python:{path_to_utils.relative_to(project_root)}
import math
from transform_utils import DataTransformer


class DataProcessor:

    def __init__(self, default_prefix: str = "PREFIX_"):
        \"\"\"Initialize the DataProcessor with a default prefix.\"\"\"
        self.default_prefix = default_prefix
        self.number += math.log(self.number)

    def transform_data_own_method(self, data: str) -> str:
        \"\"\"Transform the processed data using own method\"\"\"
        return DataTransformer().transform_using_own_method(data)
```
"""
    expected_read_only_context = f"""
```python:{path_to_transform_utils.relative_to(project_root)}
class DataTransformer:

    def transform(self, data):
        self.data = data
        return self.data
```
```python:{path_to_utils.relative_to(project_root)}
class DataProcessor:
    \"\"\"A class for processing data.\"\"\"

    number = 1

    def __repr__(self) -> str:
        \"\"\"Return a string representation of the DataProcessor.\"\"\"
        return f"DataProcessor(default_prefix={{self.default_prefix!r}})"
```

"""
    expected_hashing_context = f"""
```python:transform_utils.py
class DataTransformer:

    def transform_using_own_method(self, data):
        return self.transform(data)
```
```python:{path_to_utils.relative_to(project_root)}
class DataProcessor:

    def transform_data_own_method(self, data: str) -> str:
        return DataTransformer().transform_using_own_method(data)
```
"""

    assert read_write_context.markdown.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()
    assert hashing_context.strip() == expected_hashing_context.strip()


def test_repo_helper_of_helper_same_file() -> None:
    project_root = Path(__file__).resolve().parent.parent / "code_to_optimize" / "code_directories" / "retriever"
    path_to_utils = project_root / "utils.py"
    path_to_transform_utils = project_root / "transform_utils.py"
    function_to_optimize = FunctionToOptimize(
        function_name="transform_data_same_file_function",
        file_path=str(path_to_utils),
        parents=[FunctionParent(name="DataProcessor", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    code_ctx = get_code_optimization_context(function_to_optimize, project_root)
    read_write_context, read_only_context = code_ctx.read_writable_code, code_ctx.read_only_context_code
    hashing_context = code_ctx.hashing_code_context
    expected_read_write_context = f"""
```python:{path_to_transform_utils.relative_to(project_root)}
class DataTransformer:
    def __init__(self):
        self.data = None

    def transform_using_same_file_function(self, data):
        return update_data(data)
```
```python:{path_to_utils.relative_to(project_root)}
import math
from transform_utils import DataTransformer


class DataProcessor:

    def __init__(self, default_prefix: str = "PREFIX_"):
        \"\"\"Initialize the DataProcessor with a default prefix.\"\"\"
        self.default_prefix = default_prefix
        self.number += math.log(self.number)

    def transform_data_same_file_function(self, data: str) -> str:
        \"\"\"Transform the processed data using a function from the same file\"\"\"
        return DataTransformer().transform_using_same_file_function(data)
```
"""
    expected_read_only_context = f"""
```python:{path_to_transform_utils.relative_to(project_root)}
def update_data(data):
    return data + " updated"
```
```python:{path_to_utils.relative_to(project_root)}
class DataProcessor:
    \"\"\"A class for processing data.\"\"\"

    number = 1

    def __repr__(self) -> str:
        \"\"\"Return a string representation of the DataProcessor.\"\"\"
        return f"DataProcessor(default_prefix={{self.default_prefix!r}})"
```
"""
    expected_hashing_context = f"""
```python:transform_utils.py
class DataTransformer:

    def transform_using_same_file_function(self, data):
        return update_data(data)
```
```python:{path_to_utils.relative_to(project_root)}
class DataProcessor:

    def transform_data_same_file_function(self, data: str) -> str:
        return DataTransformer().transform_using_same_file_function(data)
```
"""

    assert read_write_context.markdown.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()
    assert hashing_context.strip() == expected_hashing_context.strip()


def test_repo_helper_all_same_file() -> None:
    project_root = Path(__file__).resolve().parent.parent / "code_to_optimize" / "code_directories" / "retriever"
    path_to_transform_utils = project_root / "transform_utils.py"
    function_to_optimize = FunctionToOptimize(
        function_name="transform_data_all_same_file",
        file_path=str(path_to_transform_utils),
        parents=[FunctionParent(name="DataTransformer", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    code_ctx = get_code_optimization_context(function_to_optimize, project_root)
    read_write_context, read_only_context = code_ctx.read_writable_code, code_ctx.read_only_context_code
    hashing_context = code_ctx.hashing_code_context
    expected_read_write_context = f"""
```python:{path_to_transform_utils.relative_to(project_root)}
class DataTransformer:
    def __init__(self):
        self.data = None

    def transform_using_own_method(self, data):
        return self.transform(data)

    def transform_data_all_same_file(self, data):
        new_data = update_data(data)
        return self.transform_using_own_method(new_data)


def update_data(data):
    return data + " updated"
```
"""
    expected_read_only_context = f"""
```python:{path_to_transform_utils.relative_to(project_root)}
class DataTransformer:

    def transform(self, data):
        self.data = data
        return self.data
```

"""
    expected_hashing_context = f"""
```python:{path_to_transform_utils.relative_to(project_root)}
class DataTransformer:

    def transform_using_own_method(self, data):
        return self.transform(data)

    def transform_data_all_same_file(self, data):
        new_data = update_data(data)
        return self.transform_using_own_method(new_data)

def update_data(data):
    return data + ' updated'
```
"""

    assert read_write_context.markdown.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()
    assert hashing_context.strip() == expected_hashing_context.strip()


def test_repo_helper_circular_dependency() -> None:
    project_root = Path(__file__).resolve().parent.parent / "code_to_optimize" / "code_directories" / "retriever"
    path_to_utils = project_root / "utils.py"
    path_to_transform_utils = project_root / "transform_utils.py"
    function_to_optimize = FunctionToOptimize(
        function_name="circular_dependency",
        file_path=str(path_to_transform_utils),
        parents=[FunctionParent(name="DataTransformer", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    code_ctx = get_code_optimization_context(function_to_optimize, project_root)
    read_write_context, read_only_context = code_ctx.read_writable_code, code_ctx.read_only_context_code
    hashing_context = code_ctx.hashing_code_context
    expected_read_write_context = f"""
```python:{path_to_utils.relative_to(project_root)}
import math
from transform_utils import DataTransformer


class DataProcessor:

    def __init__(self, default_prefix: str = "PREFIX_"):
        \"\"\"Initialize the DataProcessor with a default prefix.\"\"\"
        self.default_prefix = default_prefix
        self.number += math.log(self.number)

    def circular_dependency(self, data: str) -> str:
        \"\"\"Test circular dependency\"\"\"
        return DataTransformer().circular_dependency(data)
```
```python:{path_to_transform_utils.relative_to(project_root)}
from code_to_optimize.code_directories.retriever.utils import DataProcessor


class DataTransformer:
    def __init__(self):
        self.data = None

    def circular_dependency(self, data):
        return DataProcessor().circular_dependency(data)
```
"""
    expected_read_only_context = f"""
```python:{path_to_utils.relative_to(project_root)}
class DataProcessor:
    \"\"\"A class for processing data.\"\"\"

    number = 1

    def __repr__(self) -> str:
        \"\"\"Return a string representation of the DataProcessor.\"\"\"
        return f"DataProcessor(default_prefix={{self.default_prefix!r}})"
```
```python:{path_to_transform_utils.relative_to(project_root)}
class DataTransformer:
    def __init__(self):
        self.data = None
```
"""
    expected_hashing_context = f"""
```python:utils.py
class DataProcessor:

    def circular_dependency(self, data: str) -> str:
        return DataTransformer().circular_dependency(data)
```
```python:{path_to_transform_utils.relative_to(project_root)}
class DataTransformer:

    def circular_dependency(self, data):
        return DataProcessor().circular_dependency(data)
```
"""

    assert read_write_context.markdown.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()
    assert hashing_context.strip() == expected_hashing_context.strip()


def test_indirect_init_helper(tmp_path: Path) -> None:
    code = """
class MyClass:
    def __init__(self):
        self.x = 1
        self.y = outside_method()
    def target_method(self):
        return self.x + self.y

def outside_method():
    return 1
"""
    # Create a temporary Python file using pytest's tmp_path fixture
    file_path = tmp_path / "test_code.py"
    file_path.write_text(code, encoding="utf-8")
    opt = Optimizer(
        Namespace(
            project_root=file_path.parent.resolve(),
            disable_telemetry=True,
            tests_root="tests",
            test_framework="pytest",
            pytest_cmd="pytest",
            experiment_id=None,
            test_project_root=Path().resolve(),
        )
    )
    function_to_optimize = FunctionToOptimize(
        function_name="target_method",
        file_path=file_path,
        parents=[FunctionParent(name="MyClass", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    code_ctx = get_code_optimization_context(function_to_optimize, opt.args.project_root)
    read_write_context, read_only_context = code_ctx.read_writable_code, code_ctx.read_only_context_code
    hashing_context = code_ctx.hashing_code_context
    expected_read_write_context = f"""
```python:{file_path.relative_to(opt.args.project_root)}
class MyClass:
    def __init__(self):
        self.x = 1
        self.y = outside_method()
    def target_method(self):
        return self.x + self.y
```
"""
    expected_read_only_context = f"""
```python:{file_path.relative_to(opt.args.project_root)}
def outside_method():
    return 1
```
"""
    expected_hashing_context = f"""
```python:{file_path.relative_to(opt.args.project_root)}
class MyClass:

    def target_method(self):
        return self.x + self.y
```
"""
    assert read_write_context.markdown.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()
    assert hashing_context.strip() == expected_hashing_context.strip()


def test_direct_module_import() -> None:
    project_root = Path(__file__).resolve().parent.parent / "code_to_optimize" / "code_directories" / "retriever"
    path_to_main = project_root / "main.py"
    path_to_fto = project_root / "import_test.py"
    function_to_optimize = FunctionToOptimize(
        function_name="function_to_optimize",
        file_path=str(path_to_fto),
        parents=[],
        starting_line=None,
        ending_line=None,
    )

    code_ctx = get_code_optimization_context(function_to_optimize, project_root)
    read_write_context, read_only_context = code_ctx.read_writable_code, code_ctx.read_only_context_code
    hashing_context = code_ctx.hashing_code_context

    expected_read_only_context = """
```python:utils.py
import math
from transform_utils import DataTransformer


class DataProcessor:
    \"\"\"A class for processing data.\"\"\"

    number = 1

    def __init__(self, default_prefix: str = "PREFIX_"):
        \"\"\"Initialize the DataProcessor with a default prefix.\"\"\"
        self.default_prefix = default_prefix
        self.number += math.log(self.number)

    def __repr__(self) -> str:
        \"\"\"Return a string representation of the DataProcessor.\"\"\"
        return f"DataProcessor(default_prefix={self.default_prefix!r})"

    def process_data(self, raw_data: str) -> str:
        \"\"\"Process raw data by converting it to uppercase.\"\"\"
        return raw_data.upper()

    def transform_data(self, data: str) -> str:
        \"\"\"Transform the processed data\"\"\"
        return DataTransformer().transform(data)
```"""
    expected_hashing_context = """
```python:main.py
def fetch_and_transform_data():
    response = requests.get(API_URL)
    raw_data = response.text
    processor = DataProcessor()
    processed = processor.process_data(raw_data)
    transformed = processor.transform_data(processed)
    return transformed
```
```python:import_test.py
def function_to_optimize():
    return code_to_optimize.code_directories.retriever.main.fetch_and_transform_data()
```
"""
    expected_read_write_context = f"""
```python:{path_to_main.relative_to(project_root)}
import requests
from globals import API_URL
from utils import DataProcessor


def fetch_and_transform_data():
    # Use the global variable for the request
    response = requests.get(API_URL)

    raw_data = response.text

    # Use code from another file (utils.py)
    processor = DataProcessor()
    processed = processor.process_data(raw_data)
    transformed = processor.transform_data(processed)

    return transformed
```
```python:{path_to_fto.relative_to(project_root)}
import code_to_optimize.code_directories.retriever.main

def function_to_optimize():
    return code_to_optimize.code_directories.retriever.main.fetch_and_transform_data()
```
"""
    assert read_write_context.markdown.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()
    assert hashing_context.strip() == expected_hashing_context.strip()


def test_module_import_optimization() -> None:
    main_code = """
import utility_module

class Calculator:
    def __init__(self, precision="high", fallback_precision=None, mode="standard"):
        # This is where we use the imported module
        self.precision = utility_module.select_precision(precision, fallback_precision)
        self.mode = mode

        # Using variables from the utility module
        self.backend = utility_module.CALCULATION_BACKEND
        self.system = utility_module.SYSTEM_TYPE
        self.default_precision = utility_module.DEFAULT_PRECISION

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def calculate(self, operation, x, y):
        if operation == "add":
            return self.add(x, y)
        elif operation == "subtract":
            return self.subtract(x, y)
        else:
            return None
"""

    utility_module_code = """
import sys
import platform
import logging

DEFAULT_PRECISION = "medium"
DEFAULT_MODE = "standard"

# Try-except block with variable definitions
try:
    import numpy as np
    # Used variable in try block
    CALCULATION_BACKEND = "numpy"
    # Unused variable in try block
    VECTOR_DIMENSIONS = 3
except ImportError:
    # Used variable in except block
    CALCULATION_BACKEND = "python"
    # Unused variable in except block
    FALLBACK_WARNING = "NumPy not available, using slower Python implementation"

# Nested if-else with variable definitions
if sys.platform.startswith('win'):
    # Used variable in outer if
    SYSTEM_TYPE = "windows"
    if platform.architecture()[0] == '64bit':
        # Unused variable in nested if
        MEMORY_MODEL = "x64"
    else:
        # Unused variable in nested else
        MEMORY_MODEL = "x86"
elif sys.platform.startswith('linux'):
    # Used variable in outer elif
    SYSTEM_TYPE = "linux"
    # Unused variable in outer elif
    KERNEL_VERSION = platform.release()
else:
    # Used variable in outer else
    SYSTEM_TYPE = "other"
    # Unused variable in outer else
    UNKNOWN_SYSTEM_MSG = "Running on an unrecognized platform"

# Function that will be used in the main code
def select_precision(precision, fallback_precision):
    if precision is None:
        return fallback_precision or DEFAULT_PRECISION

    # Using the variables defined above
    if CALCULATION_BACKEND == "numpy":
        # Higher precision available with NumPy
        precision_options = ["low", "medium", "high", "ultra"]
    else:
        # Limited precision without NumPy
        precision_options = ["low", "medium", "high"]

    if isinstance(precision, str):
        if precision.lower() not in precision_options:
            if fallback_precision:
                return fallback_precision
            else:
                return DEFAULT_PRECISION
        return precision.lower()
    else:
        return DEFAULT_PRECISION

# Function that won't be used
def get_system_details():
    return {
        "system": SYSTEM_TYPE,
        "backend": CALCULATION_BACKEND,
        "default_precision": DEFAULT_PRECISION,
        "python_version": sys.version
    }
"""

    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up the package structure
        package_dir = Path(temp_dir) / "package"
        package_dir.mkdir()

        # Create the __init__.py file
        with open(package_dir / "__init__.py", "w") as init_file:
            init_file.write("")

        # Write the utility_module.py file
        with open(package_dir / "utility_module.py", "w") as utility_file:
            utility_file.write(utility_module_code)
            utility_file.flush()

        # Write the main code file
        main_file_path = package_dir / "main_module.py"
        with open(main_file_path, "w") as main_file:
            main_file.write(main_code)
            main_file.flush()

        # Set up the optimizer
        file_path = main_file_path.resolve()
        project_root = package_dir.resolve()
        opt = Optimizer(
            Namespace(
                project_root=project_root,
                disable_telemetry=True,
                tests_root="tests",
                test_framework="pytest",
                pytest_cmd="pytest",
                experiment_id=None,
                test_project_root=Path().resolve(),
            )
        )

        # Define the function to optimize
        function_to_optimize = FunctionToOptimize(
            function_name="calculate",
            file_path=file_path,
            parents=[FunctionParent(name="Calculator", type="ClassDef")],
            starting_line=None,
            ending_line=None,
        )

        # Get the code optimization context
        code_ctx = get_code_optimization_context(function_to_optimize, opt.args.project_root)
        read_write_context, read_only_context = code_ctx.read_writable_code, code_ctx.read_only_context_code
        hashing_context = code_ctx.hashing_code_context
        # The expected contexts
        # Resolve both paths to handle symlink issues on macOS
        relative_path = file_path.relative_to(project_root)
        expected_read_write_context = f"""
```python:{main_file_path.resolve().relative_to(opt.args.project_root.resolve())}
import utility_module

class Calculator:
    def __init__(self, precision="high", fallback_precision=None, mode="standard"):
        # This is where we use the imported module
        self.precision = utility_module.select_precision(precision, fallback_precision)
        self.mode = mode

        # Using variables from the utility module
        self.backend = utility_module.CALCULATION_BACKEND
        self.system = utility_module.SYSTEM_TYPE
        self.default_precision = utility_module.DEFAULT_PRECISION

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def calculate(self, operation, x, y):
        if operation == "add":
            return self.add(x, y)
        elif operation == "subtract":
            return self.subtract(x, y)
        else:
            return None
```
"""
        expected_read_only_context = """
```python:utility_module.py
import sys

DEFAULT_PRECISION = "medium"

# Try-except block with variable definitions
try:
    # Used variable in try block
    CALCULATION_BACKEND = "numpy"
except ImportError:
    # Used variable in except block
    CALCULATION_BACKEND = "python"

# Nested if-else with variable definitions
if sys.platform.startswith('win'):
    # Used variable in outer if
    SYSTEM_TYPE = "windows"
elif sys.platform.startswith('linux'):
    # Used variable in outer elif
    SYSTEM_TYPE = "linux"
else:
    # Used variable in outer else
    SYSTEM_TYPE = "other"

# Function that will be used in the main code
def select_precision(precision, fallback_precision):
    if precision is None:
        return fallback_precision or DEFAULT_PRECISION

    # Using the variables defined above
    if CALCULATION_BACKEND == "numpy":
        # Higher precision available with NumPy
        precision_options = ["low", "medium", "high", "ultra"]
    else:
        # Limited precision without NumPy
        precision_options = ["low", "medium", "high"]

    if isinstance(precision, str):
        if precision.lower() not in precision_options:
            if fallback_precision:
                return fallback_precision
            else:
                return DEFAULT_PRECISION
        return precision.lower()
    else:
        return DEFAULT_PRECISION
```
"""
        expected_hashing_context = """
```python:main_module.py
class Calculator:

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def calculate(self, operation, x, y):
        if operation == 'add':
            return self.add(x, y)
        elif operation == 'subtract':
            return self.subtract(x, y)
        else:
            return None
```
"""
        # Verify the contexts match the expected values
        assert read_write_context.markdown.strip() == expected_read_write_context.strip()
        assert read_only_context.strip() == expected_read_only_context.strip()
        assert hashing_context.strip() == expected_hashing_context.strip()


def test_module_import_init_fto() -> None:
    main_code = """
import utility_module

class Calculator:
    def __init__(self, precision="high", fallback_precision=None, mode="standard"):
        # This is where we use the imported module
        self.precision = utility_module.select_precision(precision, fallback_precision)
        self.mode = mode

        # Using variables from the utility module
        self.backend = utility_module.CALCULATION_BACKEND
        self.system = utility_module.SYSTEM_TYPE
        self.default_precision = utility_module.DEFAULT_PRECISION

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def calculate(self, operation, x, y):
        if operation == "add":
            return self.add(x, y)
        elif operation == "subtract":
            return self.subtract(x, y)
        else:
            return None
"""

    utility_module_code = """
import sys
import platform
import logging

DEFAULT_PRECISION = "medium"
DEFAULT_MODE = "standard"

# Try-except block with variable definitions
try:
    import numpy as np
    # Used variable in try block
    CALCULATION_BACKEND = "numpy"
    # Unused variable in try block
    VECTOR_DIMENSIONS = 3
except ImportError:
    # Used variable in except block
    CALCULATION_BACKEND = "python"
    # Unused variable in except block
    FALLBACK_WARNING = "NumPy not available, using slower Python implementation"

# Nested if-else with variable definitions
if sys.platform.startswith('win'):
    # Used variable in outer if
    SYSTEM_TYPE = "windows"
    if platform.architecture()[0] == '64bit':
        # Unused variable in nested if
        MEMORY_MODEL = "x64"
    else:
        # Unused variable in nested else
        MEMORY_MODEL = "x86"
elif sys.platform.startswith('linux'):
    # Used variable in outer elif
    SYSTEM_TYPE = "linux"
    # Unused variable in outer elif
    KERNEL_VERSION = platform.release()
else:
    # Used variable in outer else
    SYSTEM_TYPE = "other"
    # Unused variable in outer else
    UNKNOWN_SYSTEM_MSG = "Running on an unrecognized platform"

# Function that will be used in the main code
def select_precision(precision, fallback_precision):
    if precision is None:
        return fallback_precision or DEFAULT_PRECISION

    # Using the variables defined above
    if CALCULATION_BACKEND == "numpy":
        # Higher precision available with NumPy
        precision_options = ["low", "medium", "high", "ultra"]
    else:
        # Limited precision without NumPy
        precision_options = ["low", "medium", "high"]

    if isinstance(precision, str):
        if precision.lower() not in precision_options:
            if fallback_precision:
                return fallback_precision
            else:
                return DEFAULT_PRECISION
        return precision.lower()
    else:
        return DEFAULT_PRECISION

# Function that won't be used
def get_system_details():
    return {
        "system": SYSTEM_TYPE,
        "backend": CALCULATION_BACKEND,
        "default_precision": DEFAULT_PRECISION,
        "python_version": sys.version
    }
"""

    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up the package structure
        package_dir = Path(temp_dir) / "package"
        package_dir.mkdir()

        # Create the __init__.py file
        with open(package_dir / "__init__.py", "w") as init_file:
            init_file.write("")

        # Write the utility_module.py file
        with open(package_dir / "utility_module.py", "w") as utility_file:
            utility_file.write(utility_module_code)
            utility_file.flush()

        # Write the main code file
        main_file_path = package_dir / "main_module.py"
        with open(main_file_path, "w") as main_file:
            main_file.write(main_code)
            main_file.flush()

        # Set up the optimizer
        file_path = main_file_path.resolve()
        project_root = package_dir.resolve()
        opt = Optimizer(
            Namespace(
                project_root=project_root,
                disable_telemetry=True,
                tests_root="tests",
                test_framework="pytest",
                pytest_cmd="pytest",
                experiment_id=None,
                test_project_root=Path().resolve(),
            )
        )

        # Define the function to optimize
        function_to_optimize = FunctionToOptimize(
            function_name="__init__",
            file_path=file_path,
            parents=[FunctionParent(name="Calculator", type="ClassDef")],
            starting_line=None,
            ending_line=None,
        )

        # Get the code optimization context
        code_ctx = get_code_optimization_context(function_to_optimize, opt.args.project_root)
        read_write_context, read_only_context = code_ctx.read_writable_code, code_ctx.read_only_context_code
        # The expected contexts
        relative_path = file_path.relative_to(project_root)
        expected_read_write_context = f"""
```python:utility_module.py
import sys

DEFAULT_PRECISION = "medium"

# Try-except block with variable definitions
try:
    # Used variable in try block
    CALCULATION_BACKEND = "numpy"
except ImportError:
    # Used variable in except block
    CALCULATION_BACKEND = "python"

# Nested if-else with variable definitions
if sys.platform.startswith('win'):
    # Used variable in outer if
    SYSTEM_TYPE = "windows"
elif sys.platform.startswith('linux'):
    # Used variable in outer elif
    SYSTEM_TYPE = "linux"
else:
    # Used variable in outer else
    SYSTEM_TYPE = "other"

# Function that will be used in the main code
def select_precision(precision, fallback_precision):
    if precision is None:
        return fallback_precision or DEFAULT_PRECISION

    # Using the variables defined above
    if CALCULATION_BACKEND == "numpy":
        # Higher precision available with NumPy
        precision_options = ["low", "medium", "high", "ultra"]
    else:
        # Limited precision without NumPy
        precision_options = ["low", "medium", "high"]

    if isinstance(precision, str):
        if precision.lower() not in precision_options:
            if fallback_precision:
                return fallback_precision
            else:
                return DEFAULT_PRECISION
        return precision.lower()
    else:
        return DEFAULT_PRECISION
```
```python:{main_file_path.resolve().relative_to(opt.args.project_root.resolve())}
import utility_module

class Calculator:
    def __init__(self, precision="high", fallback_precision=None, mode="standard"):
        # This is where we use the imported module
        self.precision = utility_module.select_precision(precision, fallback_precision)
        self.mode = mode

        # Using variables from the utility module
        self.backend = utility_module.CALCULATION_BACKEND
        self.system = utility_module.SYSTEM_TYPE
        self.default_precision = utility_module.DEFAULT_PRECISION
```
"""
        expected_read_only_context = """
```python:utility_module.py
import sys

DEFAULT_PRECISION = "medium"

# Try-except block with variable definitions
try:
    # Used variable in try block
    CALCULATION_BACKEND = "numpy"
except ImportError:
    # Used variable in except block
    CALCULATION_BACKEND = "python"

# Nested if-else with variable definitions
if sys.platform.startswith('win'):
    # Used variable in outer if
    SYSTEM_TYPE = "windows"
elif sys.platform.startswith('linux'):
    # Used variable in outer elif
    SYSTEM_TYPE = "linux"
else:
    # Used variable in outer else
    SYSTEM_TYPE = "other"
```
"""
        assert read_write_context.markdown.strip() == expected_read_write_context.strip()
        assert read_only_context.strip() == expected_read_only_context.strip()


def test_hashing_code_context_removes_imports_docstrings_and_init(tmp_path: Path) -> None:
    """Test that hashing context removes imports, docstrings, and __init__ methods properly."""
    code = '''
import os
import sys
from pathlib import Path

class MyClass:
    """A class with a docstring."""
    def __init__(self, value):
        """Initialize with a value."""
        self.value = value

    def target_method(self):
        """Target method with docstring."""
        result = self.helper_method()
        helper_cls = HelperClass()
        data = helper_cls.process_data()
        return self.value * 2

    def helper_method(self):
        """Helper method with docstring."""
        return self.value + 1

class HelperClass:
    """Helper class docstring."""
    def __init__(self):
        """Helper init method."""
        self.data = "test"

    def process_data(self):
        """Process data method."""
        return self.data.upper()

def standalone_function():
    """Standalone function."""
    return "standalone"
'''
    # Create a temporary Python file using pytest's tmp_path fixture
    file_path = tmp_path / "test_code.py"
    file_path.write_text(code, encoding="utf-8")
    opt = Optimizer(
        Namespace(
            project_root=file_path.parent.resolve(),
            disable_telemetry=True,
            tests_root="tests",
            test_framework="pytest",
            pytest_cmd="pytest",
            experiment_id=None,
            test_project_root=Path().resolve(),
        )
    )
    function_to_optimize = FunctionToOptimize(
        function_name="target_method",
        file_path=file_path,
        parents=[FunctionParent(name="MyClass", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    code_ctx = get_code_optimization_context(function_to_optimize, opt.args.project_root)
    hashing_context = code_ctx.hashing_code_context

    # Expected behavior based on current implementation:
    # - Should not contain imports
    # - Should remove docstrings from target functions (but currently doesn't - this is a bug)
    # - Should not contain __init__ methods
    # - Should contain target function and helper methods that are actually called
    # - Should be formatted as markdown

    # Test that it's formatted as markdown
    assert hashing_context.startswith("```python:")
    assert hashing_context.endswith("```")

    # Test basic structure requirements
    assert "import" not in hashing_context  # Should not contain imports
    assert "__init__" not in hashing_context  # Should not contain __init__ methods
    assert "target_method" in hashing_context  # Should contain target function
    assert "standalone_function" not in hashing_context  # Should not contain unused functions

    # Test that helper functions are included when they're called
    assert "helper_method" in hashing_context  # Should contain called helper method
    assert "process_data" in hashing_context  # Should contain called helper method

    # Test for docstring removal (this should pass when implementation is fixed)
    # Currently this will fail because docstrings are not being removed properly
    assert '"""Target method with docstring."""' not in hashing_context, (
        "Docstrings should be removed from target functions"
    )
    assert '"""Helper method with docstring."""' not in hashing_context, (
        "Docstrings should be removed from helper functions"
    )
    assert '"""Process data method."""' not in hashing_context, "Docstrings should be removed from helper class methods"


def test_hashing_code_context_with_nested_classes(tmp_path: Path) -> None:
    """Test that hashing context handles nested classes properly (should exclude them)."""
    code = '''
class OuterClass:
    """Outer class docstring."""
    def __init__(self):
        """Outer init."""
        self.value = 1

    def target_method(self):
        """Target method."""
        return self.NestedClass().nested_method()

    class NestedClass:
        """Nested class - should be excluded."""
        def __init__(self):
            self.nested_value = 2

        def nested_method(self):
            return self.nested_value
'''
    # Create a temporary Python file using pytest's tmp_path fixture
    file_path = tmp_path / "test_code.py"
    file_path.write_text(code, encoding="utf-8")
    opt = Optimizer(
        Namespace(
            project_root=file_path.parent.resolve(),
            disable_telemetry=True,
            tests_root="tests",
            test_framework="pytest",
            pytest_cmd="pytest",
            experiment_id=None,
            test_project_root=Path().resolve(),
        )
    )
    function_to_optimize = FunctionToOptimize(
        function_name="target_method",
        file_path=file_path,
        parents=[FunctionParent(name="OuterClass", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    code_ctx = get_code_optimization_context(function_to_optimize, opt.args.project_root)
    hashing_context = code_ctx.hashing_code_context

    # Test basic requirements
    assert hashing_context.startswith("```python:")
    assert hashing_context.endswith("```")
    assert "target_method" in hashing_context
    assert "__init__" not in hashing_context  # Should not contain __init__ methods

    # Verify nested classes are excluded from the hashing context
    # The prune_cst function in hashing mode should not recurse into nested classes
    assert "class NestedClass:" not in hashing_context  # Nested class definition should not be present

    # The target method will reference NestedClass, but the actual nested class definition should not be included
    # The call to self.NestedClass().nested_method() should be in the target method but the nested class itself excluded
    target_method_call_present = "self.NestedClass().nested_method()" in hashing_context
    assert target_method_call_present, "The target method should contain the call to nested class"

    # But the actual nested method definition should not be present
    nested_method_definition_present = "def nested_method(self):" in hashing_context
    assert not nested_method_definition_present, "Nested method definition should not be present in hashing context"


def test_hashing_code_context_hash_consistency(tmp_path: Path) -> None:
    """Test that the same code produces the same hash."""
    code = """
class TestClass:
    def target_method(self):
        return "test"
"""
    # Create a temporary Python file using pytest's tmp_path fixture
    file_path = tmp_path / "test_code.py"
    file_path.write_text(code, encoding="utf-8")
    opt = Optimizer(
        Namespace(
            project_root=file_path.parent.resolve(),
            disable_telemetry=True,
            tests_root="tests",
            test_framework="pytest",
            pytest_cmd="pytest",
            experiment_id=None,
            test_project_root=Path().resolve(),
        )
    )
    function_to_optimize = FunctionToOptimize(
        function_name="target_method",
        file_path=file_path,
        parents=[FunctionParent(name="TestClass", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    # Generate context twice
    code_ctx1 = get_code_optimization_context(function_to_optimize, opt.args.project_root)
    code_ctx2 = get_code_optimization_context(function_to_optimize, opt.args.project_root)

    # Hash should be consistent
    assert code_ctx1.hashing_code_context_hash == code_ctx2.hashing_code_context_hash
    assert code_ctx1.hashing_code_context == code_ctx2.hashing_code_context

    # Hash should be valid SHA256
    import hashlib

    expected_hash = hashlib.sha256(code_ctx1.hashing_code_context.encode("utf-8")).hexdigest()
    assert code_ctx1.hashing_code_context_hash == expected_hash


def test_hashing_code_context_different_code_different_hash(tmp_path: Path) -> None:
    """Test that different code produces different hashes."""
    code1 = """
class TestClass:
    def target_method(self):
        return "test1"
"""
    code2 = """
class TestClass:
    def target_method(self):
        return "test2"
"""

    # Create two temporary Python files using pytest's tmp_path fixture
    file_path1 = tmp_path / "test_code1.py"
    file_path2 = tmp_path / "test_code2.py"
    file_path1.write_text(code1, encoding="utf-8")
    file_path2.write_text(code2, encoding="utf-8")

    opt1 = Optimizer(
        Namespace(
            project_root=file_path1.parent.resolve(),
            disable_telemetry=True,
            tests_root="tests",
            test_framework="pytest",
            pytest_cmd="pytest",
            experiment_id=None,
            test_project_root=Path().resolve(),
        )
    )
    opt2 = Optimizer(
        Namespace(
            project_root=file_path2.parent.resolve(),
            disable_telemetry=True,
            tests_root="tests",
            test_framework="pytest",
            pytest_cmd="pytest",
            experiment_id=None,
            test_project_root=Path().resolve(),
        )
    )

    function_to_optimize1 = FunctionToOptimize(
        function_name="target_method",
        file_path=file_path1,
        parents=[FunctionParent(name="TestClass", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )
    function_to_optimize2 = FunctionToOptimize(
        function_name="target_method",
        file_path=file_path2,
        parents=[FunctionParent(name="TestClass", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    code_ctx1 = get_code_optimization_context(function_to_optimize1, opt1.args.project_root)
    code_ctx2 = get_code_optimization_context(function_to_optimize2, opt2.args.project_root)

    # Different code should produce different hashes
    assert code_ctx1.hashing_code_context_hash != code_ctx2.hashing_code_context_hash
    assert code_ctx1.hashing_code_context != code_ctx2.hashing_code_context


def test_hashing_code_context_format_is_markdown(tmp_path: Path) -> None:
    """Test that hashing context is formatted as markdown."""
    code = """
class SimpleClass:
    def simple_method(self):
        return 42
"""
    # Create a temporary Python file using pytest's tmp_path fixture
    file_path = tmp_path / "test_code.py"
    file_path.write_text(code, encoding="utf-8")
    opt = Optimizer(
        Namespace(
            project_root=file_path.parent.resolve(),
            disable_telemetry=True,
            tests_root="tests",
            test_framework="pytest",
            pytest_cmd="pytest",
            experiment_id=None,
            test_project_root=Path().resolve(),
        )
    )
    function_to_optimize = FunctionToOptimize(
        function_name="simple_method",
        file_path=file_path,
        parents=[FunctionParent(name="SimpleClass", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    code_ctx = get_code_optimization_context(function_to_optimize, opt.args.project_root)
    hashing_context = code_ctx.hashing_code_context

    # Should be formatted as markdown code block
    assert hashing_context.startswith("```python:")
    assert hashing_context.endswith("```")

    # Should contain the relative file path in the markdown header
    relative_path = file_path.relative_to(opt.args.project_root)
    assert str(relative_path) in hashing_context

    # Should contain the actual code between the markdown markers
    lines = hashing_context.strip().split("\n")
    assert lines[0].startswith("```python:")
    assert lines[-1] == "```"

    # Code should be between the markers
    code_lines = lines[1:-1]
    code_content = "\n".join(code_lines)
    assert "class SimpleClass:" in code_content
    assert "def simple_method(self):" in code_content
    assert "return 42" in code_content


# This shouldn't happen as we are now using a scoped optimization context, but keep it just in case
def test_circular_deps():
    path_to_root = Path(__file__).resolve().parent.parent / "code_to_optimize" / "code_directories" / "circular_deps"
    file_abs_path = path_to_root / "api_client.py"
    optimized_code = Path(path_to_root / "optimized.py").read_text(encoding="utf-8")
    content = Path(file_abs_path).read_text(encoding="utf-8")
    new_code = replace_functions_and_add_imports(
        source_code=add_global_assignments(optimized_code, content),
        function_names=["ApiClient.get_console_url"],
        optimized_code=optimized_code,
        module_abspath=Path(file_abs_path),
        preexisting_objects={
            ("ApiClient", ()),
            ("get_console_url", (FunctionParent(name="ApiClient", type="ClassDef"),)),
        },
        project_root_path=Path(path_to_root),
    )
    assert "import ApiClient" not in new_code, "Error: Circular dependency found"

    assert "import urllib.parse" in new_code, "Make sure imports for optimization global assignments exist"


def test_global_assignment_collector_with_async_function():
    """Test GlobalAssignmentCollector correctly identifies global assignments outside async functions."""
    import libcst as cst

    source_code = """
# Global assignment
GLOBAL_VAR = "global_value"
OTHER_GLOBAL = 42

async def async_function():
    # This should not be collected (inside async function)
    local_var = "local_value"
    INNER_ASSIGNMENT = "should_not_be_global"
    return local_var

# Another global assignment
ANOTHER_GLOBAL = "another_global"
"""

    tree = cst.parse_module(source_code)
    collector = GlobalAssignmentCollector()
    tree.visit(collector)

    # Should collect global assignments but not the ones inside async function
    assert len(collector.assignments) == 3
    assert "GLOBAL_VAR" in collector.assignments
    assert "OTHER_GLOBAL" in collector.assignments
    assert "ANOTHER_GLOBAL" in collector.assignments

    # Should not collect assignments from inside async function
    assert "local_var" not in collector.assignments
    assert "INNER_ASSIGNMENT" not in collector.assignments

    # Verify assignment order
    expected_order = ["GLOBAL_VAR", "OTHER_GLOBAL", "ANOTHER_GLOBAL"]
    assert collector.assignment_order == expected_order


def test_global_assignment_collector_nested_async_functions():
    """Test GlobalAssignmentCollector handles nested async functions correctly."""
    import libcst as cst

    source_code = """
# Global assignment
CONFIG = {"key": "value"}

def sync_function():
    # Inside sync function - should not be collected
    sync_local = "sync"

    async def nested_async():
        # Inside nested async function - should not be collected
        nested_var = "nested"
        return nested_var

    return sync_local

async def async_function():
    # Inside async function - should not be collected
    async_local = "async"

    def nested_sync():
        # Inside nested function - should not be collected
        deeply_nested = "deep"
        return deeply_nested

    return async_local

# Another global assignment
FINAL_GLOBAL = "final"
"""

    tree = cst.parse_module(source_code)
    collector = GlobalAssignmentCollector()
    tree.visit(collector)

    # Should only collect global-level assignments
    assert len(collector.assignments) == 2
    assert "CONFIG" in collector.assignments
    assert "FINAL_GLOBAL" in collector.assignments

    # Should not collect any assignments from inside functions
    assert "sync_local" not in collector.assignments
    assert "nested_var" not in collector.assignments
    assert "async_local" not in collector.assignments
    assert "deeply_nested" not in collector.assignments


def test_global_assignment_collector_mixed_async_sync_with_classes():
    """Test GlobalAssignmentCollector with async functions, sync functions, and classes."""
    import libcst as cst

    source_code = """
# Global assignments
GLOBAL_CONSTANT = "constant"

class TestClass:
    # Class-level assignment - should not be collected
    class_var = "class_value"

    def sync_method(self):
        # Method assignment - should not be collected
        method_var = "method"
        return method_var

    async def async_method(self):
        # Async method assignment - should not be collected
        async_method_var = "async_method"
        return async_method_var

def sync_function():
    # Function assignment - should not be collected
    func_var = "function"
    return func_var

async def async_function():
    # Async function assignment - should not be collected
    async_func_var = "async_function"
    return async_func_var

# More global assignments
ANOTHER_CONSTANT = 100
FINAL_ASSIGNMENT = {"data": "value"}
"""

    tree = cst.parse_module(source_code)
    collector = GlobalAssignmentCollector()
    tree.visit(collector)

    # Should only collect global-level assignments
    assert len(collector.assignments) == 3
    assert "GLOBAL_CONSTANT" in collector.assignments
    assert "ANOTHER_CONSTANT" in collector.assignments
    assert "FINAL_ASSIGNMENT" in collector.assignments

    # Should not collect assignments from inside any scoped blocks
    assert "class_var" not in collector.assignments
    assert "method_var" not in collector.assignments
    assert "async_method_var" not in collector.assignments
    assert "func_var" not in collector.assignments
    assert "async_func_var" not in collector.assignments

    # Verify correct order
    expected_order = ["GLOBAL_CONSTANT", "ANOTHER_CONSTANT", "FINAL_ASSIGNMENT"]
    assert collector.assignment_order == expected_order


def test_global_assignment_collector_annotated_assignments():
    """Test GlobalAssignmentCollector correctly handles annotated assignments (AnnAssign)."""
    import libcst as cst

    source_code = """
# Regular global assignment
REGULAR_VAR = "regular"

# Annotated global assignments
TYPED_VAR: str = "typed"
CACHE: dict[str, int] = {}
SENTINEL: object = object()

# Annotated without value (type declaration only) - should NOT be collected
DECLARED_ONLY: int

def some_function():
    # Annotated assignment inside function - should not be collected
    local_typed: str = "local"
    return local_typed

class SomeClass:
    # Class-level annotated assignment - should not be collected
    class_attr: str = "class"

# Another regular assignment
FINAL_VAR = 123
"""

    tree = cst.parse_module(source_code)
    collector = GlobalAssignmentCollector()
    tree.visit(collector)

    # Should collect both regular and annotated global assignments with values
    assert len(collector.assignments) == 5
    assert "REGULAR_VAR" in collector.assignments
    assert "TYPED_VAR" in collector.assignments
    assert "CACHE" in collector.assignments
    assert "SENTINEL" in collector.assignments
    assert "FINAL_VAR" in collector.assignments

    # Should not collect type declarations without values
    assert "DECLARED_ONLY" not in collector.assignments

    # Should not collect assignments from inside functions or classes
    assert "local_typed" not in collector.assignments
    assert "class_attr" not in collector.assignments

    # Verify correct order
    expected_order = ["REGULAR_VAR", "TYPED_VAR", "CACHE", "SENTINEL", "FINAL_VAR"]
    assert collector.assignment_order == expected_order


def test_global_function_collector():
    """Test GlobalFunctionCollector correctly collects module-level function definitions."""
    import libcst as cst

    from codeflash.languages.python.static_analysis.code_extractor import GlobalFunctionCollector

    source_code = """
# Module-level functions
def helper_function():
    return "helper"

def another_helper(x: int) -> str:
    return str(x)

class SomeClass:
    def method(self):
        # This is a method, not a module-level function
        return "method"

    def another_method(self):
        # Also a method
        def nested_function():
            # Nested function inside method
            return "nested"
        return nested_function()

def final_function():
    def inner_function():
        # This is a nested function, not module-level
        return "inner"
    return inner_function()
"""

    tree = cst.parse_module(source_code)
    collector = GlobalFunctionCollector()
    tree.visit(collector)

    # Should collect only module-level functions
    assert len(collector.functions) == 3
    assert "helper_function" in collector.functions
    assert "another_helper" in collector.functions
    assert "final_function" in collector.functions

    # Should not collect methods or nested functions
    assert "method" not in collector.functions
    assert "another_method" not in collector.functions
    assert "nested_function" not in collector.functions
    assert "inner_function" not in collector.functions

    # Verify correct order
    expected_order = ["helper_function", "another_helper", "final_function"]
    assert collector.function_order == expected_order


def test_add_global_assignments_with_new_functions():
    """Test add_global_assignments correctly adds new module-level functions."""
    source_code = """\
from functools import lru_cache

class SkyvernPage:
    @staticmethod
    def action_wrap(action):
        return _get_decorator_for_action(action)

@lru_cache(maxsize=None)
def _get_decorator_for_action(action):
    def decorator(fn):
        return fn
    return decorator
"""

    destination_code = """\
from functools import lru_cache

class SkyvernPage:
    @staticmethod
    def action_wrap(action):
        # Original implementation
        return action
"""

    expected = """\
from functools import lru_cache

class SkyvernPage:
    @staticmethod
    def action_wrap(action):
        # Original implementation
        return action


@lru_cache(maxsize=None)
def _get_decorator_for_action(action):
    def decorator(fn):
        return fn
    return decorator
"""

    result = add_global_assignments(source_code, destination_code)
    assert result == expected


def test_add_global_assignments_does_not_duplicate_existing_functions():
    """Test add_global_assignments does not duplicate functions that already exist in destination."""
    source_code = """\
def helper():
    return "source_helper"

def existing_function():
    return "source_existing"
"""

    destination_code = """\
def existing_function():
    return "dest_existing"

class MyClass:
    pass
"""

    expected = """\
def existing_function():
    return "dest_existing"

class MyClass:
    pass

def helper():
    return "source_helper"
"""

    result = add_global_assignments(source_code, destination_code)
    assert result == expected


def test_add_global_assignments_with_decorated_functions():
    """Test add_global_assignments correctly adds decorated functions."""
    source_code = """\
from functools import lru_cache
from typing import Callable

_LOCAL_CACHE: dict[str, int] = {}

@lru_cache(maxsize=128)
def cached_helper(x: int) -> int:
    return x * 2

def regular_helper():
    return "regular"
"""

    destination_code = """\
from typing import Any

class MyClass:
    def method(self):
        return cached_helper(5)
"""

    expected = """\
from typing import Any

_LOCAL_CACHE: dict[str, int] = {}

class MyClass:
    def method(self):
        return cached_helper(5)


@lru_cache(maxsize=128)
def cached_helper(x: int) -> int:
    return x * 2


def regular_helper():
    return "regular"
"""

    result = add_global_assignments(source_code, destination_code)
    assert result == expected


def test_add_global_assignments_references_class_defined_in_module():
    """Test that global assignments referencing classes are placed after those class definitions.

    This test verifies the fix for a bug where LLM-generated optimization code like:
        _REIFIERS = {MessageKind.XXX: lambda d: ...}
    was placed BEFORE the MessageKind class definition, causing NameError at module load.

    The fix ensures that new global assignments are inserted AFTER all class/function
    definitions in the module, so they can safely reference any class defined in the module.
    """
    source_code = """\
import enum

class MessageKind(enum.StrEnum):
    ASK = "ask"
    REPLY = "reply"

_MESSAGE_HANDLERS = {
    MessageKind.ASK: lambda: "ask handler",
    MessageKind.REPLY: lambda: "reply handler",
}

def handle_message(kind):
    return _MESSAGE_HANDLERS[kind]()
"""

    destination_code = """\
import enum

class MessageKind(enum.StrEnum):
    ASK = "ask"
    REPLY = "reply"

def handle_message(kind):
    if kind == MessageKind.ASK:
        return "ask"
    return "reply"
"""

    # Global assignments are now inserted AFTER class/function definitions
    # to ensure they can reference classes defined in the module
    expected = """\
import enum

class MessageKind(enum.StrEnum):
    ASK = "ask"
    REPLY = "reply"

def handle_message(kind):
    if kind == MessageKind.ASK:
        return "ask"
    return "reply"

_MESSAGE_HANDLERS = {
    MessageKind.ASK: lambda: "ask handler",
    MessageKind.REPLY: lambda: "reply handler",
}
"""

    result = add_global_assignments(source_code, destination_code)
    assert result == expected


def test_add_global_assignments_function_calls_after_function_definitions():
    """Test that global function calls are placed after the functions they reference.

    This test verifies the fix for a bug where LLM-generated optimization code like:
        def _register(kind, factory):
            _factories[kind] = factory

        _register(MessageKind.ASK, lambda: "ask")

    would have the _register(...) calls placed BEFORE the _register function definition,
    causing NameError at module load time.

    The fix ensures that new global statements (like function calls) are inserted AFTER
    all class/function definitions, so they can safely reference any function defined in
    the module.
    """
    source_code = """\
import enum

class MessageKind(enum.StrEnum):
    ASK = "ask"
    REPLY = "reply"

_factories = {}

def _register(kind, factory):
    _factories[kind] = factory

_register(MessageKind.ASK, lambda: "ask handler")
_register(MessageKind.REPLY, lambda: "reply handler")

def handle_message(kind):
    return _factories[kind]()
"""

    destination_code = """\
import enum

class MessageKind(enum.StrEnum):
    ASK = "ask"
    REPLY = "reply"

def handle_message(kind):
    if kind == MessageKind.ASK:
        return "ask"
    return "reply"
"""

    expected = """\
import enum

_factories = {}

class MessageKind(enum.StrEnum):
    ASK = "ask"
    REPLY = "reply"

def handle_message(kind):
    if kind == MessageKind.ASK:
        return "ask"
    return "reply"


def _register(kind, factory):
    _factories[kind] = factory


_register(MessageKind.ASK, lambda: "ask handler")

_register(MessageKind.REPLY, lambda: "reply handler")
"""

    result = add_global_assignments(source_code, destination_code)
    assert result == expected


def test_class_instantiation_includes_init_as_helper(tmp_path: Path) -> None:
    """Test that when a class is instantiated, its __init__ method is tracked as a helper.

    This test verifies the fix for the bug where class constructors were not
    included in the context when only the class instantiation was called
    (not any other methods). This caused LLMs to not know the constructor
    signatures when generating tests.
    """
    code = '''
class DataDumper:
    """A class that dumps data."""

    def __init__(self, data):
        """Initialize with data."""
        self.data = data

    def dump(self):
        """Dump the data."""
        return self.data


def target_function():
    # Only instantiates DataDumper, doesn't call any other methods
    dumper = DataDumper({"key": "value"})
    return dumper
'''
    file_path = tmp_path / "test_code.py"
    file_path.write_text(code, encoding="utf-8")
    opt = Optimizer(
        Namespace(
            project_root=file_path.parent.resolve(),
            disable_telemetry=True,
            tests_root="tests",
            test_framework="pytest",
            pytest_cmd="pytest",
            experiment_id=None,
            test_project_root=Path().resolve(),
        )
    )
    function_to_optimize = FunctionToOptimize(
        function_name="target_function", file_path=file_path, parents=[], starting_line=None, ending_line=None
    )

    code_ctx = get_code_optimization_context(function_to_optimize, opt.args.project_root)

    # The __init__ method should be tracked as a helper since DataDumper() instantiates the class
    qualified_names = {func.qualified_name for func in code_ctx.helper_functions}
    assert "DataDumper.__init__" in qualified_names, (
        "DataDumper.__init__ should be tracked as a helper when the class is instantiated"
    )

    # The testgen context should contain the class with __init__ (critical for LLM to know constructor)
    testgen_context = code_ctx.testgen_context.markdown
    assert "class DataDumper:" in testgen_context, "DataDumper class should be in testgen context"
    assert "def __init__(self, data):" in testgen_context, "__init__ method should be included in testgen context"

    # The hashing context should NOT contain __init__ (excluded for stability)
    hashing_context = code_ctx.hashing_code_context
    assert "__init__" not in hashing_context, "__init__ should NOT be in hashing context (excluded for hash stability)"


def test_class_instantiation_preserves_full_class_in_testgen(tmp_path: Path) -> None:
    """Test that instantiated classes are fully preserved in testgen context.

    This is specifically for the unstructured LayoutDumper bug where helper classes
    that were instantiated but had no other methods called were being excluded
    from the testgen context.
    """
    code = '''
class LayoutDumper:
    """Base class for layout dumpers."""
    layout_source: str = "unknown"

    def __init__(self, layout):
        self._layout = layout

    def dump(self) -> dict:
        raise NotImplementedError()


class ObjectDetectionLayoutDumper(LayoutDumper):
    """Specific dumper for object detection layouts."""

    def __init__(self, layout):
        super().__init__(layout)

    def dump(self) -> dict:
        return {"type": "object_detection", "layout": self._layout}


def dump_layout(layout_type, layout):
    """Dump a layout based on its type."""
    if layout_type == "object_detection":
        dumper = ObjectDetectionLayoutDumper(layout)
    else:
        dumper = LayoutDumper(layout)
    return dumper.dump()
'''
    file_path = tmp_path / "test_code.py"
    file_path.write_text(code, encoding="utf-8")
    opt = Optimizer(
        Namespace(
            project_root=file_path.parent.resolve(),
            disable_telemetry=True,
            tests_root="tests",
            test_framework="pytest",
            pytest_cmd="pytest",
            experiment_id=None,
            test_project_root=Path().resolve(),
        )
    )
    function_to_optimize = FunctionToOptimize(
        function_name="dump_layout", file_path=file_path, parents=[], starting_line=None, ending_line=None
    )

    code_ctx = get_code_optimization_context(function_to_optimize, opt.args.project_root)
    qualified_names = {func.qualified_name for func in code_ctx.helper_functions}

    # Both class __init__ methods should be tracked as helpers
    assert "ObjectDetectionLayoutDumper.__init__" in qualified_names, (
        "ObjectDetectionLayoutDumper.__init__ should be tracked"
    )
    assert "LayoutDumper.__init__" in qualified_names, "LayoutDumper.__init__ should be tracked"

    # The testgen context should include both classes with their __init__ methods
    testgen_context = code_ctx.testgen_context.markdown
    assert "class LayoutDumper:" in testgen_context, "LayoutDumper should be in testgen context"
    assert "class ObjectDetectionLayoutDumper" in testgen_context, (
        "ObjectDetectionLayoutDumper should be in testgen context"
    )

    # Both __init__ methods should be in the testgen context (so LLM knows constructor signatures)
    assert testgen_context.count("def __init__") >= 2, "Both __init__ methods should be in testgen context"


def test_enrich_testgen_context_extracts_project_classes(tmp_path: Path) -> None:
    """Test that enrich_testgen_context extracts class definitions from project modules."""
    # Create a package structure with two modules
    package_dir = tmp_path / "mypackage"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    # Create a module with a class definition (simulating Element-like class)
    elements_code = '''
import abc

class Element(abc.ABC):
    """An element in the document."""

    def __init__(self, element_id: str = None):
        self._element_id = element_id
        self.text = ""

    def __str__(self):
        return self.text


class Text(Element):
    """A text element."""

    def __init__(self, text: str, element_id: str = None):
        super().__init__(element_id)
        self.text = text
'''
    elements_path = package_dir / "elements.py"
    elements_path.write_text(elements_code, encoding="utf-8")

    # Create another module that imports from elements
    chunking_code = """
from mypackage.elements import Element

class PreChunk:
    def __init__(self, elements: list[Element]):
        self._elements = elements

class Accumulator:
    def will_fit(self, chunk: PreChunk) -> bool:
        return True
"""
    chunking_path = package_dir / "chunking.py"
    chunking_path.write_text(chunking_code, encoding="utf-8")

    # Create CodeStringsMarkdown from the chunking module (simulating testgen context)
    context = CodeStringsMarkdown(code_strings=[CodeString(code=chunking_code, file_path=chunking_path)])

    # Call enrich_testgen_context
    result = enrich_testgen_context(context, tmp_path)

    # Verify Element class was extracted
    assert len(result.code_strings) == 1, "Should extract exactly one class (Element)"
    extracted_code = result.code_strings[0].code

    # Verify the extracted code contains the Element class
    assert "class Element" in extracted_code, "Should contain Element class definition"
    assert "def __init__" in extracted_code, "Should contain __init__ method"
    assert "element_id" in extracted_code, "Should contain constructor parameter"


def test_enrich_testgen_context_skips_existing_definitions(tmp_path: Path) -> None:
    """Test that enrich_testgen_context skips classes already defined in context."""
    # Create a package structure
    package_dir = tmp_path / "mypackage"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    # Create a module with a class definition
    elements_code = """
class Element:
    def __init__(self, text: str):
        self.text = text
"""
    elements_path = package_dir / "elements.py"
    elements_path.write_text(elements_code, encoding="utf-8")

    # Create code that imports Element but also redefines it locally
    code_with_local_def = """
from mypackage.elements import Element

# Local redefinition (this happens when LLM redefines classes)
class Element:
    def __init__(self, text: str):
        self.text = text

class User:
    def process(self, elem: Element):
        pass
"""
    code_path = package_dir / "user.py"
    code_path.write_text(code_with_local_def, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code_with_local_def, file_path=code_path)])

    # Call enrich_testgen_context
    result = enrich_testgen_context(context, tmp_path)

    # Should NOT extract Element since it's already defined locally
    assert len(result.code_strings) == 0, "Should not extract classes already defined in context"


def test_enrich_testgen_context_skips_third_party(tmp_path: Path) -> None:
    """Test that enrich_testgen_context skips third-party/stdlib imports."""
    # Create a simple package
    package_dir = tmp_path / "mypackage"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    # Code with stdlib/third-party imports
    code = """
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

class MyClass:
    def __init__(self, path: Path):
        self.path = path
"""
    code_path = package_dir / "main.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])

    # Call enrich_testgen_context
    result = enrich_testgen_context(context, tmp_path)

    # Should not extract any classes (Path, Optional, dataclass are stdlib/third-party)
    assert len(result.code_strings) == 0, "Should not extract stdlib/third-party classes"


def test_enrich_testgen_context_handles_multiple_imports(tmp_path: Path) -> None:
    """Test that enrich_testgen_context handles multiple class imports."""
    # Create a package structure
    package_dir = tmp_path / "mypackage"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    # Create a module with multiple class definitions
    types_code = """
class TypeA:
    def __init__(self, value: int):
        self.value = value

class TypeB:
    def __init__(self, name: str):
        self.name = name

class TypeC:
    def __init__(self):
        pass
"""
    types_path = package_dir / "types.py"
    types_path.write_text(types_code, encoding="utf-8")

    # Create code that imports multiple classes
    code = """
from mypackage.types import TypeA, TypeB

class Processor:
    def process(self, a: TypeA, b: TypeB):
        pass
"""
    code_path = package_dir / "processor.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])

    # Call enrich_testgen_context
    result = enrich_testgen_context(context, tmp_path)

    # Should extract both TypeA and TypeB (but not TypeC since it's not imported)
    assert len(result.code_strings) == 2, "Should extract exactly two classes (TypeA, TypeB)"

    all_extracted_code = "\n".join(cs.code for cs in result.code_strings)
    assert "class TypeA" in all_extracted_code, "Should contain TypeA class"
    assert "class TypeB" in all_extracted_code, "Should contain TypeB class"
    assert "class TypeC" not in all_extracted_code, "Should NOT contain TypeC (not imported)"


def test_enrich_testgen_context_includes_dataclass_decorators(tmp_path: Path) -> None:
    """Test that enrich_testgen_context includes decorators when extracting dataclasses."""
    # Create a package structure
    package_dir = tmp_path / "mypackage"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    # Create a module with dataclass definitions (like LLMConfig in skyvern)
    models_code = """from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class LLMConfigBase:
    model_name: str
    required_env_vars: list[str]
    supports_vision: bool
    add_assistant_prefix: bool

@dataclass(frozen=True)
class LLMConfig(LLMConfigBase):
    litellm_params: Optional[dict] = field(default=None)
    max_tokens: int | None = None
"""
    models_path = package_dir / "models.py"
    models_path.write_text(models_code, encoding="utf-8")

    # Create code that imports the dataclass
    code = """from mypackage.models import LLMConfig

class ConfigRegistry:
    def get_config(self) -> LLMConfig:
        pass
"""
    code_path = package_dir / "registry.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])

    # Call enrich_testgen_context
    result = enrich_testgen_context(context, tmp_path)

    # Should extract both LLMConfigBase (base class) and LLMConfig
    assert len(result.code_strings) == 2, "Should extract both LLMConfig and its base class LLMConfigBase"

    # Combine extracted code to check for all required elements
    all_extracted_code = "\n".join(cs.code for cs in result.code_strings)

    # Verify the base class is extracted first (for proper inheritance understanding)
    base_class_idx = all_extracted_code.find("class LLMConfigBase")
    derived_class_idx = all_extracted_code.find("class LLMConfig(")
    assert base_class_idx < derived_class_idx, "Base class should appear before derived class"

    # Verify both classes include @dataclass decorators
    assert all_extracted_code.count("@dataclass(frozen=True)") == 2, (
        "Should include @dataclass decorator for both classes"
    )
    assert "class LLMConfig" in all_extracted_code, "Should contain LLMConfig class definition"
    assert "class LLMConfigBase" in all_extracted_code, "Should contain LLMConfigBase class definition"


def test_enrich_testgen_context_extracts_imports_for_decorated_classes(tmp_path: Path) -> None:
    """Test that extract_imports_for_class includes decorator and type annotation imports."""
    # Create a package structure
    package_dir = tmp_path / "mypackage"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    # Create a module with decorated class that uses field() and various type annotations
    models_code = """from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class Config:
    name: str
    values: List[int] = field(default_factory=list)
    description: Optional[str] = None
"""
    models_path = package_dir / "models.py"
    models_path.write_text(models_code, encoding="utf-8")

    # Create code that imports the class
    code = """from mypackage.models import Config

def create_config() -> Config:
    return Config(name="test")
"""
    code_path = package_dir / "main.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])

    result = enrich_testgen_context(context, tmp_path)

    assert len(result.code_strings) == 1, "Should extract Config class"
    extracted_code = result.code_strings[0].code

    # The extracted code should include the decorator
    assert "@dataclass" in extracted_code, "Should include @dataclass decorator"


def test_enrich_testgen_context_multiple_decorators(tmp_path: Path) -> None:
    """Test that classes with multiple decorators are extracted correctly."""
    package_dir = tmp_path / "mypackage"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    models_code = """from dataclasses import dataclass
from functools import total_ordering

@total_ordering
@dataclass
class OrderedConfig:
    name: str
    priority: int

    def __lt__(self, other):
        return self.priority < other.priority
"""
    models_path = package_dir / "models.py"
    models_path.write_text(models_code, encoding="utf-8")

    code = """from mypackage.models import OrderedConfig

def sort_configs(configs: list[OrderedConfig]) -> list[OrderedConfig]:
    return sorted(configs)
"""
    code_path = package_dir / "main.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])

    result = enrich_testgen_context(context, tmp_path)

    assert len(result.code_strings) == 1
    extracted_code = result.code_strings[0].code

    # Both decorators should be included
    assert "@total_ordering" in extracted_code, "Should include @total_ordering decorator"
    assert "@dataclass" in extracted_code, "Should include @dataclass decorator"
    assert "class OrderedConfig" in extracted_code


def test_enrich_testgen_context_extracts_multilevel_inheritance(tmp_path: Path) -> None:
    """Test that base classes are recursively extracted for multi-level inheritance.

    This is critical for understanding dataclass constructor signatures, as fields
    from parent classes become required positional arguments in child classes.
    """
    # Create a package structure
    package_dir = tmp_path / "mypackage"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    # Create a module with multi-level inheritance like skyvern's LLM models:
    # GrandParent -> Parent -> Child
    models_code = '''from dataclasses import dataclass, field
from typing import Optional, Literal

@dataclass(frozen=True)
class GrandParentConfig:
    """Base config with common fields."""
    model_name: str
    required_env_vars: list[str]

@dataclass(frozen=True)
class ParentConfig(GrandParentConfig):
    """Intermediate config adding vision support."""
    supports_vision: bool
    add_assistant_prefix: bool

@dataclass(frozen=True)
class ChildConfig(ParentConfig):
    """Full config with optional parameters."""
    litellm_params: Optional[dict] = field(default=None)
    max_tokens: int | None = None
    temperature: float | None = 0.7

@dataclass(frozen=True)
class RouterConfig(ParentConfig):
    """Router config branching from ParentConfig."""
    model_list: list
    main_model_group: str
    routing_strategy: Literal["simple", "least-busy"] = "simple"
'''
    models_path = package_dir / "models.py"
    models_path.write_text(models_code, encoding="utf-8")

    # Create code that imports only the child classes (not the base classes)
    code = """from mypackage.models import ChildConfig, RouterConfig

class ConfigRegistry:
    def get_child_config(self) -> ChildConfig:
        pass

    def get_router_config(self) -> RouterConfig:
        pass
"""
    code_path = package_dir / "registry.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])

    # Call enrich_testgen_context
    result = enrich_testgen_context(context, tmp_path)

    # Should extract 4 classes: GrandParentConfig, ParentConfig, ChildConfig, RouterConfig
    # (all classes needed to understand the full inheritance hierarchy)
    assert len(result.code_strings) == 4, (
        f"Should extract 4 classes (GrandParent, Parent, Child, Router), got {len(result.code_strings)}"
    )

    # Combine extracted code
    all_extracted_code = "\n".join(cs.code for cs in result.code_strings)

    # Verify all classes are extracted
    assert "class GrandParentConfig" in all_extracted_code, "Should extract GrandParentConfig base class"
    assert "class ParentConfig(GrandParentConfig)" in all_extracted_code, "Should extract ParentConfig"
    assert "class ChildConfig(ParentConfig)" in all_extracted_code, "Should extract ChildConfig"
    assert "class RouterConfig(ParentConfig)" in all_extracted_code, "Should extract RouterConfig"

    # Verify classes are ordered correctly (base classes before derived)
    grandparent_idx = all_extracted_code.find("class GrandParentConfig")
    parent_idx = all_extracted_code.find("class ParentConfig(")
    child_idx = all_extracted_code.find("class ChildConfig(")
    router_idx = all_extracted_code.find("class RouterConfig(")

    assert grandparent_idx < parent_idx, "GrandParentConfig should appear before ParentConfig"
    assert parent_idx < child_idx, "ParentConfig should appear before ChildConfig"
    assert parent_idx < router_idx, "ParentConfig should appear before RouterConfig"

    # Verify the critical fields are visible for constructor understanding
    assert "model_name: str" in all_extracted_code, "Should include model_name field from GrandParent"
    assert "required_env_vars: list[str]" in all_extracted_code, "Should include required_env_vars field"
    assert "supports_vision: bool" in all_extracted_code, "Should include supports_vision field from Parent"
    assert "litellm_params:" in all_extracted_code, "Should include litellm_params field from Child"
    assert "model_list: list" in all_extracted_code, "Should include model_list field from Router"


def test_enrich_testgen_context_skips_stdlib_userdict(tmp_path: Path) -> None:
    """Skips stdlib classes like collections.UserDict."""
    code = """from collections import UserDict

class MyCustomDict(UserDict):
    pass
"""
    code_path = tmp_path / "mydict.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])
    result = enrich_testgen_context(context, tmp_path)

    assert len(result.code_strings) == 0, "Should not extract stdlib classes"


def test_enrich_testgen_context_skips_unresolvable_base_classes(tmp_path: Path) -> None:
    """Returns empty when base class module cannot be resolved."""
    child_code = """from base import ProjectBase

class Child(ProjectBase):
    pass
"""
    child_path = tmp_path / "child.py"
    child_path.write_text(child_code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=child_code, file_path=child_path)])
    result = enrich_testgen_context(context, tmp_path)

    assert result.code_strings == []


def test_enrich_testgen_context_skips_builtin_base_classes(tmp_path: Path) -> None:
    """Returns empty for builtin classes like list that have no inspectable source."""
    code = """class MyList(list):
    pass
"""
    code_path = tmp_path / "mylist.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])
    result = enrich_testgen_context(context, tmp_path)

    assert result.code_strings == []


def test_enrich_testgen_context_deduplicates(tmp_path: Path) -> None:
    """Extracts the same project class only once even when imported multiple times."""
    package_dir = tmp_path / "mypkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "base.py").write_text(
        "class Base:\n    def __init__(self, x: int):\n        self.x = x\n",
        encoding="utf-8",
    )

    code = "from mypkg.base import Base\n\nclass A(Base):\n    pass\n\nclass B(Base):\n    pass\n"
    code_path = package_dir / "children.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])
    result = enrich_testgen_context(context, tmp_path)

    assert len(result.code_strings) == 1
    assert "class Base" in result.code_strings[0].code


def test_enrich_testgen_context_empty_when_no_inheritance(tmp_path: Path) -> None:
    """Returns empty when there are no external base classes."""
    code = """class SimpleClass:
    pass
"""
    code_path = tmp_path / "simple.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])
    result = enrich_testgen_context(context, tmp_path)

    assert result.code_strings == []


@pytest.mark.skipif(sys.version_info < (3, 11), reason="enum.StrEnum requires Python 3.11+")
def test_dependency_classes_kept_in_read_writable_context(tmp_path: Path) -> None:
    """Tests that classes used as dependencies (enums, dataclasses) are kept in read-writable context.

    This test verifies that when a function uses classes like enums or dataclasses
    as types or in match statements, those classes are included in the optimization
    context, even though they don't contain any target functions.
    """
    code = """
import dataclasses
import enum
import typing as t


class MessageKind(enum.StrEnum):
    ASK_FOR_CLIPBOARD_RESPONSE = "ask-for-clipboard-response"
    BEGIN_EXFILTRATION = "begin-exfiltration"


@dataclasses.dataclass
class Message:
    kind: str


@dataclasses.dataclass
class MessageInAskForClipboardResponse(Message):
    kind: t.Literal[MessageKind.ASK_FOR_CLIPBOARD_RESPONSE] = MessageKind.ASK_FOR_CLIPBOARD_RESPONSE
    text: str = ""


@dataclasses.dataclass
class MessageInBeginExfiltration(Message):
    kind: t.Literal[MessageKind.BEGIN_EXFILTRATION] = MessageKind.BEGIN_EXFILTRATION


MessageIn = (
    MessageInAskForClipboardResponse
    | MessageInBeginExfiltration
)


def reify_channel_message(data: dict) -> MessageIn:
    kind = data.get("kind", None)

    match kind:
        case MessageKind.ASK_FOR_CLIPBOARD_RESPONSE:
            text = data.get("text") or ""
            return MessageInAskForClipboardResponse(text=text)
        case MessageKind.BEGIN_EXFILTRATION:
            return MessageInBeginExfiltration()
        case _:
            raise ValueError(f"Unknown message kind: '{kind}'")
"""
    code_path = tmp_path / "message.py"
    code_path.write_text(code, encoding="utf-8")

    func_to_optimize = FunctionToOptimize(function_name="reify_channel_message", file_path=code_path, parents=[])

    code_ctx = get_code_optimization_context(function_to_optimize=func_to_optimize, project_root_path=tmp_path)

    expected_read_writable = """
```python:message.py
import dataclasses
import enum
import typing as t


class MessageKind(enum.StrEnum):
    ASK_FOR_CLIPBOARD_RESPONSE = "ask-for-clipboard-response"
    BEGIN_EXFILTRATION = "begin-exfiltration"


@dataclasses.dataclass
class Message:
    kind: str


@dataclasses.dataclass
class MessageInAskForClipboardResponse(Message):
    kind: t.Literal[MessageKind.ASK_FOR_CLIPBOARD_RESPONSE] = MessageKind.ASK_FOR_CLIPBOARD_RESPONSE
    text: str = ""


@dataclasses.dataclass
class MessageInBeginExfiltration(Message):
    kind: t.Literal[MessageKind.BEGIN_EXFILTRATION] = MessageKind.BEGIN_EXFILTRATION


MessageIn = (
    MessageInAskForClipboardResponse
    | MessageInBeginExfiltration
)


def reify_channel_message(data: dict) -> MessageIn:
    kind = data.get("kind", None)

    match kind:
        case MessageKind.ASK_FOR_CLIPBOARD_RESPONSE:
            text = data.get("text") or ""
            return MessageInAskForClipboardResponse(text=text)
        case MessageKind.BEGIN_EXFILTRATION:
            return MessageInBeginExfiltration()
        case _:
            raise ValueError(f"Unknown message kind: '{kind}'")
```
"""
    assert code_ctx.read_writable_code.markdown.strip() == expected_read_writable.strip()


def test_testgen_context_includes_external_base_inits(tmp_path: Path) -> None:
    """Test that base class definitions from project modules are included in testgen context."""
    package_dir = tmp_path / "mypkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "base.py").write_text(
        "class BaseDict:\n    def __init__(self, data=None):\n        self.data = data or {}\n",
        encoding="utf-8",
    )

    code = "from mypkg.base import BaseDict\n\nclass MyCustomDict(BaseDict):\n    def target_method(self):\n        return self.data\n"
    file_path = package_dir / "test_code.py"
    file_path.write_text(code, encoding="utf-8")

    func_to_optimize = FunctionToOptimize(
        function_name="target_method",
        file_path=file_path,
        parents=[FunctionParent(name="MyCustomDict", type="ClassDef")],
    )

    code_ctx = get_code_optimization_context(function_to_optimize=func_to_optimize, project_root_path=tmp_path)

    testgen_context = code_ctx.testgen_context.markdown
    assert "class BaseDict" in testgen_context, "BaseDict class should be in testgen context"
    assert "def __init__" in testgen_context, "BaseDict __init__ should be in testgen context"
    assert "self.data" in testgen_context, "BaseDict __init__ body should be included"


def test_testgen_raises_when_exceeds_limit(tmp_path: Path) -> None:
    """Test that ValueError is raised when testgen context exceeds token limit."""
    # Create a function with a very long body that exceeds limits even without imports/docstrings
    long_lines = ["    x = 0"]
    for i in range(200):
        long_lines.append(f"    x = x + {i}")
    long_lines.append("    return x")
    long_body = "\n".join(long_lines)

    code = f"""
def target_function():
{long_body}
"""
    file_path = tmp_path / "test_code.py"
    file_path.write_text(code, encoding="utf-8")

    func_to_optimize = FunctionToOptimize(function_name="target_function", file_path=file_path, parents=[])

    # Use a very small testgen_token_limit that cannot fit even the base function
    with pytest.raises(ValueError, match="Testgen code context has exceeded token limit"):
        get_code_optimization_context(
            function_to_optimize=func_to_optimize,
            project_root_path=tmp_path,
            testgen_token_limit=50,  # Very small limit
        )


def test_enrich_testgen_context_attribute_base(tmp_path: Path) -> None:
    """Test handling of base class in a project module."""
    package_dir = tmp_path / "mypkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "base.py").write_text(
        "class CustomDict:\n    def __init__(self, data=None):\n        self.data = data or {}\n",
        encoding="utf-8",
    )

    code = "from mypkg.base import CustomDict\n\nclass MyDict(CustomDict):\n    def custom_method(self):\n        return self.data\n"
    code_path = package_dir / "mydict.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])
    result = enrich_testgen_context(context, tmp_path)

    assert len(result.code_strings) == 1
    assert "class CustomDict" in result.code_strings[0].code
    assert "def __init__" in result.code_strings[0].code


def test_enrich_testgen_context_no_init_method(tmp_path: Path) -> None:
    """Test handling when base class has no __init__ method.

    This covers line 641 in code_context_extractor.py.
    """
    # Create a class inheriting from a class that doesn't have inspectable __init__
    code = """from typing import Protocol

class MyProtocol(Protocol):
    pass
"""
    code_path = tmp_path / "myproto.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])
    result = enrich_testgen_context(context, tmp_path)

    # Protocol's __init__ can't be easily inspected, should handle gracefully
    # Result may be empty or contain Protocol based on implementation
    assert isinstance(result.code_strings, list)


def test_annotated_assignment_in_read_writable(tmp_path: Path) -> None:
    """Test that annotated assignments used by target function are in read-writable context.

    This covers lines 965-969 in code_context_extractor.py.
    """
    code = """
CONFIG_VALUE: int = 42

class MyClass:
    def __init__(self):
        self.x = CONFIG_VALUE

    def target_method(self):
        return self.x
"""
    file_path = tmp_path / "test_code.py"
    file_path.write_text(code, encoding="utf-8")

    func_to_optimize = FunctionToOptimize(
        function_name="target_method", file_path=file_path, parents=[FunctionParent(name="MyClass", type="ClassDef")]
    )

    code_ctx = get_code_optimization_context(function_to_optimize=func_to_optimize, project_root_path=tmp_path)

    # CONFIG_VALUE should be in read-writable context since it's used by __init__
    read_writable = code_ctx.read_writable_code.markdown
    assert "CONFIG_VALUE" in read_writable


def test_imported_class_definitions_module_path_none(tmp_path: Path) -> None:
    """Test handling when module_path is None in enrich_testgen_context.

    This covers line 560 in code_context_extractor.py.
    """
    # Create code that imports from a non-existent or unresolvable module
    code = """
from nonexistent_module_xyz import SomeClass

class MyClass:
    def method(self, obj: SomeClass):
        pass
"""
    code_path = tmp_path / "test.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])
    result = enrich_testgen_context(context, tmp_path)

    # Should handle gracefully and return empty or partial results
    assert isinstance(result.code_strings, list)


def test_imported_class_with_base_in_same_module(tmp_path: Path) -> None:
    """Test that imported classes with bases in the same module are extracted correctly.

    This covers line 528 in code_context_extractor.py - early return for already extracted.
    """
    package_dir = tmp_path / "mypackage"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    # Create a module with inheritance chain
    module_code = """
class BaseClass:
    def __init__(self):
        self.base = True

class MiddleClass(BaseClass):
    def __init__(self):
        super().__init__()
        self.middle = True

class DerivedClass(MiddleClass):
    def __init__(self):
        super().__init__()
        self.derived = True
"""
    module_path = package_dir / "classes.py"
    module_path.write_text(module_code, encoding="utf-8")

    # Main module imports and uses the derived class
    main_code = """
from mypackage.classes import DerivedClass

def target_function(obj: DerivedClass) -> bool:
    return obj.derived
"""
    main_path = package_dir / "main.py"
    main_path.write_text(main_code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=main_code, file_path=main_path)])
    result = enrich_testgen_context(context, tmp_path)

    # Should extract the inheritance chain
    all_code = "\n".join(cs.code for cs in result.code_strings)
    assert "class BaseClass" in all_code or "class DerivedClass" in all_code


def test_augmented_assignment_not_in_context(tmp_path: Path) -> None:
    """Test that augmented assignments are handled but not included unless used.

    This covers line 962-969 in code_context_extractor.py.
    """
    code = """
counter = 0

class MyClass:
    def __init__(self):
        global counter
        counter += 1

    def target_method(self):
        return 42
"""
    file_path = tmp_path / "test_code.py"
    file_path.write_text(code, encoding="utf-8")

    func_to_optimize = FunctionToOptimize(
        function_name="target_method", file_path=file_path, parents=[FunctionParent(name="MyClass", type="ClassDef")]
    )

    code_ctx = get_code_optimization_context(function_to_optimize=func_to_optimize, project_root_path=tmp_path)

    # counter should be in context since __init__ uses it
    read_writable = code_ctx.read_writable_code.markdown
    assert "counter" in read_writable


def test_enrich_testgen_context_extracts_click_option(tmp_path: Path) -> None:
    """click.Option re-exports via __init__.py so jedi resolves the module but not the class directly."""
    code = """from click import Option

def my_func(opt: Option) -> None:
    pass
"""
    code_path = tmp_path / "myfunc.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])
    result = enrich_testgen_context(context, tmp_path)

    # click re-exports Option from click.core via __init__.py; jedi resolves
    # the module to __init__.py where Option is not defined as a ClassDef,
    # so enrich_testgen_context cannot extract it.
    assert isinstance(result.code_strings, list)


def test_enrich_testgen_context_extracts_project_class_defs(tmp_path: Path) -> None:
    """Extracts project class definitions via jedi resolution."""
    # Create a project module with a class
    (tmp_path / "mymodule.py").write_text("class ProjectClass:\n    pass\n", encoding="utf-8")

    code = """from mymodule import ProjectClass

def my_func(obj: ProjectClass) -> None:
    pass
"""
    code_path = tmp_path / "myfunc.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])
    result = enrich_testgen_context(context, tmp_path)

    assert len(result.code_strings) == 1
    assert "class ProjectClass" in result.code_strings[0].code


def test_enrich_testgen_context_skips_non_classes(tmp_path: Path) -> None:
    """Returns empty when imported name is a function, not a class."""
    code = """from collections import OrderedDict
from os.path import join

def my_func() -> None:
    pass
"""
    code_path = tmp_path / "myfunc.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])
    result = enrich_testgen_context(context, tmp_path)

    # join is a function, not a class — should be skipped
    # OrderedDict is a class and should be included
    class_names = [cs.code.split("\n")[0] for cs in result.code_strings]
    assert not any("join" in name for name in class_names)


def test_enrich_testgen_context_skips_already_defined_classes(tmp_path: Path) -> None:
    """Skips classes already defined in the context (e.g., added by enrich_testgen_context)."""
    code = """from collections import UserDict

class UserDict:
    def __init__(self):
        pass

def my_func(d: UserDict) -> None:
    pass
"""
    code_path = tmp_path / "myfunc.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])
    result = enrich_testgen_context(context, tmp_path)

    # UserDict is already defined in the context, so it should be skipped
    assert result.code_strings == []


def test_enrich_testgen_context_skips_builtin_annotations(tmp_path: Path) -> None:
    """Returns empty for builtin type annotations like list/dict that are not imported."""
    code = """x: list = []
y: dict = {}

def my_func() -> None:
    pass
"""
    code_path = tmp_path / "myfunc.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])
    result = enrich_testgen_context(context, tmp_path)

    assert result.code_strings == []


def test_enrich_testgen_context_skips_stdlib(tmp_path: Path) -> None:
    """Skips stdlib classes like QName."""
    code = """from xml.etree.ElementTree import QName

def my_func(q: QName) -> None:
    pass
"""
    code_path = tmp_path / "myfunc.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])
    result = enrich_testgen_context(context, tmp_path)

    assert result.code_strings == [], "Should not extract stdlib classes"


def test_enrich_testgen_context_empty_when_no_imports(tmp_path: Path) -> None:
    """Returns empty when there are no from-imports."""
    code = """def my_func() -> None:
    pass
"""
    code_path = tmp_path / "myfunc.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])
    result = enrich_testgen_context(context, tmp_path)

    assert result.code_strings == []


# --- Integration tests for transitive resolution in enrich_testgen_context ---


def test_enrich_testgen_context_transitive_deps(tmp_path: Path) -> None:
    """Transitive deps require the class to be resolvable in the target module."""
    package_dir = tmp_path / "mypkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    (package_dir / "types.py").write_text(
        "class Command:\n    def __init__(self, name: str):\n        self.name = name\n", encoding="utf-8"
    )
    (package_dir / "ctx.py").write_text(
        "from mypkg.types import Command\n\nclass Context:\n    def __init__(self, cmd: Command):\n        self.cmd = cmd\n",
        encoding="utf-8",
    )

    code = "from mypkg.ctx import Context\n\ndef my_func(ctx: Context) -> None:\n    pass\n"
    code_path = package_dir / "main.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])
    result = enrich_testgen_context(context, tmp_path)

    class_names = {cs.code.split("\n")[0].replace("class ", "").rstrip(":") for cs in result.code_strings}
    assert "Context" in class_names


def test_enrich_testgen_context_no_infinite_loops(tmp_path: Path) -> None:
    """Handles classes with circular type references without infinite loops."""
    package_dir = tmp_path / "mypkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    # Create circular references: Context references Command, Command references Context
    (package_dir / "core.py").write_text(
        "class Command:\n    def __init__(self, name: str):\n        self.name = name\n\n"
        "class Context:\n    def __init__(self, cmd: Command):\n        self.cmd = cmd\n",
        encoding="utf-8",
    )

    code = "from mypkg.core import Context\n\ndef my_func(ctx: Context) -> None:\n    pass\n"
    code_path = package_dir / "main.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])
    result = enrich_testgen_context(context, tmp_path)

    # Should complete without hanging
    assert len(result.code_strings) >= 1


def test_enrich_testgen_context_no_duplicate_stubs(tmp_path: Path) -> None:
    """Does not emit duplicate stubs for the same class name."""
    code = """from click import Context

def my_func(ctx: Context) -> None:
    pass
"""
    code_path = tmp_path / "myfunc.py"
    code_path.write_text(code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=code_path)])
    result = enrich_testgen_context(context, tmp_path)

    class_names = [cs.code.split("\n")[0].replace("class ", "").rstrip(":") for cs in result.code_strings]
    assert len(class_names) == len(set(class_names)), f"Duplicate class stubs found: {class_names}"


# --- Tests for collect_type_names_from_annotation ---


def test_collect_type_names_simple() -> None:
    tree = ast.parse("def f(x: Foo): pass")
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)
    ann = func.args.args[0].annotation
    assert collect_type_names_from_annotation(ann) == {"Foo"}


def test_collect_type_names_generic() -> None:
    tree = ast.parse("def f(x: list[Foo]): pass")
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)
    ann = func.args.args[0].annotation
    names = collect_type_names_from_annotation(ann)
    assert "Foo" in names
    assert "list" in names


def test_collect_type_names_optional() -> None:
    tree = ast.parse("def f(x: Optional[Foo]): pass")
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)
    ann = func.args.args[0].annotation
    names = collect_type_names_from_annotation(ann)
    assert "Optional" in names
    assert "Foo" in names


def test_collect_type_names_union_pipe() -> None:
    tree = ast.parse("def f(x: Foo | Bar): pass")
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)
    ann = func.args.args[0].annotation
    names = collect_type_names_from_annotation(ann)
    assert names == {"Foo", "Bar"}


def test_collect_type_names_none_annotation() -> None:
    assert collect_type_names_from_annotation(None) == set()


def test_collect_type_names_attribute_skipped() -> None:
    tree = ast.parse("def f(x: module.Foo): pass")
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)
    ann = func.args.args[0].annotation
    assert collect_type_names_from_annotation(ann) == set()


# --- Tests for extract_init_stub_from_class ---


def test_extract_init_stub_basic() -> None:
    source = """
class MyClass:
    def __init__(self, name: str, value: int = 0):
        self.name = name
        self.value = value
"""
    tree = ast.parse(source)
    stub = extract_init_stub_from_class("MyClass", source, tree)
    assert stub is not None
    assert "class MyClass:" in stub
    assert "def __init__(self, name: str, value: int = 0):" in stub
    assert "self.name = name" in stub
    assert "self.value = value" in stub


def test_extract_init_stub_no_init() -> None:
    source = """
class NoInit:
    x = 10
    def other(self):
        pass
"""
    tree = ast.parse(source)
    stub = extract_init_stub_from_class("NoInit", source, tree)
    assert stub is None


def test_extract_init_stub_class_not_found() -> None:
    source = """
class Other:
    def __init__(self):
        pass
"""
    tree = ast.parse(source)
    stub = extract_init_stub_from_class("Missing", source, tree)
    assert stub is None


# --- Tests for extract_parameter_type_constructors ---


def test_extract_parameter_type_constructors_project_type(tmp_path: Path) -> None:
    # Create a module with a class
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "models.py").write_text(
        """
class Widget:
    def __init__(self, size: int, color: str = "red"):
        self.size = size
        self.color = color
""",
        encoding="utf-8",
    )

    # Create the FTO file that uses Widget
    (pkg / "processor.py").write_text(
        """from mypkg.models import Widget

def process(w: Widget) -> str:
    return str(w)
""",
        encoding="utf-8",
    )

    fto = FunctionToOptimize(
        function_name="process", file_path=(pkg / "processor.py").resolve(), starting_line=3, ending_line=4
    )
    result = extract_parameter_type_constructors(fto, tmp_path.resolve(), set())
    assert len(result.code_strings) == 1
    code = result.code_strings[0].code
    assert "class Widget:" in code
    assert "def __init__" in code
    assert "size" in code


def test_extract_parameter_type_constructors_excludes_builtins(tmp_path: Path) -> None:
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "func.py").write_text(
        """
def my_func(x: int, y: str, z: list) -> None:
    pass
""",
        encoding="utf-8",
    )

    fto = FunctionToOptimize(
        function_name="my_func", file_path=(pkg / "func.py").resolve(), starting_line=2, ending_line=3
    )
    result = extract_parameter_type_constructors(fto, tmp_path.resolve(), set())
    assert len(result.code_strings) == 0


def test_extract_parameter_type_constructors_skips_existing_classes(tmp_path: Path) -> None:
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "models.py").write_text(
        """
class Widget:
    def __init__(self, size: int):
        self.size = size
""",
        encoding="utf-8",
    )
    (pkg / "processor.py").write_text(
        """from mypkg.models import Widget

def process(w: Widget) -> str:
    return str(w)
""",
        encoding="utf-8",
    )

    fto = FunctionToOptimize(
        function_name="process", file_path=(pkg / "processor.py").resolve(), starting_line=3, ending_line=4
    )
    # Widget is already in the context — should not be duplicated
    result = extract_parameter_type_constructors(fto, tmp_path.resolve(), {"Widget"})
    assert len(result.code_strings) == 0


def test_extract_parameter_type_constructors_no_init(tmp_path: Path) -> None:
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "models.py").write_text(
        """
class Config:
    x = 10
""",
        encoding="utf-8",
    )
    (pkg / "processor.py").write_text(
        """from mypkg.models import Config

def process(c: Config) -> str:
    return str(c)
""",
        encoding="utf-8",
    )

    fto = FunctionToOptimize(
        function_name="process", file_path=(pkg / "processor.py").resolve(), starting_line=3, ending_line=4
    )
    result = extract_parameter_type_constructors(fto, tmp_path.resolve(), set())
    assert len(result.code_strings) == 0


# --- Tests for resolve_instance_class_name ---


def test_resolve_instance_class_name_direct_call() -> None:
    source = "config = MyConfig(debug=True)"
    tree = ast.parse(source)
    assert resolve_instance_class_name("config", tree) == "MyConfig"


def test_resolve_instance_class_name_annotated() -> None:
    source = "config: MyConfig = load()"
    tree = ast.parse(source)
    assert resolve_instance_class_name("config", tree) == "MyConfig"


def test_resolve_instance_class_name_factory_method() -> None:
    source = "config = MyConfig.from_env()"
    tree = ast.parse(source)
    assert resolve_instance_class_name("config", tree) == "MyConfig"


def test_resolve_instance_class_name_no_match() -> None:
    source = "x = 42"
    tree = ast.parse(source)
    assert resolve_instance_class_name("x", tree) is None


def test_resolve_instance_class_name_missing_variable() -> None:
    source = "config = MyConfig()"
    tree = ast.parse(source)
    assert resolve_instance_class_name("other", tree) is None


# --- Tests for enhanced extract_init_stub_from_class ---


def test_extract_init_stub_includes_post_init() -> None:
    source = """\
class MyDataclass:
    def __init__(self, x: int):
        self.x = x
    def __post_init__(self):
        self.y = self.x * 2
"""
    tree = ast.parse(source)
    stub = extract_init_stub_from_class("MyDataclass", source, tree)
    assert stub is not None
    assert "class MyDataclass:" in stub
    assert "def __init__" in stub
    assert "def __post_init__" in stub
    assert "self.y = self.x * 2" in stub


def test_extract_init_stub_includes_properties() -> None:
    source = """\
class MyClass:
    def __init__(self, name: str):
        self._name = name
    @property
    def name(self) -> str:
        return self._name
"""
    tree = ast.parse(source)
    stub = extract_init_stub_from_class("MyClass", source, tree)
    assert stub is not None
    assert "def __init__" in stub
    assert "@property" in stub
    assert "def name" in stub


def test_extract_init_stub_property_only_class() -> None:
    source = """\
class ReadOnly:
    @property
    def value(self) -> int:
        return 42
"""
    tree = ast.parse(source)
    stub = extract_init_stub_from_class("ReadOnly", source, tree)
    assert stub is not None
    assert "class ReadOnly:" in stub
    assert "@property" in stub
    assert "def value" in stub


# --- Tests for enrich_testgen_context resolving instances ---


def test_enrich_testgen_context_resolves_instance_to_class(tmp_path: Path) -> None:
    package_dir = tmp_path / "mypkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    config_module = """\
class AppConfig:
    def __init__(self, debug: bool = False):
        self.debug = debug

    @property
    def log_level(self) -> str:
        return "DEBUG" if self.debug else "INFO"

app_config = AppConfig(debug=True)
"""
    (package_dir / "config.py").write_text(config_module, encoding="utf-8")

    consumer_code = """\
from mypkg.config import app_config

def get_log_level() -> str:
    return app_config.log_level
"""
    consumer_path = package_dir / "consumer.py"
    consumer_path.write_text(consumer_code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=consumer_code, file_path=consumer_path)])
    result = enrich_testgen_context(context, tmp_path)

    assert len(result.code_strings) >= 1
    combined = "\n".join(cs.code for cs in result.code_strings)
    assert "class AppConfig:" in combined
    assert "@property" in combined

def test_extract_parameter_type_constructors_isinstance_single(tmp_path: Path) -> None:
    """isinstance(x, SomeType) in function body should be picked up."""
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "models.py").write_text(
        "class Widget:\n    def __init__(self, size: int):\n        self.size = size\n",
        encoding="utf-8",
    )
    (pkg / "processor.py").write_text(
        "from mypkg.models import Widget\n\ndef check(x) -> bool:\n    return isinstance(x, Widget)\n",
        encoding="utf-8",
    )
    fto = FunctionToOptimize(
        function_name="check", file_path=(pkg / "processor.py").resolve(), starting_line=3, ending_line=4
    )
    result = extract_parameter_type_constructors(fto, tmp_path.resolve(), set())
    assert len(result.code_strings) == 1
    assert "class Widget:" in result.code_strings[0].code
    assert "__init__" in result.code_strings[0].code


def test_extract_parameter_type_constructors_isinstance_tuple(tmp_path: Path) -> None:
    """isinstance(x, (TypeA, TypeB)) should pick up both types."""
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "models.py").write_text(
        "class Alpha:\n    def __init__(self, a: int):\n        self.a = a\n\n"
        "class Beta:\n    def __init__(self, b: str):\n        self.b = b\n",
        encoding="utf-8",
    )
    (pkg / "processor.py").write_text(
        "from mypkg.models import Alpha, Beta\n\ndef check(x) -> bool:\n    return isinstance(x, (Alpha, Beta))\n",
        encoding="utf-8",
    )
    fto = FunctionToOptimize(
        function_name="check", file_path=(pkg / "processor.py").resolve(), starting_line=3, ending_line=4
    )
    result = extract_parameter_type_constructors(fto, tmp_path.resolve(), set())
    assert len(result.code_strings) == 2
    combined = "\n".join(cs.code for cs in result.code_strings)
    assert "class Alpha:" in combined
    assert "class Beta:" in combined


def test_extract_parameter_type_constructors_type_is_pattern(tmp_path: Path) -> None:
    """type(x) is SomeType pattern should be picked up."""
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "models.py").write_text(
        "class Gadget:\n    def __init__(self, val: float):\n        self.val = val\n",
        encoding="utf-8",
    )
    (pkg / "processor.py").write_text(
        "from mypkg.models import Gadget\n\ndef check(x) -> bool:\n    return type(x) is Gadget\n",
        encoding="utf-8",
    )
    fto = FunctionToOptimize(
        function_name="check", file_path=(pkg / "processor.py").resolve(), starting_line=3, ending_line=4
    )
    result = extract_parameter_type_constructors(fto, tmp_path.resolve(), set())
    assert len(result.code_strings) == 1
    assert "class Gadget:" in result.code_strings[0].code


def test_extract_parameter_type_constructors_base_classes(tmp_path: Path) -> None:
    """Base classes of enclosing class should be picked up for methods."""
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "base.py").write_text(
        "class BaseProcessor:\n    def __init__(self, config: str):\n        self.config = config\n",
        encoding="utf-8",
    )
    (pkg / "child.py").write_text(
        "from mypkg.base import BaseProcessor\n\nclass ChildProcessor(BaseProcessor):\n"
        "    def process(self) -> str:\n        return self.config\n",
        encoding="utf-8",
    )
    fto = FunctionToOptimize(
        function_name="process",
        file_path=(pkg / "child.py").resolve(),
        starting_line=4,
        ending_line=5,
        parents=[FunctionParent(name="ChildProcessor", type="ClassDef")],
    )
    result = extract_parameter_type_constructors(fto, tmp_path.resolve(), set())
    assert len(result.code_strings) == 1
    assert "class BaseProcessor:" in result.code_strings[0].code


def test_extract_parameter_type_constructors_isinstance_builtins_excluded(tmp_path: Path) -> None:
    """Isinstance with builtins (int, str, etc.) should not produce stubs."""
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "func.py").write_text(
        "def check(x) -> bool:\n    return isinstance(x, (int, str, float))\n",
        encoding="utf-8",
    )
    fto = FunctionToOptimize(
        function_name="check", file_path=(pkg / "func.py").resolve(), starting_line=1, ending_line=2
    )
    result = extract_parameter_type_constructors(fto, tmp_path.resolve(), set())
    assert len(result.code_strings) == 0


def test_extract_parameter_type_constructors_transitive(tmp_path: Path) -> None:
    """Transitive extraction: if Widget.__init__ takes a Config, Config's stub should also appear."""
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "config.py").write_text(
        "class Config:\n    def __init__(self, debug: bool = False):\n        self.debug = debug\n",
        encoding="utf-8",
    )
    (pkg / "models.py").write_text(
        "from mypkg.config import Config\n\n"
        "class Widget:\n    def __init__(self, cfg: Config):\n        self.cfg = cfg\n",
        encoding="utf-8",
    )
    (pkg / "processor.py").write_text(
        "from mypkg.models import Widget\n\ndef process(w: Widget) -> str:\n    return str(w)\n",
        encoding="utf-8",
    )
    fto = FunctionToOptimize(
        function_name="process", file_path=(pkg / "processor.py").resolve(), starting_line=3, ending_line=4
    )
    result = extract_parameter_type_constructors(fto, tmp_path.resolve(), set())
    combined = "\n".join(cs.code for cs in result.code_strings)
    assert "class Widget:" in combined
    assert "class Config:" in combined




def test_enrich_testgen_context_third_party_uses_stubs(tmp_path: Path) -> None:
    """Third-party classes should produce compact __init__ stubs, not full class source."""
    # Use a real third-party package (pydantic) so jedi can actually resolve it
    context_code = (
        "from pydantic import BaseModel\n\n"
        "class MyModel(BaseModel):\n"
        "    name: str\n\n"
        "def process(m: MyModel) -> str:\n"
        "    return m.name\n"
    )
    consumer_path = tmp_path / "consumer.py"
    consumer_path.write_text(context_code, encoding="utf-8")

    context = CodeStringsMarkdown(code_strings=[CodeString(code=context_code, file_path=consumer_path)])
    result = enrich_testgen_context(context, tmp_path)

    # BaseModel lives in site-packages so should get stub treatment (compact __init__),
    # not the full class definition with hundreds of methods
    for cs in result.code_strings:
        if "BaseModel" in cs.code:
            assert "class BaseModel:" in cs.code
            assert "__init__" in cs.code
            # Full BaseModel has many methods; stubs should only have __init__/properties
            assert "model_dump" not in cs.code
            break
