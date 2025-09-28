from __future__ import annotations

import sys
import tempfile
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import pytest
from codeflash.context.code_context_extractor import get_code_optimization_context
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import FunctionParent
from codeflash.optimization.optimizer import Optimizer
from codeflash.code_utils.code_replacer import replace_functions_and_add_imports
from codeflash.code_utils.code_extractor import add_global_assignments


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
    assert qualified_names == {"HelperClass.helper_method"}  # Nested method should not be in here
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

    expected_read_write_context = f"""
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
        ["This is a long docstring that will be used to fill up the token limit." for _ in range(1000)]
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
    expected_read_only_context = ""
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
        ["This is a long string that will be used to fill up the token limit." for _ in range(1000)]
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
        code_ctx = get_code_optimization_context(function_to_optimize, opt.args.project_root)


def test_example_class_token_limit_4(tmp_path: Path) -> None:
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

    # In this scenario, the testgen code context is too long, so we abort.
    with pytest.raises(ValueError, match="Testgen code context has exceeded token limit, cannot proceed"):
        code_ctx = get_code_optimization_context(function_to_optimize, opt.args.project_root)


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
    expected_read_write_context = f"""
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
    expected_read_write_context = f"""
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
from transform_utils import DataTransformer

class DataProcessor:
    \"\"\"A class for processing data.\"\"\"

    number = 1

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
        opt = Optimizer(
            Namespace(
                project_root=package_dir.resolve(),
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
        expected_read_write_context = f"""
```python:{main_file_path.relative_to(opt.args.project_root)}
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
DEFAULT_PRECISION = "medium"

# Try-except block with variable definitions
try:
    # Used variable in try block
    CALCULATION_BACKEND = "numpy"
except ImportError:
    # Used variable in except block
    CALCULATION_BACKEND = "python"

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
        opt = Optimizer(
            Namespace(
                project_root=package_dir.resolve(),
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
        expected_read_write_context = f"""
```python:utility_module.py
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
```python:{main_file_path.relative_to(opt.args.project_root)}
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
DEFAULT_PRECISION = "medium"

# Try-except block with variable definitions
try:
    # Used variable in try block
    CALCULATION_BACKEND = "numpy"
except ImportError:
    # Used variable in except block
    CALCULATION_BACKEND = "python"
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
    assert '"""Process data method."""' not in hashing_context, (
        "Docstrings should be removed from helper class methods"
    )


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
    # The prune_cst_for_code_hashing function should not recurse into nested classes
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
        source_code= add_global_assignments(optimized_code, content),
        function_names= ["ApiClient.get_console_url"],
        optimized_code= optimized_code,
        module_abspath= Path(file_abs_path),
        preexisting_objects= {('ApiClient', ()), ('get_console_url', (FunctionParent(name='ApiClient', type='ClassDef'),))},
        project_root_path= Path(path_to_root),
    )
    assert "import ApiClient" not in new_code, "Error: Circular dependency found"
    
    assert "import urllib.parse" in new_code, "Make sure imports for optimization global assignments exist" 
