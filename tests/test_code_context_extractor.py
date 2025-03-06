from __future__ import annotations

import tempfile
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import pytest

from codeflash.context.code_context_extractor import get_code_optimization_context
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import FunctionParent
from codeflash.optimization.optimizer import Optimizer


class HelperClass:
    def __init__(self, name):
        self.name = name

    def innocent_bystander(self):
        pass

    def helper_method(self):
        return self.name


def main_method():
    return "hello"


class MainClass:
    def __init__(self, name):
        self.name = name

    def main_method(self):
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
    read_write_context, read_only_context = code_ctx.read_writable_code, code_ctx.read_only_context_code

    expected_read_write_context = """
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
        return HelperClass(self.name).helper_method()
"""
    expected_read_only_context = """
    """

    assert read_write_context.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()


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
    expected_read_write_context = """
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

"""
    expected_read_only_context = ""
    assert read_write_context.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()


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

    expected_read_write_context = """
import math
from bubble_sort_with_math import sorter

def sorter(arr):
    arr.sort()
    x = math.sqrt(2)
    print(x)
    return arr



def sort_from_another_file(arr):
    sorted_arr = sorter(arr)
    return sorted_arr

"""
    expected_read_only_context = ""

    assert read_write_context.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()


def test_flavio_typed_code_helper() -> None:
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
    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write(code)
        f.flush()
        file_path = Path(f.name).resolve()
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
        expected_read_write_context = """
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
        assert read_write_context.strip() == expected_read_write_context.strip()
        assert read_only_context.strip() == expected_read_only_context.strip()


def test_example_class() -> None:
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
    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write(code)
        f.flush()
        file_path = Path(f.name).resolve()
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
        expected_read_write_context = """
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
        assert read_write_context.strip() == expected_read_write_context.strip()
        assert read_only_context.strip() == expected_read_only_context.strip()


def test_example_class_token_limit_1() -> None:
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
    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write(code)
        f.flush()
        file_path = Path(f.name).resolve()
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
        # In this scenario, the read-only code context is too long, so the read-only docstrings are removed.
        expected_read_write_context = """
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
        assert read_write_context.strip() == expected_read_write_context.strip()
        assert read_only_context.strip() == expected_read_only_context.strip()


def test_example_class_token_limit_2() -> None:
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
    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write(code)
        f.flush()
        file_path = Path(f.name).resolve()
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
        # In this scenario, the read-only code context is too long even after removing docstrings, hence we remove it completely.
        expected_read_write_context = """
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
        """
        expected_read_only_context = ""
        assert read_write_context.strip() == expected_read_write_context.strip()
        assert read_only_context.strip() == expected_read_only_context.strip()


def test_example_class_token_limit_3() -> None:
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
    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write(code)
        f.flush()
        file_path = Path(f.name).resolve()
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

def test_example_class_token_limit_4() -> None:
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
    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write(code)
        f.flush()
        file_path = Path(f.name).resolve()
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
    expected_read_write_context = """
import math
import requests
from globals import API_URL
from utils import DataProcessor

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

    """
    expected_read_only_context = f"""
```python:{path_to_utils.relative_to(project_root)}
GLOBAL_VAR = 10


class DataProcessor:
    \"\"\"A class for processing data.\"\"\"

    number = 1

    def __repr__(self) -> str:
        \"\"\"Return a string representation of the DataProcessor.\"\"\"
        return f"DataProcessor(default_prefix={{self.default_prefix!r}})"
```
```python:{path_to_file.relative_to(project_root)}
if __name__ == "__main__":
    result = fetch_and_process_data()
    print("Processed data:", result)
```
"""
    assert read_write_context.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()


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
    expected_read_write_context = """
import math
from transform_utils import DataTransformer
import requests
from globals import API_URL
from utils import DataProcessor

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



def fetch_and_transform_data():
    # Use the global variable for the request
    response = requests.get(API_URL)

    raw_data = response.text

    # Use code from another file (utils.py)
    processor = DataProcessor()
    processed = processor.process_data(raw_data)
    transformed = processor.transform_data(processed)

    return transformed

    """
    expected_read_only_context = f"""
```python:{path_to_utils.relative_to(project_root)}
GLOBAL_VAR = 10


class DataProcessor:
    \"\"\"A class for processing data.\"\"\"

    number = 1

    def __repr__(self) -> str:
        \"\"\"Return a string representation of the DataProcessor.\"\"\"
        return f"DataProcessor(default_prefix={{self.default_prefix!r}})"
```
```python:{path_to_file.relative_to(project_root)}
if __name__ == "__main__":
    result = fetch_and_process_data()
    print("Processed data:", result)
```
```python:{path_to_transform_utils.relative_to(project_root)}
class DataTransformer:

    def transform(self, data):
        self.data = data
        return self.data
```
"""

    assert read_write_context.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()


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
    expected_read_write_context = """
import math
from transform_utils import DataTransformer

class DataTransformer:
    def __init__(self):
        self.data = None

    def transform_using_own_method(self, data):
        return self.transform(data)



class DataProcessor:

    def __init__(self, default_prefix: str = "PREFIX_"):
        \"\"\"Initialize the DataProcessor with a default prefix.\"\"\"
        self.default_prefix = default_prefix
        self.number += math.log(self.number)

    def transform_data_own_method(self, data: str) -> str:
        \"\"\"Transform the processed data using own method\"\"\"
        return DataTransformer().transform_using_own_method(data)

"""
    expected_read_only_context = f"""
```python:{path_to_transform_utils.relative_to(project_root)}
class DataTransformer:

    def transform(self, data):
        self.data = data
        return self.data
```
```python:{path_to_utils.relative_to(project_root)}
GLOBAL_VAR = 10


class DataProcessor:
    \"\"\"A class for processing data.\"\"\"

    number = 1

    def __repr__(self) -> str:
        \"\"\"Return a string representation of the DataProcessor.\"\"\"
        return f"DataProcessor(default_prefix={{self.default_prefix!r}})"
```

"""

    assert read_write_context.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()


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
    expected_read_write_context = """
import math
from transform_utils import DataTransformer

class DataTransformer:
    def __init__(self):
        self.data = None

    def transform_using_same_file_function(self, data):
        return update_data(data)



class DataProcessor:

    def __init__(self, default_prefix: str = "PREFIX_"):
        \"\"\"Initialize the DataProcessor with a default prefix.\"\"\"
        self.default_prefix = default_prefix
        self.number += math.log(self.number)

    def transform_data_same_file_function(self, data: str) -> str:
        \"\"\"Transform the processed data using a function from the same file\"\"\"
        return DataTransformer().transform_using_same_file_function(data)
    """
    expected_read_only_context = f"""
```python:{path_to_transform_utils.relative_to(project_root)}
def update_data(data):
    return data + " updated"
```
```python:{path_to_utils.relative_to(project_root)}
GLOBAL_VAR = 10


class DataProcessor:
    \"\"\"A class for processing data.\"\"\"

    number = 1

    def __repr__(self) -> str:
        \"\"\"Return a string representation of the DataProcessor.\"\"\"
        return f"DataProcessor(default_prefix={{self.default_prefix!r}})"
```
"""

    assert read_write_context.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()


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
    expected_read_write_context = """
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
    """
    expected_read_only_context = f"""
```python:{path_to_transform_utils.relative_to(project_root)}
class DataTransformer:

    def transform(self, data):
        self.data = data
        return self.data
```

"""

    assert read_write_context.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()


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
    expected_read_write_context = """
import math
from transform_utils import DataTransformer
from code_to_optimize.code_directories.retriever.utils import DataProcessor

class DataProcessor:

    def __init__(self, default_prefix: str = "PREFIX_"):
        \"\"\"Initialize the DataProcessor with a default prefix.\"\"\"
        self.default_prefix = default_prefix
        self.number += math.log(self.number)

    def circular_dependency(self, data: str) -> str:
        \"\"\"Test circular dependency\"\"\"
        return DataTransformer().circular_dependency(data)



class DataTransformer:
    def __init__(self):
        self.data = None

    def circular_dependency(self, data):
        return DataProcessor().circular_dependency(data)


    """
    expected_read_only_context = f"""
```python:{path_to_utils.relative_to(project_root)}
GLOBAL_VAR = 10


class DataProcessor:
    \"\"\"A class for processing data.\"\"\"

    number = 1

    def __repr__(self) -> str:
        \"\"\"Return a string representation of the DataProcessor.\"\"\"
        return f"DataProcessor(default_prefix={{self.default_prefix!r}})"
```

"""

    assert read_write_context.strip() == expected_read_write_context.strip()
    assert read_only_context.strip() == expected_read_only_context.strip()
