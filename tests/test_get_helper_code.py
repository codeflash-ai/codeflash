import pathlib
import tempfile
from argparse import Namespace

import pytest
from codeflash.discovery.functions_to_optimize import FunctionParent, FunctionToOptimize
from codeflash.optimization.optimizer import Optimizer
from returns.pipeline import is_successful


class HelperClass:
    def helper_method(self, a, b, c):
        return a + b + c


def OptimizeMe(a, b, c):
    return HelperClass().helper_method(a, b, c)


@pytest.mark.skip()
def test_get_outside_method_helper() -> None:
    file_path = pathlib.Path(__file__).resolve()
    opt = Optimizer(
        Namespace(
            project_root=str(file_path.parent.resolve()),
            disable_telemetry=True,
            tests_root="tests",
            test_framework="pytest",
            pytest_cmd="pytest",
            experiment_id=None,
        ),
    )
    function_to_optimize = FunctionToOptimize(
        function_name="OptimizeMe",
        file_path=str(file_path),
        parents=[],
        starting_line=None,
        ending_line=None,
    )
    with open(file_path) as f:
        original_code = f.read()
    ctx_result = opt.get_code_optimization_context(
        function_to_optimize,
        opt.args.project_root,
        original_code,
    )
    if not is_successful(ctx_result):
        pytest.fail()
    code_context = ctx_result.unwrap()
    print("hi")


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
        file_path = pathlib.Path(f.name).resolve()
        opt = Optimizer(
            Namespace(
                project_root=str(file_path.parent.resolve()),
                disable_telemetry=True,
                tests_root="tests",
                test_framework="pytest",
                pytest_cmd="pytest",
                experiment_id=None,
            ),
        )
        function_to_optimize = FunctionToOptimize(
            function_name="__call__",
            file_path=str(file_path),
            parents=[FunctionParent(name="_PersistentCache", type="ClassDef")],
            starting_line=None,
            ending_line=None,
        )
        with open(file_path) as f:
            original_code = f.read()
        ctx_result = opt.get_code_optimization_context(
            function_to_optimize,
            opt.args.project_root,
            original_code,
        )
        if not is_successful(ctx_result):
            pytest.fail()
        code_context = ctx_result.unwrap()
        assert code_context.helper_functions[0].qualified_name == "_R"
        assert code_context.helper_functions[1].qualified_name == "AbstractCacheBackend.get_cache_or_call"
        assert len(code_context.contextual_dunder_methods) == 2

        assert (
            code_context.code_to_optimize_with_helpers
            == '''_R = TypeVar("_R")

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
        )
