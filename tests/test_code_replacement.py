from __future__ import annotations

import os

from codeflash.code_utils.code_replacer import replace_functions_in_file

os.environ["CODEFLASH_API_KEY"] = "cf-test-key"


def test_test_libcst_code_replacement() -> None:
    optim_code = """import libcst as cst
from typing import Optional

def totally_new_function(value):
    return value

class NewClass:
    def __init__(self, name):
        self.name = name
    def new_function(self, value):
        return self.name
    def new_function2(value):
        return value
    """

    original_code = """class NewClass:
    def __init__(self, name):
        self.name = name
    def new_function(self, value):
        return "I am still old"

print("Hello world")
"""
    expected = """import libcst as cst
from typing import Optional
class NewClass:
    def __init__(self, name):
        self.name = name
    def new_function(self, value):
        return self.name
    def new_function2(value):
        return value

def totally_new_function(value):
    return value

print("Hello world")
"""

    function_name: str = "NewClass.new_function"
    preexisting_functions: list[str] = ["new_function"]
    contextual_functions: set[tuple[str, str]] = {("NewClass", "__init__")}
    new_code: str = replace_functions_in_file(
        original_code,
        [function_name],
        optim_code,
        preexisting_functions,
        contextual_functions,
    )
    assert new_code == expected


def test_test_libcst_code_replacement2() -> None:
    optim_code = """import libcst as cst
from typing import Optional

def totally_new_function(value):
    return value

def other_function(st):
    return(st * 2)

class NewClass:
    def __init__(self, name):
        self.name = name
    def new_function(self, value):
        return other_function(self.name)
    def new_function2(value):
        return value
    """

    original_code = """from OtherModule import other_function

class NewClass:
    def __init__(self, name):
        self.name = name
    def new_function(self, value):
        return other_function("I am still old")

print("Hello world")
"""
    expected = """import libcst as cst
from typing import Optional
from OtherModule import other_function

class NewClass:
    def __init__(self, name):
        self.name = name
    def new_function(self, value):
        return other_function(self.name)
    def new_function2(value):
        return value

def totally_new_function(value):
    return value

print("Hello world")
"""

    function_name: str = "NewClass.new_function"
    preexisting_functions: list[str] = ["new_function", "other_function"]
    contextual_functions: set[tuple[str, str]] = {("NewClass", "__init__")}
    new_code: str = replace_functions_in_file(
        original_code,
        [function_name],
        optim_code,
        preexisting_functions,
        contextual_functions,
    )
    assert new_code == expected


def test_test_libcst_code_replacement3() -> None:
    optim_code = """import libcst as cst
from typing import Optional

def totally_new_function(value):
    return value

def other_function(st):
    return(st * 2)

class NewClass:
    def __init__(self, name):
        self.name = name
    def new_function(self, value):
        return other_function(self.name)
    def new_function2(value):
        return value
    """

    original_code = """import libcst as cst
from typing import Mandatory

print("Au revoir")

def yet_another_function(values):
    return len(values)

def other_function(st):
    return(st + st)

print("Salut monde")
"""
    expected = """import libcst as cst
from typing import Optional
import libcst as cst
from typing import Mandatory

print("Au revoir")

def yet_another_function(values):
    return len(values)

def other_function(st):
    return(st * 2)

print("Salut monde")
"""

    function_names: list[str] = ["module.other_function"]
    preexisting_functions: list[str] = []
    contextual_functions: set[tuple[str, str]] = set()
    new_code: str = replace_functions_in_file(
        original_code,
        function_names,
        optim_code,
        preexisting_functions,
        contextual_functions,
    )
    assert new_code == expected


def test_test_libcst_code_replacement4() -> None:
    optim_code = """import libcst as cst
from typing import Optional

def totally_new_function(value):
    return value

def yet_another_function(values):
    return len(values) + 2

def other_function(st):
    return(st * 2)

class NewClass:
    def __init__(self, name):
        self.name = name
    def new_function(self, value):
        return other_function(self.name)
    def new_function2(value):
        return value
    """

    original_code = """import libcst as cst
from typing import Mandatory

print("Au revoir")

def yet_another_function(values):
    return len(values)

def other_function(st):
    return(st + st)

print("Salut monde")
"""
    expected = """import libcst as cst
from typing import Optional
import libcst as cst
from typing import Mandatory

print("Au revoir")

def yet_another_function(values):
    return len(values) + 2

def other_function(st):
    return(st * 2)

print("Salut monde")
"""

    function_names: list[str] = ["module.yet_another_function", "module.other_function"]
    preexisting_functions: list[str] = []
    contextual_functions: set[tuple[str, str]] = set()
    new_code: str = replace_functions_in_file(
        original_code,
        function_names,
        optim_code,
        preexisting_functions,
        contextual_functions,
    )
    assert new_code == expected


def test_test_libcst_code_replacement5() -> None:
    optim_code = """def sorter_deps(arr):
    supersort(badsort(arr))
    return arr

def badsort(ploc):
    donothing(ploc)

def supersort(doink):
    for i in range(len(doink)):
        fix(doink, i)
"""

    original_code = """from code_to_optimize.bubble_sort_dep1_helper import dep1_comparer
from code_to_optimize.bubble_sort_dep2_swap import dep2_swap

def sorter_deps(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if dep1_comparer(arr, j):
                dep2_swap(arr, j)
    return arr
"""
    expected = """from code_to_optimize.bubble_sort_dep1_helper import dep1_comparer
from code_to_optimize.bubble_sort_dep2_swap import dep2_swap
def sorter_deps(arr):
    supersort(badsort(arr))
    return arr

def badsort(ploc):
    donothing(ploc)

def supersort(doink):
    for i in range(len(doink)):
        fix(doink, i)
"""

    function_names: list[str] = ["sorter_deps"]
    preexisting_functions: list[str] = ["sorter_deps"]
    contextual_functions: set[tuple[str, str]] = set()
    new_code: str = replace_functions_in_file(
        original_code,
        function_names,
        optim_code,
        preexisting_functions,
        contextual_functions,
    )
    assert new_code == expected


def test_test_libcst_code_replacement6() -> None:
    optim_code = """import libcst as cst
from typing import Optional

def other_function(st):
    return(st * blob(st))

def blob(st):
    return(st * 2)
"""
    original_code_main = """import libcst as cst
from typing import Mandatory
from dependent import blob

print("Au revoir")

def yet_another_function(values):
    return len(values)

def other_function(st):
    return(st + blob(st))

print("Salut monde")
"""

    original_code_dependent = """import numpy as np

print("Cool")

def blob(values):
    return len(values)

def blab(st):
    return(st + st)

print("Not cool")
"""
    expected_main = """import libcst as cst
from typing import Optional
import libcst as cst
from typing import Mandatory
from dependent import blob

print("Au revoir")

def yet_another_function(values):
    return len(values)

def other_function(st):
    return(st * blob(st))

print("Salut monde")
"""

    expected_dependent = """import libcst as cst
from typing import Optional
import numpy as np

print("Cool")

def blob(st):
    return(st * 2)

def blab(st):
    return(st + st)

print("Not cool")
"""
    new_main_code: str = replace_functions_in_file(
        original_code_main,
        ["other_function"],
        optim_code,
        ["other_function", "yet_another_function", "blob"],
        set(),
    )
    assert new_main_code == expected_main

    new_dependent_code: str = replace_functions_in_file(
        original_code_dependent,
        ["blob"],
        optim_code,
        [],
        set(),
    )
    assert new_dependent_code == expected_dependent


def test_test_libcst_code_replacement7() -> None:
    optim_code = """@register_deserializable
class CacheSimilarityEvalConfig(BaseConfig):

    def __init__(
        self,
        strategy: Optional[str] = "distance",
        max_distance: Optional[float] = 1.0,
        positive: Optional[bool] = False,
    ):
        self.strategy = strategy
        self.max_distance = max_distance
        self.positive = positive

    @staticmethod
    def from_config(config: Optional[dict[str, Any]]):
        if config is None:
            return CacheSimilarityEvalConfig()

        strategy = config.get("strategy", "distance")
        max_distance = config.get("max_distance", 1.0)
        positive = config.get("positive", False)

        return CacheSimilarityEvalConfig(strategy, max_distance, positive)
"""

    original_code = """from typing import Any, Optional

from embedchain.config.base_config import BaseConfig
from embedchain.helpers.json_serializable import register_deserializable


@register_deserializable
class CacheSimilarityEvalConfig(BaseConfig):

    def __init__(
        self,
        strategy: Optional[str] = "distance",
        max_distance: Optional[float] = 1.0,
        positive: Optional[bool] = False,
    ):
        self.strategy = strategy
        self.max_distance = max_distance
        self.positive = positive

    @staticmethod
    def from_config(config: Optional[dict[str, Any]]):
        if config is None:
            return CacheSimilarityEvalConfig()
        else:
            return CacheSimilarityEvalConfig(
                strategy=config.get("strategy", "distance"),
                max_distance=config.get("max_distance", 1.0),
                positive=config.get("positive", False),
            )


@register_deserializable
class CacheInitConfig(BaseConfig):

    def __init__(
        self,
        similarity_threshold: Optional[float] = 0.8,
        auto_flush: Optional[int] = 20,
    ):
        if similarity_threshold < 0 or similarity_threshold > 1:
            raise ValueError(f"similarity_threshold {similarity_threshold} should be between 0 and 1")

        self.similarity_threshold = similarity_threshold
        self.auto_flush = auto_flush

    @staticmethod
    def from_config(config: Optional[dict[str, Any]]):
        if config is None:
            return CacheInitConfig()
        else:
            return CacheInitConfig(
                similarity_threshold=config.get("similarity_threshold", 0.8),
                auto_flush=config.get("auto_flush", 20),
            )


@register_deserializable
class CacheConfig(BaseConfig):

    def __init__(
        self,
        similarity_eval_config: Optional[CacheSimilarityEvalConfig] = CacheSimilarityEvalConfig(),
        init_config: Optional[CacheInitConfig] = CacheInitConfig(),
    ):
        self.similarity_eval_config = similarity_eval_config
        self.init_config = init_config

    @staticmethod
    def from_config(config: Optional[dict[str, Any]]):
        if config is None:
            return CacheConfig()
        else:
            return CacheConfig(
                similarity_eval_config=CacheSimilarityEvalConfig.from_config(config.get("similarity_evaluation", {})),
                init_config=CacheInitConfig.from_config(config.get("init_config", {})),
            )
"""
    expected = """from typing import Any, Optional

from embedchain.config.base_config import BaseConfig
from embedchain.helpers.json_serializable import register_deserializable


@register_deserializable
class CacheSimilarityEvalConfig(BaseConfig):

    def __init__(
        self,
        strategy: Optional[str] = "distance",
        max_distance: Optional[float] = 1.0,
        positive: Optional[bool] = False,
    ):
        self.strategy = strategy
        self.max_distance = max_distance
        self.positive = positive

    @staticmethod
    def from_config(config: Optional[dict[str, Any]]):
        if config is None:
            return CacheSimilarityEvalConfig()

        strategy = config.get("strategy", "distance")
        max_distance = config.get("max_distance", 1.0)
        positive = config.get("positive", False)

        return CacheSimilarityEvalConfig(strategy, max_distance, positive)


@register_deserializable
class CacheInitConfig(BaseConfig):

    def __init__(
        self,
        similarity_threshold: Optional[float] = 0.8,
        auto_flush: Optional[int] = 20,
    ):
        if similarity_threshold < 0 or similarity_threshold > 1:
            raise ValueError(f"similarity_threshold {similarity_threshold} should be between 0 and 1")

        self.similarity_threshold = similarity_threshold
        self.auto_flush = auto_flush

    @staticmethod
    def from_config(config: Optional[dict[str, Any]]):
        if config is None:
            return CacheInitConfig()
        else:
            return CacheInitConfig(
                similarity_threshold=config.get("similarity_threshold", 0.8),
                auto_flush=config.get("auto_flush", 20),
            )


@register_deserializable
class CacheConfig(BaseConfig):

    def __init__(
        self,
        similarity_eval_config: Optional[CacheSimilarityEvalConfig] = CacheSimilarityEvalConfig(),
        init_config: Optional[CacheInitConfig] = CacheInitConfig(),
    ):
        self.similarity_eval_config = similarity_eval_config
        self.init_config = init_config

    @staticmethod
    def from_config(config: Optional[dict[str, Any]]):
        if config is None:
            return CacheConfig()
        else:
            return CacheConfig(
                similarity_eval_config=CacheSimilarityEvalConfig.from_config(config.get("similarity_evaluation", {})),
                init_config=CacheInitConfig.from_config(config.get("init_config", {})),
            )
"""
    function_names: list[str] = ["CacheSimilarityEvalConfig.from_config"]
    preexisting_functions: list[str] = [
        "__init__",
        "from_config",
    ]
    contextual_functions: set[tuple[str, str]] = {
        ("CacheSimilarityEvalConfig", "__init__"),
        ("CacheConfig", "__init__"),
        ("CacheInitConfig", "__init__"),
    }
    new_code: str = replace_functions_in_file(
        original_code,
        function_names,
        optim_code,
        preexisting_functions,
        contextual_functions,
    )
    assert new_code == expected


def test_test_libcst_code_replacement8() -> None:
    optim_code = '''class _EmbeddingDistanceChainMixin(Chain):
    @staticmethod
    def _hamming_distance(a: np.ndarray, b: np.ndarray) -> np.floating:
        """Compute the Hamming distance between two vectors.

        Args:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            np.floating: The Hamming distance.
        """
        return np.sum(a != b) / a.size
'''

    original_code = '''class _EmbeddingDistanceChainMixin(Chain):

    class Config:
        """Permit embeddings to go unvalidated."""

        arbitrary_types_allowed: bool = True


    @staticmethod
    def _hamming_distance(a: np.ndarray, b: np.ndarray) -> np.floating:
        """Compute the Hamming distance between two vectors.

        Args:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            np.floating: The Hamming distance.
        """
        return np.mean(a != b)
'''
    expected = '''class _EmbeddingDistanceChainMixin(Chain):

    class Config:
        """Permit embeddings to go unvalidated."""

        arbitrary_types_allowed: bool = True
    @staticmethod
    def _hamming_distance(a: np.ndarray, b: np.ndarray) -> np.floating:
        """Compute the Hamming distance between two vectors.

        Args:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            np.floating: The Hamming distance.
        """
        return np.sum(a != b) / a.size
'''
    function_names: list[str] = ["_EmbeddingDistanceChainMixin._hamming_distance"]
    preexisting_functions: list[str] = [
        "_hamming_distance",
    ]
    contextual_functions: set[tuple[str, str]] = set()
    new_code: str = replace_functions_in_file(
        original_code,
        function_names,
        optim_code,
        preexisting_functions,
        contextual_functions,
    )
    assert new_code == expected


def test_test_libcst_code_replacement9() -> None:
    optim_code = """import libcst as cst
from typing import Optional

def totally_new_function(value):
    return value

class NewClass:
    def __init__(self, name):
        self.name = str(name)
    def __call__(self, value):
        return self.name
    def new_function2(value):
        return value
    """

    original_code = """class NewClass:
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"

print("Hello world")
"""
    expected = """import libcst as cst
from typing import Optional
class NewClass:
    def __init__(self, name):
        self.name = str(name)
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return value

def totally_new_function(value):
    return value

print("Hello world")
"""

    function_name: str = "NewClass.__init__"
    preexisting_functions: list[str] = ["__init__", "__call__"]
    contextual_functions: set[tuple[str, str]] = {
        ("NewClass", "__init__"),
        ("NewClass", "__call__"),
    }
    new_code: str = replace_functions_in_file(
        original_code,
        [function_name],
        optim_code,
        preexisting_functions,
        contextual_functions,
    )
    assert new_code == expected
