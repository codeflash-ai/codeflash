from __future__ import annotations

import dataclasses
import os
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import libcst as cst
from codeflash.code_utils.code_extractor import remove_first_imported_aliased_objects
from codeflash.code_utils.code_replacer import replace_functions_and_add_imports, replace_functions_in_file
from codeflash.discovery.functions_to_optimize import FunctionParent, FunctionToOptimize
from codeflash.optimization.optimizer import Optimizer

os.environ["CODEFLASH_API_KEY"] = "cf-test-key"


@dataclasses.dataclass
class JediDefinition:
    type: str


@dataclasses.dataclass
class FakeFunctionSource:
    file_path: str
    qualified_name: str
    fully_qualified_name: str
    only_function_name: str
    source_code: str
    jedi_definition: JediDefinition


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
    expected = """class NewClass:
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
    preexisting_objects: list[tuple[str, list[FunctionParent]]] = [
        ("new_function", [FunctionParent(name="NewClass", type="ClassDef")]),
    ]
    contextual_functions: set[tuple[str, str]] = {("NewClass", "__init__")}
    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=[function_name],
        optimized_code=optim_code,
        file_path_of_module_with_function_to_optimize=str(Path(__file__).resolve()),
        module_abspath=str(Path(__file__).resolve()),
        preexisting_objects=preexisting_objects,
        contextual_functions=contextual_functions,
        project_root_path=str(Path(__file__).resolve().parent.resolve()),
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
    expected = """from OtherModule import other_function

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
    preexisting_objects: list[tuple[str, list[FunctionParent]]] = [
        ("new_function", []),
        ("other_function", []),
    ]
    contextual_functions: set[tuple[str, str]] = {("NewClass", "__init__")}
    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=[function_name],
        optimized_code=optim_code,
        file_path_of_module_with_function_to_optimize=str(Path(__file__).resolve()),
        module_abspath=str(Path(__file__).resolve()),
        preexisting_objects=preexisting_objects,
        contextual_functions=contextual_functions,
        project_root_path=str(Path(__file__).resolve().parent.resolve()),
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
    def new_function(self, value: cst.Name):
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
    expected = """from typing import Mandatory

print("Au revoir")

def yet_another_function(values):
    return len(values)

def other_function(st):
    return(st * 2)

print("Salut monde")
"""

    function_names: list[str] = ["module.other_function"]
    preexisting_objects: list[tuple[str, list[FunctionParent]]] = []
    contextual_functions: set[tuple[str, str]] = set()
    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=function_names,
        optimized_code=optim_code,
        file_path_of_module_with_function_to_optimize=str(Path(__file__).resolve()),
        module_abspath=str(Path(__file__).resolve()),
        preexisting_objects=preexisting_objects,
        contextual_functions=contextual_functions,
        project_root_path=str(Path(__file__).resolve().parent.resolve()),
    )
    assert new_code == expected


def test_test_libcst_code_replacement4() -> None:
    optim_code = """import libcst as cst
from typing import Optional

def totally_new_function(value):
    return value

def yet_another_function(values: Optional[str]):
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
    expected = """from typing import Optional, Mandatory

print("Au revoir")

def yet_another_function(values: Optional[str]):
    return len(values) + 2

def other_function(st):
    return(st * 2)

print("Salut monde")
"""

    function_names: list[str] = ["module.yet_another_function", "module.other_function"]
    preexisting_objects: list[tuple[str, list[FunctionParent]]] = []
    contextual_functions: set[tuple[str, str]] = set()
    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=function_names,
        optimized_code=optim_code,
        file_path_of_module_with_function_to_optimize=str(Path(__file__).resolve()),
        module_abspath=str(Path(__file__).resolve()),
        preexisting_objects=preexisting_objects,
        contextual_functions=contextual_functions,
        project_root_path=str(Path(__file__).resolve().parent.resolve()),
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
    preexisting_objects: list[tuple[str, list[FunctionParent]]] = [("sorter_deps", [])]
    contextual_functions: set[tuple[str, str]] = set()
    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=function_names,
        optimized_code=optim_code,
        file_path_of_module_with_function_to_optimize=str(Path(__file__).resolve()),
        module_abspath=str(Path(__file__).resolve()),
        preexisting_objects=preexisting_objects,
        contextual_functions=contextual_functions,
        project_root_path=str(Path(__file__).resolve().parent.resolve()),
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
from helper import blob

print("Au revoir")

def yet_another_function(values):
    return len(values)

def other_function(st):
    return(st + blob(st))

print("Salut monde")
"""

    original_code_helper = """import numpy as np

print("Cool")

def blob(values):
    return len(values)

def blab(st):
    return(st + st)

print("Not cool")
"""
    expected_main = """from typing import Mandatory
from helper import blob

print("Au revoir")

def yet_another_function(values):
    return len(values)

def other_function(st):
    return(st * blob(st))

print("Salut monde")
"""

    expected_helper = """import numpy as np

print("Cool")

def blob(st):
    return(st * 2)

def blab(st):
    return(st + st)

print("Not cool")
"""
    new_main_code: str = replace_functions_and_add_imports(
        source_code=original_code_main,
        function_names=["other_function"],
        optimized_code=optim_code,
        file_path_of_module_with_function_to_optimize=str(Path(__file__).resolve()),
        module_abspath=str(Path(__file__).resolve()),
        preexisting_objects=[("other_function", []), ("yet_another_function", []), ("blob", [])],
        contextual_functions=set(),
        project_root_path=str(Path(__file__).resolve().parent.resolve()),
    )
    assert new_main_code == expected_main

    new_helper_code: str = replace_functions_and_add_imports(
        source_code=original_code_helper,
        function_names=["blob"],
        optimized_code=optim_code,
        file_path_of_module_with_function_to_optimize=str(Path(__file__).resolve()),
        module_abspath=str(Path(__file__).resolve()),
        preexisting_objects=[],
        contextual_functions=set(),
        project_root_path=str(Path(__file__).resolve().parent.resolve()),
    )
    assert new_helper_code == expected_helper


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
    parents = [FunctionParent(name="CacheConfig", type="ClassDef")]
    preexisting_objects: list[tuple[str, list[FunctionParent]]] = [
        ("__init__", parents),
        ("from_config", parents),
    ]

    contextual_functions: set[tuple[str, str]] = {
        ("CacheSimilarityEvalConfig", "__init__"),
        ("CacheConfig", "__init__"),
        ("CacheInitConfig", "__init__"),
    }
    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=function_names,
        optimized_code=optim_code,
        file_path_of_module_with_function_to_optimize=str(Path(__file__).resolve()),
        module_abspath=str(Path(__file__).resolve()),
        preexisting_objects=preexisting_objects,
        contextual_functions=contextual_functions,
        project_root_path=str(Path(__file__).resolve().parent.resolve()),
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
    preexisting_objects: list[tuple[str, list[FunctionParent]]] = [
        ("_hamming_distance", [FunctionParent("_EmbeddingDistanceChainMixin", "ClassDef")]),
    ]
    contextual_functions: set[tuple[str, str]] = set()
    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=function_names,
        optimized_code=optim_code,
        file_path_of_module_with_function_to_optimize=str(Path(__file__).resolve()),
        module_abspath=str(Path(__file__).resolve()),
        preexisting_objects=preexisting_objects,
        contextual_functions=contextual_functions,
        project_root_path=str(Path(__file__).resolve().parent.resolve()),
    )
    assert new_code == expected


def test_test_libcst_code_replacement9() -> None:
    optim_code = """import libcst as cst
from typing import Optional

def totally_new_function(value: Optional[str]):
    return value

class NewClass:
    def __init__(self, name):
        self.name = str(name)
    def __call__(self, value):
        return self.name
    def new_function2(value):
        return cst.ensure_type(value, str)
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
        return cst.ensure_type(value, str)

def totally_new_function(value: Optional[str]):
    return value

print("Hello world")
"""
    parents = [FunctionParent(name="NewClass", type="ClassDef")]
    function_name: str = "NewClass.__init__"
    preexisting_objects: list[tuple[str, list[FunctionParent]]] = [
        ("__init__", parents),
        ("__call__", parents),
    ]
    contextual_functions: set[tuple[str, str]] = {
        ("NewClass", "__init__"),
        ("NewClass", "__call__"),
    }
    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=[function_name],
        optimized_code=optim_code,
        file_path_of_module_with_function_to_optimize=str(Path(__file__).resolve()),
        module_abspath=str(Path(__file__).resolve()),
        preexisting_objects=preexisting_objects,
        contextual_functions=contextual_functions,
        project_root_path=str(Path(__file__).resolve().parent.resolve()),
    )
    assert new_code == expected


class HelperClass:
    def __init__(self, name):
        self.name = name

    def innocent_bystander(self):
        pass

    def helper_method(self):
        return self.name


class MainClass:
    def __init__(self, name):
        self.name = name

    def main_method(self):
        return HelperClass(self.name).helper_method()


def test_code_replacement10() -> None:
    get_code_output = """from __future__ import annotations

class HelperClass:
    def __init__(self, name):
        self.name = name

    def innocent_bystander(self):
        pass

    def helper_method(self):
        return self.name

class MainClass:
    def __init__(self, name):
        self.name = name
    def main_method(self):
        return HelperClass(self.name).helper_method()
"""
    file_path = Path(__file__).resolve()
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
    func_top_optimize = FunctionToOptimize(
        function_name="main_method",
        file_path=str(file_path),
        parents=[FunctionParent("MainClass", "ClassDef")],
    )
    with open(file_path) as f:
        original_code = f.read()
        code_context = opt.get_code_optimization_context(
            function_to_optimize=func_top_optimize,
            project_root=str(file_path.parent),
            original_source_code=original_code,
        ).unwrap()
        assert code_context.code_to_optimize_with_helpers == get_code_output


def test_code_replacement11() -> None:
    optim_code = '''class Fu():
    def foo(self) -> dict[str, str]:
        payload: dict[str, str] = {"bar": self.bar(), "real_bar": str(self.real_bar() + 1)}
        return payload

    def real_bar(self) -> int:
        """No abstract nonsense"""
        pass
'''
    original_code = '''class Fu():
    def foo(self) -> dict[str, str]:
        payload: dict[str, str] = {"bar": self.bar(), "real_bar": str(self.real_bar())}
        return payload

    def real_bar(self) -> int:
        """No abstract nonsense"""
        return 0
'''
    expected_code = '''class Fu():
    def foo(self) -> dict[str, str]:
        payload: dict[str, str] = {"bar": self.bar(), "real_bar": str(self.real_bar() + 1)}
        return payload

    def real_bar(self) -> int:
        """No abstract nonsense"""
        return 0
'''

    function_name: str = "Fu.foo"
    parents = [FunctionParent("Fu", "ClassDef")]
    preexisting_objects: list[tuple[str, list[FunctionParent]]] = [("foo", parents), ("real_bar", parents)]
    contextual_functions: set[tuple[str, str]] = set()
    new_code: str = replace_functions_in_file(
        source_code=original_code,
        original_function_names=[function_name],
        optimized_code=optim_code,
        preexisting_objects=preexisting_objects,
        contextual_functions=contextual_functions,
    )
    assert new_code == expected_code


def test_code_replacement12() -> None:
    optim_code = '''class Fu():
    def foo(self) -> dict[str, str]:
        payload: dict[str, str] = {"bar": self.bar(), "real_bar": str(self.real_bar() + 1)}
        return payload

    def real_bar(self) -> int:
        """No abstract nonsense"""
        pass
'''
    original_code = '''class Fu():
    def foo(self) -> dict[str, str]:
        payload: dict[str, str] = {"bar": self.bar(), "real_bar": str(self.real_bar())}
        return payload

    def real_bar(self) -> int:
        """No abstract nonsense"""
        return 0
'''
    expected_code = '''class Fu():
    def foo(self) -> dict[str, str]:
        payload: dict[str, str] = {"bar": self.bar(), "real_bar": str(self.real_bar())}
        return payload

    def real_bar(self) -> int:
        """No abstract nonsense"""
        pass
'''

    preexisting_objects: list[tuple[str, list[FunctionParent]]] = []
    contextual_functions: set[tuple[str, str]] = set()
    new_code: str = replace_functions_in_file(
        source_code=original_code,
        original_function_names=["Fu.real_bar"],
        optimized_code=optim_code,
        preexisting_objects=preexisting_objects,
        contextual_functions=contextual_functions,
    )
    assert new_code == expected_code


def test_test_libcst_code_replacement13() -> None:
    # Test if the dunder method is not modified
    optim_code = """class NewClass:
    def __init__(self, name):
        self.name = name
        self.new_attribute = "Sorry i modified a dunder method"
    def new_function(self, value):
        return other_function(self.name)
    def new_function2(value):
        return value
    def __call__(self, value):
        return self.new_attribute
    """

    original_code = """class NewClass:
    def __init__(self, name):
        self.name = name
    def new_function(self, value):
        return other_function(self.name)
    def new_function2(value):
        return value
    def __call__(self, value):
        return self.name
"""

    function_names: list[str] = ["module.yet_another_function", "module.other_function"]
    preexisting_objects: list[tuple[str, list[FunctionParent]]] = []
    contextual_functions: set[tuple[str, str]] = set()
    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=function_names,
        optimized_code=optim_code,
        file_path_of_module_with_function_to_optimize=str(Path(__file__).resolve()),
        module_abspath=str(Path(__file__).resolve()),
        preexisting_objects=preexisting_objects,
        contextual_functions=contextual_functions,
        project_root_path=str(Path(__file__).resolve().parent.resolve()),
    )
    assert new_code == original_code


def test_different_class_code_replacement():
    original_code = """from __future__ import annotations
import sys
from codeflash.verification.comparator import comparator
from enum import Enum
from pydantic import BaseModel
from typing import Iterator

class TestType(Enum):
    EXISTING_UNIT_TEST = 1
    INSPIRED_REGRESSION = 2
    GENERATED_REGRESSION = 3
    REPLAY_TEST = 4

    def to_name(self) -> str:
        names = {
            TestType.EXISTING_UNIT_TEST: "⚙️ Existing Unit Tests",
            TestType.INSPIRED_REGRESSION: "🎨 Inspired Regression Tests",
            TestType.GENERATED_REGRESSION: "🌀 Generated Regression Tests",
            TestType.REPLAY_TEST: "⏪ Replay Tests",
        }
        return names[self]

class TestResults(BaseModel):
    def __iter__(self) -> Iterator[FunctionTestInvocation]:
        return iter(self.test_results)
    def __len__(self) -> int:
        return len(self.test_results)
    def __getitem__(self, index: int) -> FunctionTestInvocation:
        return self.test_results[index]
    def __setitem__(self, index: int, value: FunctionTestInvocation) -> None:
        self.test_results[index] = value
    def __delitem__(self, index: int) -> None:
        del self.test_results[index]
    def __contains__(self, value: FunctionTestInvocation) -> bool:
        return value in self.test_results
    def __bool__(self) -> bool:
        return bool(self.test_results)
    def __eq__(self, other: object) -> bool:
        # Unordered comparison
        if type(self) != type(other):
            return False
        if len(self) != len(other):
            return False
        original_recursion_limit = sys.getrecursionlimit()
        for test_result in self:
            other_test_result = other.get_by_id(test_result.id)
            if other_test_result is None:
                return False

            if original_recursion_limit < 5000:
                sys.setrecursionlimit(5000)
            if (
                test_result.file_name != other_test_result.file_name
                or test_result.did_pass != other_test_result.did_pass
                or test_result.runtime != other_test_result.runtime
                or test_result.test_framework != other_test_result.test_framework
                or test_result.test_type != other_test_result.test_type
                or not comparator(
                    test_result.return_value,
                    other_test_result.return_value,
                )
            ):
                sys.setrecursionlimit(original_recursion_limit)
                return False
        sys.setrecursionlimit(original_recursion_limit)
        return True
    def get_test_pass_fail_report_by_type(self) -> dict[TestType, dict[str, int]]:
        report = {}
        for test_type in TestType:
            report[test_type] = {"passed": 0, "failed": 0}
        for test_result in self.test_results:
            if test_result.test_type != TestType.EXISTING_UNIT_TEST or test_result.id.function_getting_tested:
                if test_result.did_pass:
                    report[test_result.test_type]["passed"] += 1
                else:
                    report[test_result.test_type]["failed"] += 1
        return report"""
    optim_code = """from __future__ import annotations

import sys
from enum import Enum
from typing import Iterator

from codeflash.verification.comparator import comparator
from pydantic import BaseModel


class TestType(Enum):
    EXISTING_UNIT_TEST = 1
    INSPIRED_REGRESSION = 2
    GENERATED_REGRESSION = 3
    REPLAY_TEST = 4

    def to_name(self) -> str:
        if self == TestType.EXISTING_UNIT_TEST:
            return "⚙️ Existing Unit Tests"
        elif self == TestType.INSPIRED_REGRESSION:
            return "🎨 Inspired Regression Tests"
        elif self == TestType.GENERATED_REGRESSION:
            return "🌀 Generated Regression Tests"
        elif self == TestType.REPLAY_TEST:
            return "⏪ Replay Tests"

class TestResults(BaseModel):
    def __iter__(self) -> Iterator[FunctionTestInvocation]:
        return iter(self.test_results)
    
    def __len__(self) -> int:
        return len(self.test_results)
    
    def __getitem__(self, index: int) -> FunctionTestInvocation:
        return self.test_results[index]
    
    def __setitem__(self, index: int, value: FunctionTestInvocation) -> None:
        self.test_results[index] = value
    
    def __delitem__(self, index: int) -> None:
        del self.test_results[index]
    
    def __contains__(self, value: FunctionTestInvocation) -> bool:
        return value in self.test_results
    
    def __bool__(self) -> bool:
        return bool(self.test_results)
    
    def __eq__(self, other: object) -> bool:
        # Unordered comparison
        if not isinstance(other, TestResults) or len(self) != len(other):
            return False
        
        # Increase recursion limit only if necessary
        original_recursion_limit = sys.getrecursionlimit()
        if original_recursion_limit < 5000:
            sys.setrecursionlimit(5000)
        
        for test_result in self:
            other_test_result = other.get_by_id(test_result.id)
            if other_test_result is None or not (
                test_result.file_name == other_test_result.file_name and 
                test_result.did_pass == other_test_result.did_pass and 
                test_result.runtime == other_test_result.runtime and 
                test_result.test_framework == other_test_result.test_framework and 
                test_result.test_type == other_test_result.test_type and 
                comparator(test_result.return_value, other_test_result.return_value)
            ):
                sys.setrecursionlimit(original_recursion_limit)
                return False
        
        sys.setrecursionlimit(original_recursion_limit)
        return True
    
    def get_test_pass_fail_report_by_type(self) -> dict[TestType, dict[str, int]]:
        report = {test_type: {"passed": 0, "failed": 0} for test_type in TestType}
        for test_result in self.test_results:
            if test_result.test_type != TestType.EXISTING_UNIT_TEST or test_result.id.function_getting_tested:
                key = "passed" if test_result.did_pass else "failed"
                report[test_result.test_type][key] += 1
        return report"""

    preexisting_objects = [
        ("__contains__", [FunctionParent(name="TestResults", type="ClassDef")]),
        ("__len__", [FunctionParent(name="TestResults", type="ClassDef")]),
        ("__bool__", [FunctionParent(name="TestResults", type="ClassDef")]),
        ("__eq__", [FunctionParent(name="TestResults", type="ClassDef")]),
        ("__delitem__", [FunctionParent(name="TestResults", type="ClassDef")]),
        ("__iter__", [FunctionParent(name="TestResults", type="ClassDef")]),
        ("__setitem__", [FunctionParent(name="TestResults", type="ClassDef")]),
        ("__getitem__", [FunctionParent(name="TestResults", type="ClassDef")]),
        ("get_test_pass_fail_report_by_type", [FunctionParent(name="TestResults", type="ClassDef")]),
        ("TestType", []),
        ("TestResults", []),
        ("to_name", [FunctionParent(name="TestType", type="ClassDef")]),
    ]

    contextual_functions = {
        ("TestResults", "__bool__"),
        ("TestResults", "__contains__"),
        ("TestResults", "__delitem__"),
        ("TestResults", "__eq__"),
        ("TestResults", "__getitem__"),
        ("TestResults", "__iter__"),
        ("TestResults", "__len__"),
        ("TestResults", "__setitem__"),
    }

    helper_functions = [
        FakeFunctionSource(
            file_path="/Users/saurabh/Library/CloudStorage/Dropbox/codeflash/cli/codeflash/verification/test_results.py",
            qualified_name="TestType",
            fully_qualified_name="codeflash.verification.test_results.TestType",
            only_function_name="TestType",
            source_code="",
            jedi_definition=JediDefinition(type="class"),
        ),
    ]

    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=["TestResults.get_test_pass_fail_report_by_type"],
        optimized_code=optim_code,
        file_path_of_module_with_function_to_optimize=str(Path(__file__).resolve()),
        module_abspath=str(Path(__file__).resolve()),
        preexisting_objects=preexisting_objects,
        contextual_functions=contextual_functions,
        project_root_path=str(Path(__file__).parent.resolve()),
    )

    helper_functions_by_module_abspath = defaultdict(set)
    for helper_function in helper_functions:
        if helper_function.jedi_definition.type != "class":
            helper_functions_by_module_abspath[helper_function.file_path].add(
                helper_function.qualified_name,
            )
    for (
        module_abspath,
        qualified_names,
    ) in helper_functions_by_module_abspath.items():
        new_code: str = replace_functions_and_add_imports(
            source_code=new_code,
            function_names=list(qualified_names),
            optimized_code=optim_code,
            file_path_of_module_with_function_to_optimize=str(Path(__file__).resolve()),
            module_abspath=module_abspath,
            preexisting_objects=preexisting_objects,
            contextual_functions=contextual_functions,
            project_root_path=str(Path(__file__).parent.resolve()),
        )

    assert (
        new_code
        == """from __future__ import annotations
import sys
from codeflash.verification.comparator import comparator
from enum import Enum
from pydantic import BaseModel
from typing import Iterator

class TestType(Enum):
    EXISTING_UNIT_TEST = 1
    INSPIRED_REGRESSION = 2
    GENERATED_REGRESSION = 3
    REPLAY_TEST = 4

    def to_name(self) -> str:
        names = {
            TestType.EXISTING_UNIT_TEST: "⚙️ Existing Unit Tests",
            TestType.INSPIRED_REGRESSION: "🎨 Inspired Regression Tests",
            TestType.GENERATED_REGRESSION: "🌀 Generated Regression Tests",
            TestType.REPLAY_TEST: "⏪ Replay Tests",
        }
        return names[self]

class TestResults(BaseModel):
    def __iter__(self) -> Iterator[FunctionTestInvocation]:
        return iter(self.test_results)
    def __len__(self) -> int:
        return len(self.test_results)
    def __getitem__(self, index: int) -> FunctionTestInvocation:
        return self.test_results[index]
    def __setitem__(self, index: int, value: FunctionTestInvocation) -> None:
        self.test_results[index] = value
    def __delitem__(self, index: int) -> None:
        del self.test_results[index]
    def __contains__(self, value: FunctionTestInvocation) -> bool:
        return value in self.test_results
    def __bool__(self) -> bool:
        return bool(self.test_results)
    def __eq__(self, other: object) -> bool:
        # Unordered comparison
        if type(self) != type(other):
            return False
        if len(self) != len(other):
            return False
        original_recursion_limit = sys.getrecursionlimit()
        for test_result in self:
            other_test_result = other.get_by_id(test_result.id)
            if other_test_result is None:
                return False

            if original_recursion_limit < 5000:
                sys.setrecursionlimit(5000)
            if (
                test_result.file_name != other_test_result.file_name
                or test_result.did_pass != other_test_result.did_pass
                or test_result.runtime != other_test_result.runtime
                or test_result.test_framework != other_test_result.test_framework
                or test_result.test_type != other_test_result.test_type
                or not comparator(
                    test_result.return_value,
                    other_test_result.return_value,
                )
            ):
                sys.setrecursionlimit(original_recursion_limit)
                return False
        sys.setrecursionlimit(original_recursion_limit)
        return True
    
    def get_test_pass_fail_report_by_type(self) -> dict[TestType, dict[str, int]]:
        report = {test_type: {"passed": 0, "failed": 0} for test_type in TestType}
        for test_result in self.test_results:
            if test_result.test_type != TestType.EXISTING_UNIT_TEST or test_result.id.function_getting_tested:
                key = "passed" if test_result.did_pass else "failed"
                report[test_result.test_type][key] += 1
        return report"""
    )


def test_code_replacement_type_annotation() -> None:
    original_code = '''import numpy as np
from pydantic.dataclasses import dataclass
from typing import List, Optional, Tuple, Union
@dataclass(config=dict(arbitrary_types_allowed=True))
class Matrix:
    data: Union[List[List[float]], List[np.ndarray], np.ndarray]
def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X.data) == 0 or len(Y.data) == 0:
        return np.array([])
    X = np.array(X.data)
    Y = np.array(Y.data)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}.",
        )
    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity
def cosine_similarity_top_k(
    X: Matrix,
    Y: Matrix,
    top_k: Optional[int] = 5,
    score_threshold: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """Row-wise cosine similarity with optional top-k and score threshold filtering.
    Args:
    ----
        X: Matrix.
        Y: Matrix, same width as X.
        top_k: Max number of results to return.
        score_threshold: Minimum cosine similarity of results.
    Returns:
    -------
        Tuple of two lists. First contains two-tuples of indices (X_idx, Y_idx),
            second contains corresponding cosine similarities.
    """
    if len(X.data) == 0 or len(Y.data) == 0:
        return [], []
    score_array = cosine_similarity(X, Y)
    sorted_idxs = score_array.flatten().argsort()[::-1]
    top_k = top_k or len(sorted_idxs)
    top_idxs = sorted_idxs[:top_k]
    score_threshold = score_threshold or -1.0
    top_idxs = top_idxs[score_array.flatten()[top_idxs] > score_threshold]
    ret_idxs = [(x // score_array.shape[1], x % score_array.shape[1]) for x in top_idxs]
    scores = score_array.flatten()[top_idxs].tolist()
    return ret_idxs, scores
'''
    optim_code = '''from typing import List, Optional, Tuple, Union
import numpy as np
from pydantic.dataclasses import dataclass
@dataclass(config=dict(arbitrary_types_allowed=True))
class Matrix:
    data: Union[list[list[float]], List[np.ndarray], np.ndarray]
def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X.data) == 0 or len(Y.data) == 0:
        return np.array([])
    
    X_np, Y_np = np.asarray(X.data), np.asarray(Y.data)
    if X_np.shape[1] != Y_np.shape[1]:
        raise ValueError(f"Number of columns in X and Y must be the same. X has shape {X_np.shape} and Y has shape {Y_np.shape}.")
    X_norm = np.linalg.norm(X_np, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y_np, axis=1, keepdims=True)
    
    norm_product = X_norm * Y_norm.T
    norm_product[norm_product == 0] = np.inf  # Prevent division by zero
    dot_product = np.dot(X_np, Y_np.T)
    similarity = dot_product / norm_product
    
    # Any NaN or Inf values are set to 0.0
    np.nan_to_num(similarity, copy=False)
    
    return similarity
def cosine_similarity_top_k(
    X: Matrix,
    Y: Matrix,
    top_k: Optional[int] = 5,
    score_threshold: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """Row-wise cosine similarity with optional top-k and score threshold filtering."""
    if len(X.data) == 0 or len(Y.data) == 0:
        return [], []
    
    score_array = cosine_similarity(X, Y)
    
    sorted_idxs = np.argpartition(-score_array.flatten(), range(top_k or len(score_array.flatten())))[:(top_k or len(score_array.flatten()))]
    sorted_idxs = sorted_idxs[score_array.flatten()[sorted_idxs] > (score_threshold if score_threshold is not None else -1)]
    
    ret_idxs = [(x // score_array.shape[1], x % score_array.shape[1]) for x in sorted_idxs]
    scores = score_array.flatten()[sorted_idxs].tolist()
    
    return ret_idxs, scores
'''
    preexisting_objects = [("cosine_similarity_top_k", []), ("Matrix", []), ("cosine_similarity", [])]

    contextual_functions = set()
    helper_functions = [
        FakeFunctionSource(
            file_path=str((Path(__file__).parent / "code_to_optimize" / "math_utils.py").resolve()),
            qualified_name="Matrix",
            fully_qualified_name="code_to_optimize.math_utils.Matrix",
            only_function_name="Matrix",
            source_code="",
            jedi_definition=JediDefinition(type="class"),
        ),
        FakeFunctionSource(
            file_path=str((Path(__file__).parent / "code_to_optimize" / "math_utils.py").resolve()),
            qualified_name="cosine_similarity",
            fully_qualified_name="code_to_optimize.math_utils.cosine_similarity",
            only_function_name="cosine_similarity",
            source_code="",
            jedi_definition=JediDefinition(type="function"),
        ),
    ]

    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=["cosine_similarity_top_k"],
        optimized_code=optim_code,
        file_path_of_module_with_function_to_optimize=str(Path(__file__).resolve()),
        module_abspath=str((Path(__file__).parent / "code_to_optimize").resolve()),
        preexisting_objects=preexisting_objects,
        contextual_functions=contextual_functions,
        project_root_path=str(Path(__file__).parent.parent.resolve()),
    )
    assert (
        new_code
        == '''import numpy as np
from pydantic.dataclasses import dataclass
from typing import List, Optional, Tuple, Union
@dataclass(config=dict(arbitrary_types_allowed=True))
class Matrix:
    data: Union[List[List[float]], List[np.ndarray], np.ndarray]
def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X.data) == 0 or len(Y.data) == 0:
        return np.array([])
    X = np.array(X.data)
    Y = np.array(Y.data)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}.",
        )
    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity
def cosine_similarity_top_k(
    X: Matrix,
    Y: Matrix,
    top_k: Optional[int] = 5,
    score_threshold: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """Row-wise cosine similarity with optional top-k and score threshold filtering."""
    if len(X.data) == 0 or len(Y.data) == 0:
        return [], []
    
    score_array = cosine_similarity(X, Y)
    
    sorted_idxs = np.argpartition(-score_array.flatten(), range(top_k or len(score_array.flatten())))[:(top_k or len(score_array.flatten()))]
    sorted_idxs = sorted_idxs[score_array.flatten()[sorted_idxs] > (score_threshold if score_threshold is not None else -1)]
    
    ret_idxs = [(x // score_array.shape[1], x % score_array.shape[1]) for x in sorted_idxs]
    scores = score_array.flatten()[sorted_idxs].tolist()
    
    return ret_idxs, scores
'''
    )
    helper_functions_by_module_abspath = defaultdict(set)
    for helper_function in helper_functions:
        if helper_function.jedi_definition.type != "class":
            helper_functions_by_module_abspath[helper_function.file_path].add(
                helper_function.qualified_name,
            )
    for (
        module_abspath,
        qualified_names,
    ) in helper_functions_by_module_abspath.items():
        new_helper_code: str = replace_functions_and_add_imports(
            source_code=new_code,
            function_names=list(qualified_names),
            optimized_code=optim_code,
            file_path_of_module_with_function_to_optimize=str(Path(__file__).resolve()),
            module_abspath=module_abspath,
            preexisting_objects=preexisting_objects,
            contextual_functions=contextual_functions,
            project_root_path=str(Path(__file__).parent.parent.resolve()),
        )

    assert (
        new_helper_code
        == '''import numpy as np
from pydantic.dataclasses import dataclass
from typing import List, Optional, Tuple, Union
@dataclass(config=dict(arbitrary_types_allowed=True))
class Matrix:
    data: Union[List[List[float]], List[np.ndarray], np.ndarray]
def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X.data) == 0 or len(Y.data) == 0:
        return np.array([])
    
    X_np, Y_np = np.asarray(X.data), np.asarray(Y.data)
    if X_np.shape[1] != Y_np.shape[1]:
        raise ValueError(f"Number of columns in X and Y must be the same. X has shape {X_np.shape} and Y has shape {Y_np.shape}.")
    X_norm = np.linalg.norm(X_np, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y_np, axis=1, keepdims=True)
    
    norm_product = X_norm * Y_norm.T
    norm_product[norm_product == 0] = np.inf  # Prevent division by zero
    dot_product = np.dot(X_np, Y_np.T)
    similarity = dot_product / norm_product
    
    # Any NaN or Inf values are set to 0.0
    np.nan_to_num(similarity, copy=False)
    
    return similarity
def cosine_similarity_top_k(
    X: Matrix,
    Y: Matrix,
    top_k: Optional[int] = 5,
    score_threshold: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """Row-wise cosine similarity with optional top-k and score threshold filtering."""
    if len(X.data) == 0 or len(Y.data) == 0:
        return [], []
    
    score_array = cosine_similarity(X, Y)
    
    sorted_idxs = np.argpartition(-score_array.flatten(), range(top_k or len(score_array.flatten())))[:(top_k or len(score_array.flatten()))]
    sorted_idxs = sorted_idxs[score_array.flatten()[sorted_idxs] > (score_threshold if score_threshold is not None else -1)]
    
    ret_idxs = [(x // score_array.shape[1], x % score_array.shape[1]) for x in sorted_idxs]
    scores = score_array.flatten()[sorted_idxs].tolist()
    
    return ret_idxs, scores
'''
    )


def test_future_aliased_imports_removal() -> None:
    module_code1 = """from __future__ import annotations as _annotations
print("Hello monde")
"""

    expected_code1 = """print("Hello monde")
"""

    assert remove_first_imported_aliased_objects(module_code1, "__future__")[0] == expected_code1

    module_code2 = """from __future__ import annotations
print("Hello monde")
"""

    assert remove_first_imported_aliased_objects(module_code2, "__future__")[0] == module_code2

    module_code3 = """from __future__ import annotations as _annotations
from __future__ import annotations
from past import autopasta as dood
print("Hello monde")
"""

    expected_code3 = """from __future__ import annotations
from past import autopasta as dood
print("Hello monde")
"""

    assert remove_first_imported_aliased_objects(module_code3, "__future__")[0] == expected_code3

    module_code4 = """from __future__ import annotations
from __future__ import annotations  as _annotations
from past import autopasta as dood
print("Hello monde")
"""

    assert remove_first_imported_aliased_objects(module_code4, "__future__")[0] == module_code4

    module_code5 = """from future import annotations as _annotations
from __future__ import annotations  as _annotations
from past import autopasta as dood
print("Hello monde")
"""

    assert remove_first_imported_aliased_objects(module_code5, "__future__")[0] == module_code5
