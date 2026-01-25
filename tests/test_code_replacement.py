from __future__ import annotations
import re
import libcst as cst
from codeflash.code_utils.code_replacer import AutouseFixtureModifier, PytestMarkAdder, AddRequestArgument
import dataclasses
import os
from collections import defaultdict
from pathlib import Path

from codeflash.code_utils.code_extractor import delete___future___aliased_imports, find_preexisting_objects
from codeflash.code_utils.code_replacer import (
    is_zero_diff,
    replace_functions_and_add_imports,
    replace_functions_in_file,
    OptimFunctionCollector,
)
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import CodeOptimizationContext, CodeStringsMarkdown, FunctionParent
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.verification_utils import TestConfig

os.environ["CODEFLASH_API_KEY"] = "cf-test-key"


@dataclasses.dataclass
class JediDefinition:
    type: str


@dataclasses.dataclass
class FakeFunctionSource:
    file_path: Path
    qualified_name: str
    fully_qualified_name: str
    only_function_name: str
    source_code: str
    jedi_definition: JediDefinition


class Args:
    disable_imports_sorting = True
    formatter_cmds = ["disabled"]


def test_code_replacement_global_statements():
    project_root = Path(__file__).parent.parent.resolve()
    code_path = (project_root / "code_to_optimize/bubble_sort_optimized.py").resolve()
    optimized_code = f"""```python:{code_path.relative_to(project_root)}
import numpy as np

inconsequential_var = '123'
def sorter(arr):
    return arr.sort()
```
"""
    original_code_str = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort.py").read_text(
        encoding="utf-8"
    )
    code_path.write_text(original_code_str, encoding="utf-8")
    tests_root = Path("/Users/codeflash/Downloads/codeflash-dev/codeflash/code_to_optimize/tests/pytest/")
    project_root_path = (Path(__file__).parent / "..").resolve()
    func = FunctionToOptimize(function_name="sorter", parents=[], file_path=code_path)
    test_config = TestConfig(
        tests_root=tests_root,
        tests_project_rootdir=project_root_path,
        project_root_path=project_root_path,
        test_framework="pytest",
        pytest_cmd="pytest",
    )
    func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
    code_context: CodeOptimizationContext = func_optimizer.get_code_optimization_context().unwrap()
    original_helper_code: dict[Path, str] = {}
    helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
    for helper_function_path in helper_function_paths:
        with helper_function_path.open(encoding="utf8") as f:
            helper_code = f.read()
            original_helper_code[helper_function_path] = helper_code
    func_optimizer.args = Args()
    func_optimizer.replace_function_and_helpers_with_optimized_code(
        code_context=code_context, optimized_code=CodeStringsMarkdown.parse_markdown_code(optimized_code), original_helper_code=original_helper_code
    )
    final_output = code_path.read_text(encoding="utf-8")
    assert "inconsequential_var = '123'" in final_output
    code_path.unlink(missing_ok=True)


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
    @staticmethod
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
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]] = find_preexisting_objects(original_code)
    print(f"Preexisting objects: {preexisting_objects}")
    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=[function_name],
        optimized_code=optim_code,
        module_abspath=Path(__file__).resolve(),
        preexisting_objects=preexisting_objects,
        project_root_path=Path(__file__).resolve().parent.resolve(),
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

def other_function(st):
    return(st * 2)

print("Hello world")
"""

    function_name: str = "NewClass.new_function"
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]] = find_preexisting_objects(original_code)
    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=[function_name],
        optimized_code=optim_code,
        module_abspath=Path(__file__).resolve(),
        preexisting_objects=preexisting_objects,
        project_root_path=Path(__file__).resolve().parent.resolve(),
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
    expected = """import libcst as cst
from typing import Mandatory

class NewClass:
    def __init__(self, name):
        self.name = name
    def new_function(self, value: cst.Name):
        return other_function(self.name)
    def new_function2(value):
        return value

print("Au revoir")

def yet_another_function(values):
    return len(values)

def totally_new_function(value):
    return value

def other_function(st):
    return(st * 2)

print("Salut monde")
"""

    function_names: list[str] = ["other_function"]
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]] = find_preexisting_objects(original_code)
    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=function_names,
        optimized_code=optim_code,
        module_abspath=Path(__file__).resolve(),
        preexisting_objects=preexisting_objects,
        project_root_path=Path(__file__).resolve().parent.resolve(),
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
    expected = """from typing import Mandatory

class NewClass:
    def __init__(self, name):
        self.name = name
    def new_function(self, value):
        return other_function(self.name)
    def new_function2(value):
        return value

print("Au revoir")

def yet_another_function(values):
    return len(values) + 2

def totally_new_function(value):
    return value

def other_function(st):
    return(st * 2)

print("Salut monde")
"""

    function_names: list[str] = ["yet_another_function", "other_function"]
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]] = find_preexisting_objects(original_code)
    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=function_names,
        optimized_code=optim_code,
        module_abspath=Path(__file__).resolve(),
        preexisting_objects=preexisting_objects,
        project_root_path=Path(__file__).resolve().parent.resolve(),
    )
    assert new_code == expected


def test_test_libcst_code_replacement5() -> None:
    optim_code = """@lru_cache(17)
def sorter_deps(arr: list[int]) -> list[int]:
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

@lru_cache(17)
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
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]] = find_preexisting_objects(original_code)
    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=function_names,
        optimized_code=optim_code,
        module_abspath=Path(__file__).resolve(),
        preexisting_objects=preexisting_objects,
        project_root_path=Path(__file__).resolve().parent.resolve(),
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

def blob(values):
    return(st * 2)

def blab(st):
    return(st + st)

print("Not cool")
"""
    preexisting_objects = find_preexisting_objects(original_code_main) | find_preexisting_objects(original_code_helper)
    new_main_code: str = replace_functions_and_add_imports(
        source_code=original_code_main,
        function_names=["other_function"],
        optimized_code=optim_code,
        module_abspath=Path(__file__).resolve(),
        preexisting_objects=preexisting_objects,
        project_root_path=Path(__file__).resolve().parent.resolve(),
    )
    assert new_main_code == expected_main

    new_helper_code: str = replace_functions_and_add_imports(
        source_code=original_code_helper,
        function_names=["blob"],
        optimized_code=optim_code,
        module_abspath=Path(__file__).resolve(),
        preexisting_objects=preexisting_objects,
        project_root_path=Path(__file__).resolve().parent.resolve(),
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
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]] = find_preexisting_objects(original_code)

    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=function_names,
        optimized_code=optim_code,
        module_abspath=Path(__file__).resolve(),
        preexisting_objects=preexisting_objects,
        project_root_path=Path(__file__).resolve().parent.resolve(),
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
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]] = find_preexisting_objects(original_code)
    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=function_names,
        optimized_code=optim_code,
        module_abspath=Path(__file__).resolve(),
        preexisting_objects=preexisting_objects,
        project_root_path=Path(__file__).resolve().parent.resolve(),
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
    function_name: str = "NewClass.__init__"
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]] = find_preexisting_objects(original_code)
    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=[function_name],
        optimized_code=optim_code,
        module_abspath=Path(__file__).resolve(),
        preexisting_objects=preexisting_objects,
        project_root_path=Path(__file__).resolve().parent.resolve(),
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
    get_code_output = """# file: test_code_replacement.py
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
    file_path = Path(__file__).resolve()
    func_top_optimize = FunctionToOptimize(
        function_name="main_method", file_path=file_path, parents=[FunctionParent("MainClass", "ClassDef")]
    )
    test_config = TestConfig(
        tests_root=file_path.parent,
        tests_project_rootdir=file_path.parent,
        project_root_path=file_path.parent,
        test_framework="pytest",
        pytest_cmd="pytest",
    )
    func_optimizer = FunctionOptimizer(function_to_optimize=func_top_optimize, test_cfg=test_config)
    code_context = func_optimizer.get_code_optimization_context().unwrap()
    assert code_context.testgen_context.flat.rstrip() == get_code_output.rstrip()


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
    parents = (FunctionParent("Fu", "ClassDef"),)
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]] = {("foo", parents), ("real_bar", parents)}
    new_code: str = replace_functions_in_file(
        source_code=original_code,
        original_function_names=[function_name],
        optimized_code=optim_code,
        preexisting_objects=preexisting_objects,
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

    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]] = []
    new_code: str = replace_functions_in_file(
        source_code=original_code,
        original_function_names=["Fu.real_bar"],
        optimized_code=optim_code,
        preexisting_objects=preexisting_objects,
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
        self.new_attribute = "Sorry i modified a dunder method"
    def new_function(self, value):
        return other_function(self.name)
    def new_function2(value):
        return value
    def __call__(self, value):
        return self.name
"""

    function_names: list[str] = ["yet_another_function", "other_function"]
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]] = []
    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=function_names,
        optimized_code=optim_code,
        module_abspath=Path(__file__).resolve(),
        preexisting_objects=preexisting_objects,
        project_root_path=Path(__file__).resolve().parent.resolve(),
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

    preexisting_objects = find_preexisting_objects(original_code)

    helper_functions = [
        FakeFunctionSource(
            file_path=Path(
                "/Users/saurabh/Library/CloudStorage/Dropbox/codeflash/cli/codeflash/verification/test_results.py"
            ),
            qualified_name="TestType",
            fully_qualified_name="codeflash.verification.test_results.TestType",
            only_function_name="TestType",
            source_code="",
            jedi_definition=JediDefinition(type="class"),
        )
    ]

    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=["TestResults.get_test_pass_fail_report_by_type"],
        optimized_code=optim_code,
        module_abspath=Path(__file__).resolve(),
        preexisting_objects=preexisting_objects,
        project_root_path=Path(__file__).parent.resolve(),
    )

    helper_functions_by_module_abspath = defaultdict(set)
    for helper_function in helper_functions:
        if helper_function.jedi_definition.type != "class":
            helper_functions_by_module_abspath[helper_function.file_path].add(helper_function.qualified_name)
    for module_abspath, qualified_names in helper_functions_by_module_abspath.items():
        new_code: str = replace_functions_and_add_imports(
            source_code=new_code,
            function_names=list(qualified_names),
            optimized_code=optim_code,
            module_abspath=module_abspath,
            preexisting_objects=preexisting_objects,
            project_root_path=Path(__file__).parent.resolve(),
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
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]] = find_preexisting_objects(original_code)

    helper_functions = [
        FakeFunctionSource(
            file_path=(Path(__file__).parent / "code_to_optimize" / "math_utils.py").resolve(),
            qualified_name="Matrix",
            fully_qualified_name="code_to_optimize.math_utils.Matrix",
            only_function_name="Matrix",
            source_code="",
            jedi_definition=JediDefinition(type="class"),
        ),
        FakeFunctionSource(
            file_path=(Path(__file__).parent / "code_to_optimize" / "math_utils.py").resolve(),
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
        module_abspath=(Path(__file__).parent / "code_to_optimize").resolve(),
        preexisting_objects=preexisting_objects,
        project_root_path=Path(__file__).parent.parent.resolve(),
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
            helper_functions_by_module_abspath[helper_function.file_path].add(helper_function.qualified_name)
    for module_abspath, qualified_names in helper_functions_by_module_abspath.items():
        new_helper_code: str = replace_functions_and_add_imports(
            source_code=new_code,
            function_names=list(qualified_names),
            optimized_code=optim_code,
            module_abspath=module_abspath,
            preexisting_objects=preexisting_objects,
            project_root_path=Path(__file__).parent.parent.resolve(),
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

    assert delete___future___aliased_imports(module_code1) == expected_code1

    module_code2 = """from __future__ import annotations
print("Hello monde")
"""

    assert delete___future___aliased_imports(module_code2) == module_code2

    module_code3 = """from __future__ import annotations as _annotations
from __future__ import annotations
from past import autopasta as dood
print("Hello monde")
"""

    expected_code3 = """from __future__ import annotations
from past import autopasta as dood
print("Hello monde")
"""

    assert delete___future___aliased_imports(module_code3) == expected_code3

    module_code4 = """from __future__ import annotations
from __future__ import annotations  as _annotations
from past import autopasta as dood
print("Hello monde")
"""

    expected_module_code4 = """from __future__ import annotations
from past import autopasta as dood
print("Hello monde")
"""

    assert delete___future___aliased_imports(module_code4) == expected_module_code4

    module_code5 = """from future import annotations as _annotations
from past import autopasta as dood
print("Hello monde")
"""

    assert delete___future___aliased_imports(module_code5) == module_code5

    module_code6 = '''"""Private logic for creating models."""

from __future__ import annotations as _annotations
'''
    expected_code6 = '''"""Private logic for creating models."""
'''

    assert delete___future___aliased_imports(module_code6) == expected_code6


def test_0_diff_code_replacement():
    original_code = """from __future__ import annotations

import numpy as np
def functionA():
    return np.array([1, 2, 3])
"""
    optim_code_a = """from __future__ import annotations
import numpy as np
def functionA():
    return np.array([1, 2, 3])"""

    assert is_zero_diff(original_code, optim_code_a)

    optim_code_b = """
import numpy as np
def functionA():
    return np.array([1, 2, 3])"""

    assert is_zero_diff(original_code, optim_code_b)

    optim_code_c = """
def functionA():
    return np.array([1, 2, 3])"""

    assert is_zero_diff(original_code, optim_code_c)

    optim_code_d = """from __future__ import annotations

import numpy as np
def functionA():
    return np.array([1, 2, 3, 4])
"""
    assert not is_zero_diff(original_code, optim_code_d)

    optim_code_e = '''"""
Zis a Docstring?
"""
from __future__ import annotations

import ast
def functionA():
    """
    Und Zis?
    """
    import numpy as np
    return np.array([1, 2, 3])
    '''
    assert is_zero_diff(original_code, optim_code_e)


def test_nested_class() -> None:
    optim_code = """import libcst as cst
from typing import Optional

class NewClass:
    def __init__(self, name):
        self.name = str(name)
    def __call__(self, value):
        return self.name
    def new_function2(value):
        return cst.ensure_type(value, int)

    class NestedClass:
        def nested_function(self):
            return "I am nested and modified"
    """

    original_code = """class NewClass:
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)

    class NestedClass:
        def nested_function(self):
            return "I am nested"

print("Hello world")
"""
    expected = """import libcst as cst

class NewClass:
    def __init__(self, name):
        self.name = str(name)
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, int)

    class NestedClass:
        def nested_function(self):
            return "I am nested"

print("Hello world")
"""

    function_names: list[str] = [
        "NewClass.new_function2",
        "NestedClass.nested_function",
    ]  # Nested classes should be ignored, even if provided as target
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]] = find_preexisting_objects(original_code)
    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=function_names,
        optimized_code=optim_code,
        module_abspath=Path(__file__).resolve(),
        preexisting_objects=preexisting_objects,
        project_root_path=Path(__file__).resolve().parent.resolve(),
    )
    assert new_code == expected


def test_modify_back_to_original() -> None:
    optim_code = """class NewClass:
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)

print("Hello world")
"""

    original_code = """class NewClass:
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)

print("Hello world")
"""
    function_names: list[str] = ["NewClass.__init__", "NewClass.__call__", "NewClass.new_function2"]
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]] = find_preexisting_objects(original_code)
    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=function_names,
        optimized_code=optim_code,
        module_abspath=Path(__file__).resolve(),
        preexisting_objects=preexisting_objects,
        project_root_path=Path(__file__).resolve().parent.resolve(),
    )
    assert new_code == original_code


def test_global_reassignment() -> None:
    root_dir = Path(__file__).parent.parent.resolve()
    code_path = (root_dir / "code_to_optimize/global_var_original.py").resolve()

    original_code = """a=1
print("Hello world")
def some_fn():
    print("did noting")
class NewClass:
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
    """
    optimized_code = f"""```python:{code_path.relative_to(root_dir)}
import numpy as np

def some_fn():
    a=np.zeros(10)
    print("did something")
class NewClass:
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
a=2
print("Hello world")
```
"""
    expected_code = """import numpy as np

a=2
print("Hello world")
def some_fn():
    a=np.zeros(10)
    print("did something")
class NewClass:
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)"""
    code_path.write_text(original_code, encoding="utf-8")
    tests_root = Path("/Users/codeflash/Downloads/codeflash-dev/codeflash/code_to_optimize/tests/pytest/")
    project_root_path = (Path(__file__).parent / "..").resolve()
    func = FunctionToOptimize(function_name="some_fn", parents=[], file_path=code_path)
    test_config = TestConfig(
        tests_root=tests_root,
        tests_project_rootdir=project_root_path,
        project_root_path=project_root_path,
        test_framework="pytest",
        pytest_cmd="pytest",
    )
    func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
    code_context: CodeOptimizationContext = func_optimizer.get_code_optimization_context().unwrap()
    original_helper_code: dict[Path, str] = {}
    helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
    for helper_function_path in helper_function_paths:
        with helper_function_path.open(encoding="utf8") as f:
            helper_code = f.read()
            original_helper_code[helper_function_path] = helper_code
    func_optimizer.args = Args()
    func_optimizer.replace_function_and_helpers_with_optimized_code(
        code_context=code_context, optimized_code=CodeStringsMarkdown.parse_markdown_code(optimized_code), original_helper_code=original_helper_code
    )
    new_code = code_path.read_text(encoding="utf-8")
    code_path.unlink(missing_ok=True)
    assert new_code.rstrip() == expected_code.rstrip()

    original_code = """print("Hello world")
def some_fn():
    print("did noting")
class NewClass:
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
a=1
"""
    optimized_code = f"""```python:{code_path.relative_to(root_dir)}
a=2
import numpy as np
def some_fn():
    a=np.zeros(10)
    print("did something")
class NewClass:
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
print("Hello world")
```
"""
    expected_code = """import numpy as np

print("Hello world")
def some_fn():
    a=np.zeros(10)
    print("did something")
class NewClass:
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
a=2    
"""
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/global_var_original.py").resolve()
    code_path.write_text(original_code, encoding="utf-8")
    tests_root = Path("/Users/codeflash/Downloads/codeflash-dev/codeflash/code_to_optimize/tests/pytest/")
    project_root_path = (Path(__file__).parent / "..").resolve()
    func = FunctionToOptimize(function_name="some_fn", parents=[], file_path=code_path)
    test_config = TestConfig(
        tests_root=tests_root,
        tests_project_rootdir=project_root_path,
        project_root_path=project_root_path,
        test_framework="pytest",
        pytest_cmd="pytest",
    )
    func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
    code_context: CodeOptimizationContext = func_optimizer.get_code_optimization_context().unwrap()
    original_helper_code: dict[Path, str] = {}
    helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
    for helper_function_path in helper_function_paths:
        with helper_function_path.open(encoding="utf8") as f:
            helper_code = f.read()
            original_helper_code[helper_function_path] = helper_code
    func_optimizer.args = Args()
    func_optimizer.replace_function_and_helpers_with_optimized_code(
        code_context=code_context, optimized_code=CodeStringsMarkdown.parse_markdown_code(optimized_code), original_helper_code=original_helper_code
    )
    new_code = code_path.read_text(encoding="utf-8")
    code_path.unlink(missing_ok=True)
    assert new_code.rstrip() == expected_code.rstrip()

    original_code = """a=1
print("Hello world")
def some_fn():
    print("did noting")
class NewClass:
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
"""
    optimized_code = f"""```python:{code_path.relative_to(root_dir)}
import numpy as np
a=2
def some_fn():
    a=np.zeros(10)
    print("did something")
class NewClass:
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
a=3
print("Hello world")
```
"""
    expected_code = """import numpy as np

a=3
print("Hello world")
def some_fn():
    a=np.zeros(10)
    print("did something")
class NewClass:
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
"""
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/global_var_original.py").resolve()
    code_path.write_text(original_code, encoding="utf-8")
    tests_root = Path("/Users/codeflash/Downloads/codeflash-dev/codeflash/code_to_optimize/tests/pytest/")
    project_root_path = (Path(__file__).parent / "..").resolve()
    func = FunctionToOptimize(function_name="some_fn", parents=[], file_path=code_path)
    test_config = TestConfig(
        tests_root=tests_root,
        tests_project_rootdir=project_root_path,
        project_root_path=project_root_path,
        test_framework="pytest",
        pytest_cmd="pytest",
    )
    func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
    code_context: CodeOptimizationContext = func_optimizer.get_code_optimization_context().unwrap()
    original_helper_code: dict[Path, str] = {}
    helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
    for helper_function_path in helper_function_paths:
        with helper_function_path.open(encoding="utf8") as f:
            helper_code = f.read()
            original_helper_code[helper_function_path] = helper_code
    func_optimizer.args = Args()
    func_optimizer.replace_function_and_helpers_with_optimized_code(
        code_context=code_context, optimized_code=CodeStringsMarkdown.parse_markdown_code(optimized_code), original_helper_code=original_helper_code
    )
    new_code = code_path.read_text(encoding="utf-8")
    code_path.unlink(missing_ok=True)
    assert new_code.rstrip() == expected_code.rstrip()

    original_code = """a=1
print("Hello world")
def some_fn():
    print("did noting")
class NewClass:
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
"""
    optimized_code = f"""```python:{code_path.relative_to(root_dir)}
a=2
import numpy as np
def some_fn():
    a=np.zeros(10)
    print("did something")
class NewClass:
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
print("Hello world")
```
"""
    expected_code = """import numpy as np

a=2
print("Hello world")
def some_fn():
    a=np.zeros(10)
    print("did something")
class NewClass:
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
"""
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/global_var_original.py").resolve()
    code_path.write_text(original_code, encoding="utf-8")
    tests_root = Path("/Users/codeflash/Downloads/codeflash-dev/codeflash/code_to_optimize/tests/pytest/")
    project_root_path = (Path(__file__).parent / "..").resolve()
    func = FunctionToOptimize(function_name="some_fn", parents=[], file_path=code_path)
    test_config = TestConfig(
        tests_root=tests_root,
        tests_project_rootdir=project_root_path,
        project_root_path=project_root_path,
        test_framework="pytest",
        pytest_cmd="pytest",
    )
    func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
    code_context: CodeOptimizationContext = func_optimizer.get_code_optimization_context().unwrap()
    original_helper_code: dict[Path, str] = {}
    helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
    for helper_function_path in helper_function_paths:
        with helper_function_path.open(encoding="utf8") as f:
            helper_code = f.read()
            original_helper_code[helper_function_path] = helper_code
    func_optimizer.args = Args()
    func_optimizer.replace_function_and_helpers_with_optimized_code(
        code_context=code_context, optimized_code=CodeStringsMarkdown.parse_markdown_code(optimized_code), original_helper_code=original_helper_code
    )
    new_code = code_path.read_text(encoding="utf-8")
    code_path.unlink(missing_ok=True)
    assert new_code.rstrip() == expected_code.rstrip()

    original_code = """a=1
print("Hello world")
def some_fn():
    print("did noting")
class NewClass:
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
"""
    optimized_code = f"""```python:{code_path.relative_to(root_dir)}
import numpy as np
a=2
def some_fn():
    a=np.zeros(10)
    print("did something")
class NewClass:
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
a=3
print("Hello world")
```
"""
    expected_code = """import numpy as np

a=3
print("Hello world")
def some_fn():
    a=np.zeros(10)
    print("did something")
class NewClass:
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
"""
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/global_var_original.py").resolve()
    code_path.write_text(original_code, encoding="utf-8")
    tests_root = Path("/Users/codeflash/Downloads/codeflash-dev/codeflash/code_to_optimize/tests/pytest/")
    project_root_path = (Path(__file__).parent / "..").resolve()
    func = FunctionToOptimize(function_name="some_fn", parents=[], file_path=code_path)
    test_config = TestConfig(
        tests_root=tests_root,
        tests_project_rootdir=project_root_path,
        project_root_path=project_root_path,
        test_framework="pytest",
        pytest_cmd="pytest",
    )
    func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
    code_context: CodeOptimizationContext = func_optimizer.get_code_optimization_context().unwrap()
    original_helper_code: dict[Path, str] = {}
    helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
    for helper_function_path in helper_function_paths:
        with helper_function_path.open(encoding="utf8") as f:
            helper_code = f.read()
            original_helper_code[helper_function_path] = helper_code
    func_optimizer.args = Args()
    func_optimizer.replace_function_and_helpers_with_optimized_code(
        code_context=code_context, optimized_code=CodeStringsMarkdown.parse_markdown_code(optimized_code), original_helper_code=original_helper_code
    )
    new_code = code_path.read_text(encoding="utf-8")
    code_path.unlink(missing_ok=True)
    assert new_code.rstrip() == expected_code.rstrip()

    original_code = """if 2<3:
    a=4
else:
    a=5
print("Hello world")
def some_fn():
    print("did noting")
class NewClass:
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
"""
    optimized_code = f"""```python:{code_path.relative_to(root_dir)}
import numpy as np
if 1<2:
    a=2
else:
    a=3
a = 6    
def some_fn():
    a=np.zeros(10)
    print("did something")
class NewClass:
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
print("Hello world")
```
"""
    expected_code = """import numpy as np

a = 6
if 2<3:
    a=4
else:
    a=5
print("Hello world")
def some_fn():
    a=np.zeros(10)
    print("did something")
class NewClass:
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return "I am still old"
    def new_function2(value):
        return cst.ensure_type(value, str)
"""
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/global_var_original.py").resolve()
    code_path.write_text(original_code, encoding="utf-8")
    tests_root = Path("/Users/codeflash/Downloads/codeflash-dev/codeflash/code_to_optimize/tests/pytest/")
    project_root_path = (Path(__file__).parent / "..").resolve()
    func = FunctionToOptimize(function_name="some_fn", parents=[], file_path=code_path)
    test_config = TestConfig(
        tests_root=tests_root,
        tests_project_rootdir=project_root_path,
        project_root_path=project_root_path,
        test_framework="pytest",
        pytest_cmd="pytest",
    )
    func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
    code_context: CodeOptimizationContext = func_optimizer.get_code_optimization_context().unwrap()
    original_helper_code: dict[Path, str] = {}
    helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
    for helper_function_path in helper_function_paths:
        with helper_function_path.open(encoding="utf8") as f:
            helper_code = f.read()
            original_helper_code[helper_function_path] = helper_code
    func_optimizer.args = Args()
    func_optimizer.replace_function_and_helpers_with_optimized_code(
        code_context=code_context, optimized_code=CodeStringsMarkdown.parse_markdown_code(optimized_code), original_helper_code=original_helper_code
    )
    new_code = code_path.read_text(encoding="utf-8")
    code_path.unlink(missing_ok=True)
    assert new_code.rstrip() == expected_code.rstrip()


class TestAutouseFixtureModifier:
    """Test cases for AutouseFixtureModifier class."""

    def test_modifies_autouse_fixture_with_pytest_decorator(self):
        """Test that autouse fixture with @pytest.fixture is modified correctly."""
        source_code = '''
import pytest

@pytest.fixture(autouse=True)
def my_fixture(request):
    print("setup")
    yield
    print("teardown")
'''
        expected_code = '''
import pytest

@pytest.fixture(autouse=True)
def my_fixture(request):
    if request.node.get_closest_marker("codeflash_no_autouse"):
        yield
    else:
        print("setup")
        yield
        print("teardown")
'''
        module = cst.parse_module(source_code)
        modifier = AutouseFixtureModifier()
        modified_module = module.visit(modifier)

        # Parse expected to normalize formatting
        expected_module = cst.parse_module(expected_code)
        assert modified_module.code.strip() == expected_module.code.strip()

    def test_modifies_autouse_fixture_with_fixture_decorator(self):
        """Test that autouse fixture with @fixture is modified correctly."""
        source_code = '''
from pytest import fixture

@fixture(autouse=True)
def my_fixture(request):
    setup_code()
    yield "value"
    cleanup_code()
'''
        expected_code = '''
from pytest import fixture

@fixture(autouse=True)
def my_fixture(request):
    if request.node.get_closest_marker("codeflash_no_autouse"):
        yield
    else:
        setup_code()
        yield "value"
        cleanup_code()
'''
        module = cst.parse_module(source_code)
        modifier = AutouseFixtureModifier()
        modified_module = module.visit(modifier)

        # Check that the if statement was added
        assert modified_module.code.strip() == expected_code.strip()

    def test_ignores_non_autouse_fixture(self):
        """Test that non-autouse fixtures are not modified."""
        source_code = '''
import pytest

@pytest.fixture
def my_fixture(request):
    return "test_value"

@pytest.fixture(scope="session")
def session_fixture():
    return "session_value"
'''
        module = cst.parse_module(source_code)
        modifier = AutouseFixtureModifier()
        modified_module = module.visit(modifier)

        # Code should remain unchanged
        assert modified_module.code == source_code

    def test_ignores_regular_functions(self):
        """Test that regular functions are not modified."""
        source_code = '''
def regular_function():
    return "not a fixture"

@some_other_decorator
def decorated_function():
    return "also not a fixture"
'''
        module = cst.parse_module(source_code)
        modifier = AutouseFixtureModifier()
        modified_module = module.visit(modifier)

        # Code should remain unchanged
        assert modified_module.code == source_code

    def test_handles_multiple_autouse_fixtures(self):
        """Test that multiple autouse fixtures in the same file are all modified."""
        source_code = '''
import pytest

@pytest.fixture(autouse=True)
def fixture_one(request):
    yield "one"

@pytest.fixture(autouse=True)  
def fixture_two(request):
    yield "two"
'''
        expected_code = '''
import pytest

@pytest.fixture(autouse=True)
def fixture_one(request):
    if request.node.get_closest_marker("codeflash_no_autouse"):
        yield
    else:
        yield "one"

@pytest.fixture(autouse=True)  
def fixture_two(request):
    if request.node.get_closest_marker("codeflash_no_autouse"):
        yield
    else:
        yield "two"
'''
        module = cst.parse_module(source_code)
        modifier = AutouseFixtureModifier()
        modified_module = module.visit(modifier)

        # Both fixtures should be modified
        code = modified_module.code
        assert code==expected_code

    def test_preserves_fixture_with_complex_body(self):
        """Test that fixtures with complex bodies are handled correctly."""
        source_code = '''
import pytest

@pytest.fixture(autouse=True)
def complex_fixture(request):
    try:
        setup_database()
        configure_logging()
        yield get_test_client()
    finally:
        cleanup_database()
        reset_logging()
'''
        expected_code = '''
import pytest

@pytest.fixture(autouse=True)
def complex_fixture(request):
    if request.node.get_closest_marker("codeflash_no_autouse"):
        yield
    else:
        try:
            setup_database()
            configure_logging()
            yield get_test_client()
        finally:
            cleanup_database()
            reset_logging()
'''
        module = cst.parse_module(source_code)
        modifier = AutouseFixtureModifier()
        modified_module = module.visit(modifier)

        code = modified_module.code
        assert code.rstrip()==expected_code.rstrip()


class TestPytestMarkAdder:
    """Test cases for PytestMarkAdder class."""

    def test_adds_pytest_import_when_missing(self):
        """Test that pytest import is added when not present."""
        source_code = '''
def test_something():
    assert True
'''
        expected_code = '''
import pytest
@pytest.mark.codeflash_no_autouse
def test_something():
    assert True
'''
        module = cst.parse_module(source_code)
        mark_adder = PytestMarkAdder("codeflash_no_autouse")
        modified_module = module.visit(mark_adder)

        code = modified_module.code
        assert code==expected_code

    def test_skips_pytest_import_when_present(self):
        """Test that pytest import is not duplicated when already present."""
        source_code = '''
import pytest

def test_something():
    assert True
'''
        expected_code = '''
import pytest

@pytest.mark.codeflash_no_autouse
def test_something():
    assert True
'''
        module = cst.parse_module(source_code)
        mark_adder = PytestMarkAdder("codeflash_no_autouse")
        modified_module = module.visit(mark_adder)

        code = modified_module.code
        # Should only have one import pytest line
        assert code==expected_code

    def test_handles_from_pytest_import(self):
        """Test that existing 'from pytest import ...' is recognized."""
        source_code = '''
from pytest import fixture

def test_something():
    assert True
'''
        expected_code = '''
import pytest
from pytest import fixture

@pytest.mark.codeflash_no_autouse
def test_something():
    assert True
        '''
        module = cst.parse_module(source_code)
        mark_adder = PytestMarkAdder("codeflash_no_autouse")
        modified_module = module.visit(mark_adder)

        code = modified_module.code
        # Should not add import pytest since pytest is already imported
        assert code.strip()==expected_code.strip()

    def test_adds_mark_to_all_functions(self):
        """Test that marks are added to all functions in the module."""
        source_code = '''
import pytest

def test_first():
    assert True

def test_second():
    assert False

def helper_function():
    return "not a test"
'''
        expected_code = '''
import pytest

@pytest.mark.codeflash_no_autouse
def test_first():
    assert True

@pytest.mark.codeflash_no_autouse
def test_second():
    assert False

@pytest.mark.codeflash_no_autouse
def helper_function():
    return "not a test"
'''
        module = cst.parse_module(source_code)
        mark_adder = PytestMarkAdder("codeflash_no_autouse")
        modified_module = module.visit(mark_adder)

        code = modified_module.code
        # All functions should get the mark
        assert code==expected_code

    def test_skips_existing_mark(self):
        """Test that existing marks are not duplicated."""
        source_code = '''
import pytest

@pytest.mark.codeflash_no_autouse
def test_already_marked():
    assert True

def test_needs_mark():
    assert True
'''
        expected_code = '''
import pytest

@pytest.mark.codeflash_no_autouse
def test_already_marked():
    assert True

@pytest.mark.codeflash_no_autouse
def test_needs_mark():
    assert True
'''
        module = cst.parse_module(source_code)
        mark_adder = PytestMarkAdder("codeflash_no_autouse")
        modified_module = module.visit(mark_adder)

        code = modified_module.code
        # Should have exactly 2 marks total (one existing, one added)
        assert code==expected_code

    def test_handles_different_mark_names(self):
        """Test that different mark names work correctly."""
        source_code = '''
import pytest

def test_something():
    assert True
'''
        expected_code = '''
import pytest

@pytest.mark.slow
def test_something():
    assert True
'''
        module = cst.parse_module(source_code)
        mark_adder = PytestMarkAdder("slow")
        modified_module = module.visit(mark_adder)

        code = modified_module.code
        assert code==expected_code

    def test_preserves_existing_decorators(self):
        """Test that existing decorators are preserved."""
        source_code = '''
import pytest

@pytest.mark.parametrize("value", [1, 2, 3])
@pytest.fixture
def test_with_decorators():
    assert True
'''
        expected_code = '''
import pytest

@pytest.mark.parametrize("value", [1, 2, 3])
@pytest.fixture
@pytest.mark.codeflash_no_autouse
def test_with_decorators():
    assert True
'''
        module = cst.parse_module(source_code)
        mark_adder = PytestMarkAdder("codeflash_no_autouse")
        modified_module = module.visit(mark_adder)

        code = modified_module.code
        assert code==expected_code

    def test_handles_call_style_existing_marks(self):
        """Test recognition of existing marks in call style (with parentheses)."""
        source_code = '''
import pytest

@pytest.mark.codeflash_no_autouse()
def test_with_call_mark():
    assert True

def test_needs_mark():
    assert True
'''
        expected_code = '''
import pytest

@pytest.mark.codeflash_no_autouse()
def test_with_call_mark():
    assert True

@pytest.mark.codeflash_no_autouse
def test_needs_mark():
    assert True
'''
        module = cst.parse_module(source_code)
        mark_adder = PytestMarkAdder("codeflash_no_autouse")
        modified_module = module.visit(mark_adder)

        code = modified_module.code
        # Should recognize the existing call-style mark and not duplicate
        assert code==expected_code

    def test_empty_module(self):
        """Test handling of empty module."""
        source_code = ''
        module = cst.parse_module(source_code)
        mark_adder = PytestMarkAdder("codeflash_no_autouse")
        modified_module = module.visit(mark_adder)

        # Should just add the import
        code = modified_module.code
        assert code =='import pytest'

    def test_module_with_only_imports(self):
        """Test handling of module with only imports."""
        source_code = '''
import os
import sys
from pathlib import Path
'''
        expected_code = '''
import pytest
import os
import sys
from pathlib import Path
'''
        module = cst.parse_module(source_code)
        mark_adder = PytestMarkAdder("codeflash_no_autouse")
        modified_module = module.visit(mark_adder)

        code = modified_module.code
        assert code==expected_code


class TestIntegration:
    """Integration tests for all transformers working together."""

    def test_all_transformers_together(self):
        """Test that all three transformers can work on the same code."""
        source_code = '''
import pytest

@pytest.fixture(autouse=True)
def my_fixture():
    yield "value"

def test_something():
    assert True
'''
        expected_code = '''
import pytest

@pytest.fixture(autouse=True)
@pytest.mark.codeflash_no_autouse
def my_fixture(request):
    if request.node.get_closest_marker("codeflash_no_autouse"):
        yield
    else:
        yield "value"

@pytest.mark.codeflash_no_autouse
def test_something():
    assert True
'''
        # First apply AddRequestArgument
        module = cst.parse_module(source_code)
        request_adder = AddRequestArgument()
        modified_module = module.visit(request_adder)

        # Then apply AutouseFixtureModifier
        autouse_modifier = AutouseFixtureModifier()
        modified_module = modified_module.visit(autouse_modifier)

        # Finally apply PytestMarkAdder
        mark_adder = PytestMarkAdder("codeflash_no_autouse")
        final_module = modified_module.visit(mark_adder)

        # Compare complete strings
        assert final_module.code == expected_code

    def test_transformers_with_existing_request_parameter(self):
        """Test transformers when request parameter already exists."""
        source_code = '''
import pytest

@pytest.fixture(autouse=True)
def my_fixture(request):
    setup_code()
    yield "value"
    cleanup_code()

def test_something():
    assert True
'''
        expected_code = '''
import pytest

@pytest.fixture(autouse=True)
@pytest.mark.codeflash_no_autouse
def my_fixture(request):
    if request.node.get_closest_marker("codeflash_no_autouse"):
        yield
    else:
        setup_code()
        yield "value"
        cleanup_code()

@pytest.mark.codeflash_no_autouse
def test_something():
    assert True
'''
        # Apply all transformers in sequence
        module = cst.parse_module(source_code)
        request_adder = AddRequestArgument()
        modified_module = module.visit(request_adder)

        autouse_modifier = AutouseFixtureModifier()
        modified_module = modified_module.visit(autouse_modifier)

        mark_adder = PytestMarkAdder("codeflash_no_autouse")
        final_module = modified_module.visit(mark_adder)

        # Compare complete strings
        assert final_module.code == expected_code

    def test_transformers_with_self_parameter(self):
        """Test transformers when fixture has self parameter."""
        source_code = '''
import pytest

@pytest.fixture(autouse=True)
def my_fixture(self):
    yield "value"

def test_something():
    assert True
'''
        expected_code = '''
import pytest

@pytest.fixture(autouse=True)
@pytest.mark.codeflash_no_autouse
def my_fixture(self, request):
    if request.node.get_closest_marker("codeflash_no_autouse"):
        yield
    else:
        yield "value"

@pytest.mark.codeflash_no_autouse
def test_something():
    assert True
'''
        # Apply all transformers in sequence
        module = cst.parse_module(source_code)
        request_adder = AddRequestArgument()
        modified_module = module.visit(request_adder)

        autouse_modifier = AutouseFixtureModifier()
        modified_module = modified_module.visit(autouse_modifier)

        mark_adder = PytestMarkAdder("codeflash_no_autouse")
        final_module = modified_module.visit(mark_adder)

        # Compare complete strings
        assert final_module.code == expected_code

    def test_transformers_with_multiple_fixtures(self):
        """Test transformers with multiple autouse fixtures."""
        source_code = '''
import pytest

@pytest.fixture(autouse=True)
def fixture_one():
    yield "one"

@pytest.fixture(autouse=True)
def fixture_two(self, param):
    yield "two"

@pytest.fixture
def regular_fixture():
    return "regular"

def test_something():
    assert True
'''
        expected_code = '''
import pytest

@pytest.fixture(autouse=True)
@pytest.mark.codeflash_no_autouse
def fixture_one(request):
    if request.node.get_closest_marker("codeflash_no_autouse"):
        yield
    else:
        yield "one"

@pytest.fixture(autouse=True)
@pytest.mark.codeflash_no_autouse
def fixture_two(self, request, param):
    if request.node.get_closest_marker("codeflash_no_autouse"):
        yield
    else:
        yield "two"

@pytest.fixture
@pytest.mark.codeflash_no_autouse
def regular_fixture():
    return "regular"

@pytest.mark.codeflash_no_autouse
def test_something():
    assert True
'''
        # Apply all transformers in sequence
        module = cst.parse_module(source_code)
        request_adder = AddRequestArgument()
        modified_module = module.visit(request_adder)

        autouse_modifier = AutouseFixtureModifier()
        modified_module = modified_module.visit(autouse_modifier)

        mark_adder = PytestMarkAdder("codeflash_no_autouse")
        final_module = modified_module.visit(mark_adder)

        # Compare complete strings
        assert final_module.code == expected_code




class TestAddRequestArgument:
    """Test cases for AddRequestArgument transformer."""

    def test_adds_request_to_autouse_fixture_no_existing_args(self):
        """Test adding request argument to autouse fixture with no existing arguments."""
        source_code = '''
@fixture(autouse=True)
def my_fixture():
    pass
'''
        expected = '''
@fixture(autouse=True)
def my_fixture(request):
    pass
'''

        module = cst.parse_module(source_code)
        transformer = AddRequestArgument()
        modified_module = module.visit(transformer)

        assert modified_module.code.strip() == expected.strip()

    def test_adds_request_to_pytest_fixture_autouse(self):
        """Test adding request argument to pytest.fixture with autouse=True."""
        source_code = '''
@pytest.fixture(autouse=True)
def my_fixture():
    pass
'''
        expected = '''
@pytest.fixture(autouse=True)
def my_fixture(request):
    pass
'''

        module = cst.parse_module(source_code)
        transformer = AddRequestArgument()
        modified_module = module.visit(transformer)

        assert modified_module.code.strip() == expected.strip()

    def test_adds_request_after_self_parameter(self):
        """Test adding request argument after self parameter."""
        source_code = '''
@fixture(autouse=True)
def my_fixture(self):
    pass
'''
        expected = '''
@fixture(autouse=True)
def my_fixture(self, request):
    pass
'''

        module = cst.parse_module(source_code)
        transformer = AddRequestArgument()
        modified_module = module.visit(transformer)

        assert modified_module.code.strip() == expected.strip()

    def test_adds_request_after_cls_parameter(self):
        """Test adding request argument after cls parameter."""
        source_code = '''
@fixture(autouse=True)
def my_fixture(cls):
    pass
'''
        expected = '''
@fixture(autouse=True)
def my_fixture(cls, request):
    pass
'''

        module = cst.parse_module(source_code)
        transformer = AddRequestArgument()
        modified_module = module.visit(transformer)

        assert modified_module.code.strip() == expected.strip()

    def test_adds_request_before_other_parameters(self):
        """Test adding request argument before other parameters (not self/cls)."""
        source_code = '''
@fixture(autouse=True)
def my_fixture(param1, param2):
    pass
'''
        expected = '''
@fixture(autouse=True)
def my_fixture(request, param1, param2):
    pass
'''

        module = cst.parse_module(source_code)
        transformer = AddRequestArgument()
        modified_module = module.visit(transformer)

        assert modified_module.code.strip() == expected.strip()

    def test_adds_request_after_self_with_other_parameters(self):
        """Test adding request argument after self with other parameters."""
        source_code = '''
@fixture(autouse=True)
def my_fixture(self, param1, param2):
    pass
'''
        expected = '''
@fixture(autouse=True)
def my_fixture(self, request, param1, param2):
    pass
'''

        module = cst.parse_module(source_code)
        transformer = AddRequestArgument()
        modified_module = module.visit(transformer)

        assert modified_module.code.strip() == expected.strip()

    def test_skips_when_request_already_present(self):
        """Test that request argument is not added when already present."""
        source_code = '''
@fixture(autouse=True)
def my_fixture(request):
    pass
'''
        expected = '''
@fixture(autouse=True)
def my_fixture(request):
    pass
'''

        module = cst.parse_module(source_code)
        transformer = AddRequestArgument()
        modified_module = module.visit(transformer)

        assert modified_module.code.strip() == expected.strip()

    def test_skips_when_request_present_with_other_args(self):
        """Test that request argument is not added when already present with other args."""
        source_code = '''
@fixture(autouse=True)
def my_fixture(self, request, param1):
    pass
'''
        expected = '''
@fixture(autouse=True)
def my_fixture(self, request, param1):
    pass
'''

        module = cst.parse_module(source_code)
        transformer = AddRequestArgument()
        modified_module = module.visit(transformer)

        assert modified_module.code.strip() == expected.strip()

    def test_ignores_non_autouse_fixture(self):
        """Test that non-autouse fixtures are not modified."""
        source_code = '''
@fixture
def my_fixture():
    pass
'''
        expected = '''
@fixture
def my_fixture():
    pass
'''

        module = cst.parse_module(source_code)
        transformer = AddRequestArgument()
        modified_module = module.visit(transformer)

        assert modified_module.code.strip() == expected.strip()

    def test_ignores_fixture_with_autouse_false(self):
        """Test that fixtures with autouse=False are not modified."""
        source_code = '''
@fixture(autouse=False)
def my_fixture():
    pass
'''
        expected = '''
@fixture(autouse=False)
def my_fixture():
    pass
'''

        module = cst.parse_module(source_code)
        transformer = AddRequestArgument()
        modified_module = module.visit(transformer)

        assert modified_module.code.strip() == expected.strip()

    def test_ignores_regular_function(self):
        """Test that regular functions are not modified."""
        source_code = '''
def my_function():
    pass
'''
        expected = '''
def my_function():
    pass
'''

        module = cst.parse_module(source_code)
        transformer = AddRequestArgument()
        modified_module = module.visit(transformer)

        assert modified_module.code.strip() == expected.strip()

    def test_handles_multiple_autouse_fixtures(self):
        """Test handling multiple autouse fixtures in the same module."""
        source_code = '''
@fixture(autouse=True)
def fixture1():
    pass

@pytest.fixture(autouse=True)
def fixture2(self):
    pass

@fixture(autouse=True)
def fixture3(request):
    pass
'''
        expected = '''
@fixture(autouse=True)
def fixture1(request):
    pass

@pytest.fixture(autouse=True)
def fixture2(self, request):
    pass

@fixture(autouse=True)
def fixture3(request):
    pass
'''

        module = cst.parse_module(source_code)
        transformer = AddRequestArgument()
        modified_module = module.visit(transformer)

        assert modified_module.code.strip() == expected.strip()

    def test_handles_fixture_with_other_decorators(self):
        """Test handling fixture with other decorators."""
        source_code = '''
@some_decorator
@fixture(autouse=True)
@another_decorator
def my_fixture():
    pass
'''
        expected = '''
@some_decorator
@fixture(autouse=True)
@another_decorator
def my_fixture(request):
    pass
'''

        module = cst.parse_module(source_code)
        transformer = AddRequestArgument()
        modified_module = module.visit(transformer)

        assert modified_module.code.strip() == expected.strip()

    def test_preserves_function_body_and_docstring(self):
        """Test that function body and docstring are preserved."""
        source_code = '''
@fixture(autouse=True)
def my_fixture():
    """This is a docstring."""
    x = 1
    y = 2
    return x + y
'''
        expected = '''
@fixture(autouse=True)
def my_fixture(request):
    """This is a docstring."""
    x = 1
    y = 2
    return x + y
'''

        module = cst.parse_module(source_code)
        transformer = AddRequestArgument()
        modified_module = module.visit(transformer)

        assert modified_module.code.strip() == expected.strip()

    def test_handles_fixture_with_additional_arguments(self):
        """Test handling fixture with additional keyword arguments."""
        source_code = '''
@fixture(autouse=True, scope="session")
def my_fixture():
    pass
'''
        expected = '''
@fixture(autouse=True, scope="session")
def my_fixture(request):
    pass
'''

        module = cst.parse_module(source_code)
        transformer = AddRequestArgument()
        modified_module = module.visit(transformer)

        assert modified_module.code.strip() == expected.strip()


def test_type_checking_imports():
    optim_code = """from dataclasses import dataclass
from pydantic_ai.providers import Provider, infer_provider
from pydantic_ai_slim.pydantic_ai.models import Model
from pydantic_ai_slim.pydantic_ai.tools import ToolDefinition
from typing import Literal

#### problamatic imports ####
from huggingface_hub import AsyncInferenceClient, ChatCompletionInputTool
import requests
import aiohttp as aiohttp_
from math import pi as PI, sin as sine

@dataclass(init=False)
class HuggingFaceModel(Model):
    def __init__(
        self,
        model_name: str,
        *,
        provider: Literal['huggingface'] | Provider[AsyncInferenceClient] = 'huggingface',
    ):
        print(requests.__name__)
        print(aiohttp_.__name__)
        print(PI)
        print(sine)
        # Fast branch: avoid repeating provider assignment
        if isinstance(provider, str):
            provider_obj = infer_provider(provider)
        else:
            provider_obj = provider
        self._provider = provider
        self._model_name = model_name
        self.client = provider_obj.client

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> ChatCompletionInputTool:
        # Inline dict creation and single pass for possible strict attribute
        tool_dict = {
            'type': 'function',
            'function': {
                'name': f.name,
                'description': f.description,
                'parameters': f.parameters_json_schema,
            },
        }
        if f.strict is not None:
            tool_dict['function']['strict'] = f.strict
        return ChatCompletionInputTool.parse_obj_as_instance(tool_dict)  # type: ignore
"""

    original_code = """from dataclasses import dataclass
from pydantic_ai.providers import Provider, infer_provider
from pydantic_ai_slim.pydantic_ai.models import Model
from pydantic_ai_slim.pydantic_ai.tools import ToolDefinition
from typing import Literal

try:
    import aiohttp as aiohttp_
    from math import pi as PI, sin as sine
    from huggingface_hub import (
        AsyncInferenceClient,
        ChatCompletionInputMessage,
        ChatCompletionInputMessageChunk,
        ChatCompletionInputTool,
        ChatCompletionInputToolCall,
        ChatCompletionInputURL,
        ChatCompletionOutput,
        ChatCompletionOutputMessage,
        ChatCompletionStreamOutput,
    )
    from huggingface_hub.errors import HfHubHTTPError

except ImportError as _import_error:
    raise ImportError(
        'Please install `huggingface_hub` to use Hugging Face Inference Providers, '
        'you can use the `huggingface` optional group — `pip install "pydantic-ai-slim[huggingface]"`'
    ) from _import_error

if True:
    import requests

__all__ = (
    'HuggingFaceModel',
    'HuggingFaceModelSettings',
)

@dataclass(init=False)
class HuggingFaceModel(Model):

    def __init__(
        self,
        model_name: str,
        *,
        provider: Literal['huggingface'] | Provider[AsyncInferenceClient] = 'huggingface',
    ):
        self._model_name = model_name
        self._provider = provider
        if isinstance(provider, str):
            provider = infer_provider(provider)
        self.client = provider.client

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> ChatCompletionInputTool:
        tool_param: ChatCompletionInputTool = ChatCompletionInputTool.parse_obj_as_instance(  # type: ignore
            {
                'type': 'function',
                'function': {
                    'name': f.name,
                    'description': f.description,
                    'parameters': f.parameters_json_schema,
                },
            }
        )
        if f.strict is not None:
            tool_param['function']['strict'] = f.strict
        return tool_param
"""


    function_name: str = "HuggingFaceModel._map_tool_definition"
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]] = find_preexisting_objects(original_code)
    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=[function_name],
        optimized_code=optim_code,
        module_abspath=Path(__file__).resolve(),
        preexisting_objects=preexisting_objects,
        project_root_path=Path(__file__).resolve().parent.resolve(),
    )

    assert not re.search(r"^import requests\b", new_code, re.MULTILINE)  # conditional simple import: import <name>
    assert not re.search(r"^import aiohttp as aiohttp_\b", new_code, re.MULTILINE)  # conditional alias import: import <name> as <alias>
    assert not re.search(r"^from math import pi as PI, sin as sine\b", new_code, re.MULTILINE)  # conditional multiple aliases imports
    assert "from huggingface_hub import AsyncInferenceClient, ChatCompletionInputTool" not in new_code # conditional from import

def test_top_level_global_assignments() -> None:
    root_dir = Path(__file__).parent.parent.resolve()
    main_file = Path(root_dir / "code_to_optimize/temp_main.py").resolve()

    original_code = '''"""
Module for generating GeneratedWorkflowParameters schema from workflow run input_text actions.
"""

from typing import Any, Dict, List, Tuple

import structlog
from pydantic import BaseModel

from skyvern.forge import app
from skyvern.forge.sdk.prompting import PromptEngine
from skyvern.webeye.actions.actions import ActionType

LOG = structlog.get_logger(__name__)

# Initialize prompt engine
prompt_engine = PromptEngine("skyvern")


def hydrate_input_text_actions_with_field_names(
    actions_by_task: Dict[str, List[Dict[str, Any]]], field_mappings: Dict[str, str]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Add field_name to input_text actions based on generated mappings.

    Args:
        actions_by_task: Dictionary mapping task IDs to lists of action dictionaries
        field_mappings: Dictionary mapping "task_id:action_id" to field names

    Returns:
        Updated actions_by_task with field_name added to input_text actions
    """
    updated_actions_by_task = {}

    for task_id, actions in actions_by_task.items():
        updated_actions = []

        for action in actions:
            action_copy = action.copy()

            if action.get("action_type") == ActionType.INPUT_TEXT:
                action_id = action.get("action_id", "")
                mapping_key = f"{task_id}:{action_id}"

                if mapping_key in field_mappings:
                    action_copy["field_name"] = field_mappings[mapping_key]
                else:
                    # Fallback field name if mapping not found
                    intention = action.get("intention", "")
                    if intention:
                        # Simple field name generation from intention
                        field_name = intention.lower().replace(" ", "_").replace("?", "").replace("'", "")
                        field_name = "".join(c for c in field_name if c.isalnum() or c == "_")
                        action_copy["field_name"] = field_name or "unknown_field"
                    else:
                        action_copy["field_name"] = "unknown_field"

            updated_actions.append(action_copy)

        updated_actions_by_task[task_id] = updated_actions

    return updated_actions_by_task
'''
    main_file.write_text(original_code, encoding="utf-8")
    optim_code = f'''```python:{main_file.relative_to(root_dir)}
from skyvern.webeye.actions.actions import ActionType
from typing import Any, Dict, List
import re

# Precompiled regex for efficiently generating simple field_name from intention
_INTENTION_CLEANUP_RE = re.compile(r"[^a-zA-Z0-9_]+")

def hydrate_input_text_actions_with_field_names(
    actions_by_task: Dict[str, List[Dict[str, Any]]], field_mappings: Dict[str, str]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Add field_name to input_text actions based on generated mappings.

    Args:
        actions_by_task: Dictionary mapping task IDs to lists of action dictionaries
        field_mappings: Dictionary mapping "task_id:action_id" to field names

    Returns:
        Updated actions_by_task with field_name added to input_text actions
    """
    updated_actions_by_task = {{}}

    input_text_type = ActionType.INPUT_TEXT  # local variable for faster access
    intention_cleanup = _INTENTION_CLEANUP_RE

    for task_id, actions in actions_by_task.items():
        updated_actions = []

        for action in actions:
            action_copy = action.copy()

            if action.get("action_type") == input_text_type:
                action_id = action.get("action_id", "")
                mapping_key = f"{{task_id}}:{{action_id}}"

                if mapping_key in field_mappings:
                    action_copy["field_name"] = field_mappings[mapping_key]
                else:
                    # Fallback field name if mapping not found
                    intention = action.get("intention", "")
                    if intention:
                        # Simple field name generation from intention
                        field_name = intention.lower().replace(" ", "_").replace("?", "").replace("'", "")
                        # Use compiled regex instead of "".join(c for ...)
                        field_name = intention_cleanup.sub("", field_name)
                        action_copy["field_name"] = field_name or "unknown_field"
                    else:
                        action_copy["field_name"] = "unknown_field"

            updated_actions.append(action_copy)

        updated_actions_by_task[task_id] = updated_actions

    return updated_actions_by_task
```
'''
    expected = '''"""
Module for generating GeneratedWorkflowParameters schema from workflow run input_text actions.
"""

from typing import Any, Dict, List, Tuple

import structlog
from pydantic import BaseModel

from skyvern.forge import app
from skyvern.forge.sdk.prompting import PromptEngine
from skyvern.webeye.actions.actions import ActionType
import re

_INTENTION_CLEANUP_RE = re.compile(r"[^a-zA-Z0-9_]+")

LOG = structlog.get_logger(__name__)

# Initialize prompt engine
prompt_engine = PromptEngine("skyvern")


def hydrate_input_text_actions_with_field_names(
    actions_by_task: Dict[str, List[Dict[str, Any]]], field_mappings: Dict[str, str]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Add field_name to input_text actions based on generated mappings.

    Args:
        actions_by_task: Dictionary mapping task IDs to lists of action dictionaries
        field_mappings: Dictionary mapping "task_id:action_id" to field names

    Returns:
        Updated actions_by_task with field_name added to input_text actions
    """
    updated_actions_by_task = {}

    input_text_type = ActionType.INPUT_TEXT  # local variable for faster access
    intention_cleanup = _INTENTION_CLEANUP_RE

    for task_id, actions in actions_by_task.items():
        updated_actions = []

        for action in actions:
            action_copy = action.copy()

            if action.get("action_type") == input_text_type:
                action_id = action.get("action_id", "")
                mapping_key = f"{task_id}:{action_id}"

                if mapping_key in field_mappings:
                    action_copy["field_name"] = field_mappings[mapping_key]
                else:
                    # Fallback field name if mapping not found
                    intention = action.get("intention", "")
                    if intention:
                        # Simple field name generation from intention
                        field_name = intention.lower().replace(" ", "_").replace("?", "").replace("'", "")
                        # Use compiled regex instead of "".join(c for ...)
                        field_name = intention_cleanup.sub("", field_name)
                        action_copy["field_name"] = field_name or "unknown_field"
                    else:
                        action_copy["field_name"] = "unknown_field"

            updated_actions.append(action_copy)

        updated_actions_by_task[task_id] = updated_actions

    return updated_actions_by_task
'''

    func = FunctionToOptimize(function_name="hydrate_input_text_actions_with_field_names", parents=[], file_path=main_file)
    test_config = TestConfig(
        tests_root=root_dir / "tests/pytest",
        tests_project_rootdir=root_dir,
        project_root_path=root_dir,
        test_framework="pytest",
        pytest_cmd="pytest",
    )
    func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
    code_context: CodeOptimizationContext = func_optimizer.get_code_optimization_context().unwrap()
    
    original_helper_code: dict[Path, str] = {}
    helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
    for helper_function_path in helper_function_paths:
        with helper_function_path.open(encoding="utf8") as f:
            helper_code = f.read()
            original_helper_code[helper_function_path] = helper_code

    func_optimizer.args = Args()
    func_optimizer.replace_function_and_helpers_with_optimized_code(
        code_context=code_context, optimized_code=CodeStringsMarkdown.parse_markdown_code(optim_code), original_helper_code=original_helper_code
    )

  
    new_code = main_file.read_text(encoding="utf-8")
    main_file.unlink(missing_ok=True)

    assert new_code == expected


# OptimFunctionCollector async function tests
def test_optim_function_collector_with_async_functions():
    """Test OptimFunctionCollector correctly collects async functions."""
    import libcst as cst
    
    source_code = """
def sync_function():
    return "sync"

async def async_function():
    return "async"

class TestClass:
    def sync_method(self):
        return "sync_method"
    
    async def async_method(self):
        return "async_method"
"""
    
    tree = cst.parse_module(source_code)
    collector = OptimFunctionCollector(
        function_names={(None, "sync_function"), (None, "async_function"), ("TestClass", "sync_method"), ("TestClass", "async_method")},
        preexisting_objects=None
    )
    tree.visit(collector)
    
    # Should collect both sync and async functions
    assert len(collector.modified_functions) == 4
    assert (None, "sync_function") in collector.modified_functions
    assert (None, "async_function") in collector.modified_functions
    assert ("TestClass", "sync_method") in collector.modified_functions
    assert ("TestClass", "async_method") in collector.modified_functions


def test_optim_function_collector_new_async_functions():
    """Test OptimFunctionCollector identifies new async functions not in preexisting objects."""
    import libcst as cst
    
    source_code = """
def existing_function():
    return "existing"

async def new_async_function():
    return "new_async"

def new_sync_function():
    return "new_sync"

class ExistingClass:
    async def new_class_async_method(self):
        return "new_class_async"
"""
    
    # Only existing_function is in preexisting objects
    preexisting_objects = {("existing_function", ())}
    
    tree = cst.parse_module(source_code)
    collector = OptimFunctionCollector(
        function_names=set(),  # Not looking for specific functions
        preexisting_objects=preexisting_objects
    )
    tree.visit(collector)
    
    # Should identify new functions (both sync and async)
    assert len(collector.new_functions) == 2
    function_names = [func.name.value for func in collector.new_functions]
    assert "new_async_function" in function_names
    assert "new_sync_function" in function_names
    
    # Should identify new class methods
    assert "ExistingClass" in collector.new_class_functions
    assert len(collector.new_class_functions["ExistingClass"]) == 1
    assert collector.new_class_functions["ExistingClass"][0].name.value == "new_class_async_method"


def test_optim_function_collector_mixed_scenarios():
    """Test OptimFunctionCollector with complex mix of sync/async functions and classes."""
    import libcst as cst
    
    source_code = """
# Global functions
def global_sync():
    pass

async def global_async():
    pass

class ParentClass:
    def __init__(self):
        pass
    
    def sync_method(self):
        pass
    
    async def async_method(self):
        pass

class ChildClass:
    async def child_async_method(self):
        pass
    
    def child_sync_method(self):
        pass
"""
    
    # Looking for specific functions
    function_names = {
        (None, "global_sync"),
        (None, "global_async"), 
        ("ParentClass", "sync_method"),
        ("ParentClass", "async_method"),
        ("ChildClass", "child_async_method")
    }
    
    tree = cst.parse_module(source_code)
    collector = OptimFunctionCollector(
        function_names=function_names,
        preexisting_objects=None
    )
    tree.visit(collector)
    
    # Should collect all specified functions (mix of sync and async)
    assert len(collector.modified_functions) == 5
    assert (None, "global_sync") in collector.modified_functions
    assert (None, "global_async") in collector.modified_functions
    assert ("ParentClass", "sync_method") in collector.modified_functions
    assert ("ParentClass", "async_method") in collector.modified_functions
    assert ("ChildClass", "child_async_method") in collector.modified_functions
    
    # Should collect __init__ method
    assert "ParentClass" in collector.modified_init_functions



def test_is_zero_diff_async_sleep():
    original_code = '''
import time

async def task():
    time.sleep(1)
    return "done"
'''
    optimized_code = '''
import asyncio

async def task():
    await asyncio.sleep(1)
    return "done"
'''
    assert not is_zero_diff(original_code, optimized_code)

def test_is_zero_diff_with_equivalent_code():
    original_code = '''
import asyncio

async def task():
    await asyncio.sleep(1)
    return "done"
'''
    optimized_code = '''
import asyncio

async def task():
    """A task that does something."""
    await asyncio.sleep(1)
    return "done"
'''
    assert is_zero_diff(original_code, optimized_code)



def test_code_replacement_with_new_helper_class() -> None:
    optim_code = """from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

from bokeh.models import HoverTool, Plot, Tool


# Move the Item dataclass to module-level to avoid redefining it on every function call
@dataclass(frozen=True)
class _RepeatedToolItem:
    obj: Tool
    properties: dict[str, Any]

def _collect_repeated_tools(tool_objs: list[Tool]) -> Iterator[Tool]:
    key: Callable[[Tool], str] = lambda obj: obj.__class__.__name__
    # Pre-collect properties for all objects by group to avoid repeated calls
    for _, group in itertools.groupby(sorted(tool_objs, key=key), key=key):
        grouped = list(group)
        n = len(grouped)
        if n > 1:
            # Precompute all properties once for this group
            props = [_RepeatedToolItem(obj, obj.properties_with_values()) for obj in grouped]
            i = 0
            while i < len(props) - 1:
                head = props[i]
                for j in range(i+1, len(props)):
                    item = props[j]
                    if item.properties == head.properties:
                        yield item.obj
                i += 1
"""

    original_code = """from __future__ import annotations
import itertools
import re
from bokeh.models import HoverTool, Plot, Tool
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

def _collect_repeated_tools(tool_objs: list[Tool]) -> Iterator[Tool]:
    @dataclass(frozen=True)
    class Item:
        obj: Tool
        properties: dict[str, Any]

    key: Callable[[Tool], str] = lambda obj: obj.__class__.__name__

    for _, group in itertools.groupby(sorted(tool_objs, key=key), key=key):
        rest = [ Item(obj, obj.properties_with_values()) for obj in group ]
        while len(rest) > 1:
            head, *rest = rest
            for item in rest:
                if item.properties == head.properties:
                    yield item.obj
"""

    expected = """from __future__ import annotations
import itertools
from bokeh.models import Tool
from dataclasses import dataclass
from typing import Any, Callable, Iterator


# Move the Item dataclass to module-level to avoid redefining it on every function call
@dataclass(frozen=True)
class _RepeatedToolItem:
    obj: Tool
    properties: dict[str, Any]

def _collect_repeated_tools(tool_objs: list[Tool]) -> Iterator[Tool]:
    key: Callable[[Tool], str] = lambda obj: obj.__class__.__name__
    # Pre-collect properties for all objects by group to avoid repeated calls
    for _, group in itertools.groupby(sorted(tool_objs, key=key), key=key):
        grouped = list(group)
        n = len(grouped)
        if n > 1:
            # Precompute all properties once for this group
            props = [_RepeatedToolItem(obj, obj.properties_with_values()) for obj in grouped]
            i = 0
            while i < len(props) - 1:
                head = props[i]
                for j in range(i+1, len(props)):
                    item = props[j]
                    if item.properties == head.properties:
                        yield item.obj
                i += 1
"""

    function_names: list[str] = ["_collect_repeated_tools"]
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]] = find_preexisting_objects(original_code)
    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=function_names,
        optimized_code=optim_code,
        module_abspath=Path(__file__).resolve(),
        preexisting_objects=preexisting_objects,
        project_root_path=Path(__file__).resolve().parent.resolve(),
    )
    assert new_code == expected


def test_global_assignments_with_function_dependencies() -> None:
    """Test that global assignments that depend on functions are inserted after those functions.

    This tests the fix for a bug where LLM-generated optimizations that use module-level
    code like `_TABLE = unicode_to_char(...)` would fail because the assignment was inserted
    before the function definition.
    """
    from codeflash.code_utils.code_extractor import add_global_assignments

    # Original file: standardize_quotes first, unicode_to_char second
    original_code = '''def standardize_quotes(text: str) -> str:
    """Standardize quotes in text."""
    return text

def unicode_to_char(unicode_val: str) -> str:
    """Convert unicode value to char."""
    return chr(int(unicode_val.replace("U+", ""), 16))
'''

    # Optimized code: defines unicode_to_char first, then module-level code that uses it
    optimized_code = '''def unicode_to_char(unicode_val: str) -> str:
    """Convert unicode value to char."""
    return chr(int(unicode_val.replace("U+", ""), 16))

_CODES = ("U+0022", "U+201C")
_TRANSLATION_TABLE = {ord(unicode_to_char(c)): ord('"') for c in _CODES}

def standardize_quotes(text: str) -> str:
    """Standardize quotes in text."""
    return text.translate(_TRANSLATION_TABLE)
'''

    result = add_global_assignments(optimized_code, original_code)

    # The assignment that depends on unicode_to_char should be inserted AFTER unicode_to_char
    # not after imports (which would cause a NameError)

    # Parse the result and verify the order
    import libcst as cst

    module = cst.parse_module(result)

    # Find positions of key elements
    unicode_to_char_pos = None
    translation_table_pos = None

    for i, stmt in enumerate(module.body):
        if isinstance(stmt, cst.FunctionDef) and stmt.name.value == "unicode_to_char":
            unicode_to_char_pos = i
        elif isinstance(stmt, cst.SimpleStatementLine):
            for child in stmt.body:
                if isinstance(child, cst.Assign):
                    for target in child.targets:
                        if isinstance(target.target, cst.Name) and target.target.value == "_TRANSLATION_TABLE":
                            translation_table_pos = i

    # Verify that _TRANSLATION_TABLE comes AFTER unicode_to_char
    assert unicode_to_char_pos is not None, "unicode_to_char function not found in result"
    assert translation_table_pos is not None, "_TRANSLATION_TABLE assignment not found in result"
    assert translation_table_pos > unicode_to_char_pos, (
        f"_TRANSLATION_TABLE (pos {translation_table_pos}) should be after "
        f"unicode_to_char (pos {unicode_to_char_pos}) because it depends on it"
    )


def test_global_assignments_inside_for_loops_not_extracted() -> None:
    """Test that assignments inside for-loops are NOT extracted as standalone globals.

    This tests the fix for a bug where LLM-generated optimizations that build
    translation tables using for-loops would have the loop body assignments
    incorrectly extracted, causing NameError for loop variables.
    """
    from codeflash.code_utils.code_extractor import GlobalAssignmentCollector

    import libcst as cst

    # Code with assignments inside a for-loop (common optimization pattern)
    # Note: Using regular assignment (not annotated) since GlobalAssignmentCollector only handles Assign
    code_with_for_loop = '''
double_quotes = {"a": "U+0022", "b": "U+201C"}

_QUOTE_TRANSLATION = {}
for unicode_val in double_quotes.values():
    ch = unicode_to_char(unicode_val)
    _QUOTE_TRANSLATION[ord(ch)] = _double_quote_standard

def standardize_quotes(text: str) -> str:
    return text.translate(_QUOTE_TRANSLATION)
'''

    module = cst.parse_module(code_with_for_loop)
    collector = GlobalAssignmentCollector()
    module.visit(collector)

    # Only the top-level assignments should be collected, NOT the one inside the for-loop
    # _QUOTE_TRANSLATION = {} is at module level (should be collected)
    # _QUOTE_TRANSLATION[ord(ch)] = ... is inside for-loop (should NOT be collected)
    # double_quotes = {...} is at module level (should be collected)
    assert "double_quotes" in collector.assignments, "double_quotes should be collected"
    assert "_QUOTE_TRANSLATION" in collector.assignments, "_QUOTE_TRANSLATION init should be collected"

    # The assignment inside the for-loop uses subscript, not Name, so it wouldn't be
    # collected anyway. But let's verify the collector doesn't crash and works correctly.
    # More importantly, verify that simple name assignments inside loops are NOT collected.

    code_with_simple_assignment_in_loop = '''
result = {}
for item in items:
    key = process(item)
    result[key] = item
'''
    module2 = cst.parse_module(code_with_simple_assignment_in_loop)
    collector2 = GlobalAssignmentCollector()
    module2.visit(collector2)

    assert "result" in collector2.assignments, "result should be collected (top-level)"
    assert "key" not in collector2.assignments, "key should NOT be collected (inside for-loop)"


def test_global_assignments_inside_while_loops_not_extracted() -> None:
    """Test that assignments inside while-loops are NOT extracted."""
    from codeflash.code_utils.code_extractor import GlobalAssignmentCollector

    import libcst as cst

    code_with_while_loop = '''
counter = 0
while counter < 10:
    value = compute(counter)
    counter += 1
'''
    module = cst.parse_module(code_with_while_loop)
    collector = GlobalAssignmentCollector()
    module.visit(collector)

    assert "counter" in collector.assignments, "counter should be collected (top-level)"
    assert "value" not in collector.assignments, "value should NOT be collected (inside while-loop)"


def test_global_assignments_inside_with_blocks_not_extracted() -> None:
    """Test that assignments inside with-blocks are NOT extracted."""
    from codeflash.code_utils.code_extractor import GlobalAssignmentCollector

    import libcst as cst

    code_with_with_block = '''
config = {}
with open("file.txt") as f:
    content = f.read()
    data = parse(content)
'''
    module = cst.parse_module(code_with_with_block)
    collector = GlobalAssignmentCollector()
    module.visit(collector)

    assert "config" in collector.assignments, "config should be collected (top-level)"
    assert "content" not in collector.assignments, "content should NOT be collected (inside with-block)"
    assert "data" not in collector.assignments, "data should NOT be collected (inside with-block)"


def test_global_assignments_inside_try_blocks_not_extracted() -> None:
    """Test that assignments inside try-blocks are NOT extracted."""
    from codeflash.code_utils.code_extractor import GlobalAssignmentCollector

    import libcst as cst

    code_with_try_block = '''
default = "fallback"
try:
    result = risky_operation()
    processed = transform(result)
except Exception:
    pass
'''
    module = cst.parse_module(code_with_try_block)
    collector = GlobalAssignmentCollector()
    module.visit(collector)

    assert "default" in collector.assignments, "default should be collected (top-level)"
    assert "result" not in collector.assignments, "result should NOT be collected (inside try-block)"
    assert "processed" not in collector.assignments, "processed should NOT be collected (inside try-block)"


def test_global_assignment_transformer_ignores_loop_assignments() -> None:
    """Test that GlobalAssignmentTransformer doesn't replace assignments inside loops.

    The transformer should:
    1. NOT replace assignments inside for/while/with/try blocks
    2. Still add new top-level assignments that weren't in the original
    """
    from codeflash.code_utils.code_extractor import GlobalAssignmentTransformer

    import libcst as cst

    # Original code has 'key' inside a for-loop and 'result' at top level
    original_code = '''
result = {}
for item in items:
    key = process(item)
    result[key] = item
'''
    # New assignments - 'result' should replace top-level, 'key' should NOT replace loop var
    new_assignments = {
        "result": cst.parse_statement("result = {'new': 'dict'}").body[0],
        "key": cst.parse_statement("key = 'new_value'").body[0],
    }

    module = cst.parse_module(original_code)
    transformer = GlobalAssignmentTransformer(new_assignments, ["result", "key"])
    result_module = module.visit(transformer)

    result_code = result_module.code

    # The 'key' inside the for-loop should NOT be replaced
    assert "key = process(item)" in result_code, "Assignment inside for-loop should not be transformed"

    # The top-level 'result' SHOULD be replaced
    assert "result = {'new': 'dict'}" in result_code, "Top-level assignment should be replaced"
    assert "result = {}" not in result_code, "Original top-level assignment should be gone"


def test_add_global_assignments_with_loop_variables() -> None:
    """Test that add_global_assignments doesn't extract assignments that reference loop variables.

    This is the specific bug case from optimization runs where code like:
        for uval in double_quotes.values():
            ch = unicode_to_char(uval)
            _translation_map[ord(ch)] = _double_quote_ord

    Would have 'ch = unicode_to_char(uval)' extracted and inserted at module level,
    causing 'NameError: name 'uval' is not defined'.
    """
    from codeflash.code_utils.code_extractor import add_global_assignments

    # Original simple function
    original_code = '''def standardize_quotes(text: str) -> str:
    return text

def unicode_to_char(unicode_val: str) -> str:
    return chr(int(unicode_val.replace("U+", ""), 16))
'''

    # Optimized code with for-loop that builds translation table at module level
    optimized_code = '''def unicode_to_char(unicode_val: str) -> str:
    return chr(int(unicode_val.replace("U+", ""), 16))

double_quotes = {"a": "U+0022", "b": "U+201C"}
_translation_map = {}

for uval in double_quotes.values():
    ch = unicode_to_char(uval)
    _translation_map[ord(ch)] = ord('"')

def standardize_quotes(text: str) -> str:
    return text.translate(_translation_map)
'''

    result = add_global_assignments(optimized_code, original_code)

    # The result should be valid Python - no NameError when compiled
    # If 'ch = unicode_to_char(uval)' was incorrectly extracted, it would cause
    # NameError because 'uval' wouldn't be defined outside the for-loop
    try:
        compile(result, "<test>", "exec")
    except NameError as e:
        raise AssertionError(f"Generated code has NameError (loop var extracted incorrectly): {e}") from e
    except SyntaxError as e:
        raise AssertionError(f"Generated code has SyntaxError: {e}") from e

    # Verify the for-loop structure is preserved (ch assignment inside loop)
    assert "for uval in" in result or "for unicode_val in" in result or "ch" not in result, (
        "If ch is in result, the for-loop should also be present"
    )


def test_add_global_assignments_with_unicode_val_loop_variable() -> None:
    """Test that add_global_assignments correctly transfers for-loops using 'unicode_val' as loop variable.

    This is the specific bug case: NameError: name 'unicode_val' is not defined
    """
    from codeflash.code_utils.code_extractor import add_global_assignments

    original_code = '''def standardize_quotes(text: str) -> str:
    return text
'''

    # Optimized code with for-loop using unicode_val as the loop variable
    optimized_code = '''single_quotes = {"U+0027": "'", "U+2018": "'", "U+2019": "'"}
_translation_map = {}

for unicode_val in single_quotes:
    ch = chr(int(unicode_val.replace("U+", ""), 16))
    _translation_map[ord(ch)] = ord("'")

def standardize_quotes(text: str) -> str:
    return text.translate(_translation_map)
'''

    result = add_global_assignments(optimized_code, original_code)

    # The result should be valid Python - no NameError when compiled
    try:
        compile(result, "<test>", "exec")
    except NameError as e:
        raise AssertionError(f"Generated code has NameError (unicode_val extracted incorrectly): {e}") from e
    except SyntaxError as e:
        raise AssertionError(f"Generated code has SyntaxError: {e}") from e

    # Verify the for-loop is present
    assert "for unicode_val in" in result, "For-loop with unicode_val should be transferred"


def test_add_global_assignments_with_ch_loop_variable() -> None:
    """Test that add_global_assignments correctly transfers for-loops using 'ch' as loop variable.

    This is the specific bug case: NameError: name 'ch' is not defined
    """
    from codeflash.code_utils.code_extractor import add_global_assignments

    original_code = '''def normalize_text(text: str) -> str:
    return text
'''

    # Optimized code with for-loop using ch as the loop variable
    optimized_code = '''replacements = {"a": "A", "b": "B"}
_char_map = {}

for ch in replacements:
    _char_map[ord(ch)] = ord(replacements[ch])

def normalize_text(text: str) -> str:
    return text.translate(_char_map)
'''

    result = add_global_assignments(optimized_code, original_code)

    # The result should be valid Python - no NameError when compiled
    try:
        compile(result, "<test>", "exec")
    except NameError as e:
        raise AssertionError(f"Generated code has NameError (ch extracted incorrectly): {e}") from e
    except SyntaxError as e:
        raise AssertionError(f"Generated code has SyntaxError: {e}") from e

    # Verify the for-loop is present
    assert "for ch in" in result, "For-loop with ch should be transferred"


def test_add_global_assignments_with_helper_function_call() -> None:
    """Test that add_global_assignments transfers assignments that call helper functions.

    Note: add_global_assignments only handles assignments, not function definitions.
    Function definitions are transferred via replace_functions_in_file separately.

    This test verifies that assignments calling helper functions are correctly transferred.
    The actual NameError: name '_build_quote_translation_table' is not defined would occur
    if the function wasn't transferred in the full replacement flow.
    """
    from codeflash.code_utils.code_extractor import add_global_assignments

    # Original code that already has the helper function defined
    original_code = '''def _build_quote_translation_table():
    table = {}
    for i in range(128):
        table[i] = i
    return table

def standardize_quotes(text: str) -> str:
    return text
'''

    # Optimized code with an assignment that calls the helper function
    optimized_code = '''def _build_quote_translation_table():
    table = {}
    for i in range(128):
        table[i] = i
    return table

_QUOTE_TABLE = _build_quote_translation_table()

def standardize_quotes(text: str) -> str:
    return text.translate(_QUOTE_TABLE)
'''

    result = add_global_assignments(optimized_code, original_code)

    # The result should be valid Python - no NameError when compiled
    try:
        compile(result, "<test>", "exec")
    except NameError as e:
        raise AssertionError(f"Generated code has NameError: {e}") from e
    except SyntaxError as e:
        raise AssertionError(f"Generated code has SyntaxError: {e}") from e

    # Verify the assignment is present and placed AFTER the function definition
    assert "_QUOTE_TABLE = _build_quote_translation_table()" in result, "Assignment should be transferred"
    # Verify the function is still present (it was already in original)
    assert "def _build_quote_translation_table" in result, "Helper function should be preserved"

    # Verify correct ordering: function must come before the assignment that uses it
    func_pos = result.index("def _build_quote_translation_table")
    assign_pos = result.index("_QUOTE_TABLE = _build_quote_translation_table()")
    assert func_pos < assign_pos, "Function definition must come before assignment that calls it"


def test_add_global_assignments_forloop_calls_function_defined_later() -> None:
    """Test that for-loops calling functions are placed AFTER those function definitions.

    This is the specific bug case: NameError: name 'unicode_to_char' is not defined

    When the original file has a function defined later (e.g., after standardize_quotes),
    and the optimized code adds a for-loop that calls that function, the for-loop must
    be placed AFTER the function definition, not at the top of the file.
    """
    from codeflash.code_utils.code_extractor import add_global_assignments

    # Original code where unicode_to_char is defined AFTER the main function
    # This mirrors the real-world case where the helper is at the bottom
    original_code = '''def standardize_quotes(text: str) -> str:
    return text


def unicode_to_char(unicode_val: str) -> str:
    return chr(int(unicode_val.replace("U+", ""), 16))
'''

    # Optimized code with a for-loop that calls unicode_to_char at module level
    optimized_code = '''def unicode_to_char(unicode_val: str) -> str:
    return chr(int(unicode_val.replace("U+", ""), 16))

_DOUBLE_QUOTE_UNICODE_VALUES = ["U+0022", "U+201C", "U+201D"]
_TRANSLATION_TABLE = {}

for code in _DOUBLE_QUOTE_UNICODE_VALUES:
    ch = unicode_to_char(code)
    _TRANSLATION_TABLE[ord(ch)] = ord('"')

def standardize_quotes(text: str) -> str:
    return text.translate(_TRANSLATION_TABLE)
'''

    result = add_global_assignments(optimized_code, original_code)

    # The result should be valid Python - no NameError when executed
    try:
        compile(result, "<test>", "exec")
        # Also try to actually execute it to catch runtime NameErrors
        exec(compile(result, "<test>", "exec"), {})
    except NameError as e:
        raise AssertionError(
            f"Generated code has NameError (for-loop placed before function): {e}\n\nGenerated code:\n{result}"
        ) from e
    except SyntaxError as e:
        raise AssertionError(f"Generated code has SyntaxError: {e}") from e

    # Verify the for-loop is present
    assert "for code in _DOUBLE_QUOTE_UNICODE_VALUES" in result, "For-loop should be transferred"

    # Verify correct ordering: function must come before the for-loop that calls it
    func_pos = result.index("def unicode_to_char")
    forloop_pos = result.index("for code in _DOUBLE_QUOTE_UNICODE_VALUES")
    assert func_pos < forloop_pos, (
        f"Function definition must come before for-loop that calls it.\n"
        f"Function at position {func_pos}, for-loop at position {forloop_pos}\n"
        f"Generated code:\n{result}"
    )


def test_add_global_assignments_forloop_uses_computed_variable() -> None:
    """Test that for-loops are placed after variables they depend on.

    This is the specific bug case: NameError: name '_double_chars' is not defined

    When optimized code computes a variable and then uses it in a for-loop,
    the assignment must come before the for-loop.
    """
    from codeflash.code_utils.code_extractor import add_global_assignments

    original_code = '''def process_text(text: str) -> str:
    return text
'''

    # Optimized code where _double_chars is computed and then used in a for-loop
    optimized_code = '''_UNICODE_VALUES = ["U+0022", "U+201C"]
_double_chars = tuple(chr(int(u.replace("U+", ""), 16)) for u in _UNICODE_VALUES)

_TRANSLATION = {}
for ch in _double_chars:
    _TRANSLATION[ord(ch)] = ord('"')

def process_text(text: str) -> str:
    return text.translate(_TRANSLATION)
'''

    result = add_global_assignments(optimized_code, original_code)

    # The result should be valid Python - no NameError when executed
    try:
        compile(result, "<test>", "exec")
        exec(compile(result, "<test>", "exec"), {})
    except NameError as e:
        raise AssertionError(
            f"Generated code has NameError (variable not defined before for-loop): {e}\n\nGenerated code:\n{result}"
        ) from e
    except SyntaxError as e:
        raise AssertionError(f"Generated code has SyntaxError: {e}") from e

    # Verify the for-loop is present
    assert "for ch in _double_chars" in result, "For-loop should be transferred"

    # Verify correct ordering: _double_chars assignment must come before the for-loop
    assign_pos = result.index("_double_chars = ")
    forloop_pos = result.index("for ch in _double_chars")
    assert assign_pos < forloop_pos, (
        f"Variable assignment must come before for-loop that uses it.\n"
        f"Assignment at position {assign_pos}, for-loop at position {forloop_pos}\n"
        f"Generated code:\n{result}"
    )


def test_add_global_assignments_multiple_forloops_with_dependencies() -> None:
    """Test that multiple for-loops with function dependencies are ordered correctly.

    This tests the real-world case from standardize_quotes optimization where:
    1. unicode_to_char function is used
    2. Multiple for-loops build translation tables
    """
    from codeflash.code_utils.code_extractor import add_global_assignments

    # Original code with function defined after main function
    original_code = '''def standardize_quotes(text: str) -> str:
    return text


def unicode_to_char(unicode_val: str) -> str:
    return chr(int(unicode_val.replace("U+", ""), 16))
'''

    # Optimized code with multiple for-loops that depend on unicode_to_char
    optimized_code = '''def unicode_to_char(unicode_val: str) -> str:
    return chr(int(unicode_val.replace("U+", ""), 16))

double_quotes = {"U+0022": '"', "U+201C": '"'}
single_quotes = {"U+0027": "'", "U+2018": "'"}

_translation_table = {}

for unicode_val in double_quotes:
    ch = unicode_to_char(unicode_val)
    _translation_table[ord(ch)] = ord('"')

for unicode_val in single_quotes:
    ch = unicode_to_char(unicode_val)
    _translation_table[ord(ch)] = ord("'")

def standardize_quotes(text: str) -> str:
    return text.translate(_translation_table)
'''

    result = add_global_assignments(optimized_code, original_code)

    # The result should be valid Python - no NameError when executed
    try:
        compile(result, "<test>", "exec")
        exec(compile(result, "<test>", "exec"), {})
    except NameError as e:
        raise AssertionError(
            f"Generated code has NameError: {e}\n\nGenerated code:\n{result}"
        ) from e
    except SyntaxError as e:
        raise AssertionError(f"Generated code has SyntaxError: {e}") from e

    # Verify both for-loops are present
    assert "for unicode_val in double_quotes" in result, "First for-loop should be transferred"
    assert "for unicode_val in single_quotes" in result, "Second for-loop should be transferred"

    # Verify correct ordering: function must come before all for-loops
    func_pos = result.index("def unicode_to_char")
    first_forloop_pos = result.index("for unicode_val in double_quotes")
    second_forloop_pos = result.index("for unicode_val in single_quotes")

    assert func_pos < first_forloop_pos, "Function must come before first for-loop"
    assert func_pos < second_forloop_pos, "Function must come before second for-loop"


def test_add_global_assignments_variable_depends_on_existing_global() -> None:
    """Test that new global assignments depending on existing globals are inserted after them.

    This tests the fix for a bug where LLM-generated optimizations that add module-level
    cache variables like `_TRANSLATION_CACHE: dict = {None: tbl}` would fail because the
    assignment was inserted after imports but BEFORE the `tbl` variable it depends on.

    Real-world example from unstructured/cleaners/core.py optimization:
    - Original has: `tbl = dict.fromkeys(...)`
    - Optimized adds: `_TRANSLATION_CACHE: dict = {None: tbl}` (depends on tbl)
    """
    from codeflash.code_utils.code_extractor import add_global_assignments

    # Original file with imports and module-level variable
    original_code = '''from __future__ import annotations
import sys
import unicodedata

tbl = dict.fromkeys(
    i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
)

def remove_sentence_punctuation(s: str) -> str:
    tbl_new = tbl.copy()
    return s.translate(tbl_new)
'''

    # Optimized code adds a cache that depends on `tbl`
    optimized_code = '''from __future__ import annotations
import sys
import unicodedata

tbl = dict.fromkeys(
    i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
)

# Cache for translation tables
_TRANSLATION_CACHE: dict = {None: tbl}

def remove_sentence_punctuation(s: str) -> str:
    tbl_new = tbl.copy()
    return s.translate(tbl_new)
'''

    result = add_global_assignments(optimized_code, original_code)

    # The result should be valid Python - no NameError when executed
    try:
        compile(result, "<test>", "exec")
        exec(compile(result, "<test>", "exec"), {})
    except NameError as e:
        raise AssertionError(
            f"Generated code has NameError: {e}\n\nGenerated code:\n{result}"
        ) from e
    except SyntaxError as e:
        raise AssertionError(f"Generated code has SyntaxError: {e}") from e

    # Verify `_TRANSLATION_CACHE` is in the result
    assert "_TRANSLATION_CACHE" in result, "_TRANSLATION_CACHE should be in the result"

    # Verify correct ordering: `tbl` must come before `_TRANSLATION_CACHE`
    tbl_pos = result.index("tbl = dict.fromkeys")
    cache_pos = result.index("_TRANSLATION_CACHE")

    assert cache_pos > tbl_pos, (
        f"_TRANSLATION_CACHE (pos {cache_pos}) should be after "
        f"tbl (pos {tbl_pos}) because it depends on it"
    )


def test_add_global_assignments_tuple_unpacking() -> None:
    """Test that tuple unpacking assignments are properly tracked.

    Verifies the fix for: a, b = 1, 2 where the target is a Tuple, not a Name.
    Without the fix, neither 'a' nor 'b' would be tracked as defined.
    """
    from codeflash.code_utils.code_extractor import add_global_assignments

    original_code = '''import sys

def foo():
    pass
'''

    # Optimized code with tuple unpacking that a later variable depends on
    optimized_code = '''import sys

a, b = 1, 2
c = a + b

def foo():
    pass
'''

    result = add_global_assignments(optimized_code, original_code)

    # Verify the code is valid and executes without NameError
    try:
        compiled = compile(result, "<test>", "exec")
        exec(compiled, {})
    except NameError as e:
        msg = f"Tuple unpacking test failed with NameError: {e}\n\nGenerated code:\n{result}"
        raise AssertionError(msg) from e

    # Verify correct ordering: a, b = 1, 2 must come before c = a + b
    unpack_pos = result.index("a, b = 1, 2")
    c_pos = result.index("c = a + b")
    assert unpack_pos < c_pos, "Tuple unpacking must come before dependent assignment"


def test_add_global_assignments_chained_assignment() -> None:
    """Test that chained assignments are properly tracked.

    Verifies the fix for: a = b = c = 5 which has multiple targets.
    """
    from codeflash.code_utils.code_extractor import add_global_assignments

    original_code = '''import sys

def foo():
    pass
'''

    # Optimized code with chained assignment that a later variable depends on
    optimized_code = '''import sys

a = b = c = 5
d = a + b + c

def foo():
    pass
'''

    result = add_global_assignments(optimized_code, original_code)

    # Verify the code is valid and executes without NameError
    try:
        compiled = compile(result, "<test>", "exec")
        exec(compiled, {})
    except NameError as e:
        msg = f"Chained assignment test failed with NameError: {e}\n\nGenerated code:\n{result}"
        raise AssertionError(msg) from e

    # Verify correct ordering: a = b = c = 5 must come before d = a + b + c
    chain_pos = result.index("a = b = c = 5")
    d_pos = result.index("d = a + b + c")
    assert chain_pos < d_pos, "Chained assignment must come before dependent assignment"


def test_add_global_assignments_multiple_new_statements() -> None:
    """Test that multiple new statements maintain correct order.

    When inserting multiple statements with no dependencies, they should
    maintain their relative order from the optimized code.
    """
    from codeflash.code_utils.code_extractor import add_global_assignments

    original_code = '''import sys

def foo():
    pass
'''

    # Optimized code with multiple independent global assignments
    optimized_code = '''import sys

FIRST = 1
SECOND = 2
THIRD = 3

def foo():
    pass
'''

    result = add_global_assignments(optimized_code, original_code)

    # Verify the code is valid
    try:
        compiled = compile(result, "<test>", "exec")
        exec(compiled, {})
    except (NameError, SyntaxError) as e:
        msg = f"Multiple statements test failed: {e}\n\nGenerated code:\n{result}"
        raise AssertionError(msg) from e

    # Verify correct ordering: FIRST, SECOND, THIRD in order
    first_pos = result.index("FIRST = 1")
    second_pos = result.index("SECOND = 2")
    third_pos = result.index("THIRD = 3")
    assert first_pos < second_pos < third_pos, (
        f"Statements should maintain order: FIRST ({first_pos}) < SECOND ({second_pos}) < THIRD ({third_pos})"
    )


def test_add_global_assignments_annotated_no_spurious_deps() -> None:
    """Test that type annotations don't create spurious dependencies.

    Verifies the fix for: x: Tuple[int, int] = value
    Without the fix, Tuple and int would be added as spurious dependencies.
    """
    from codeflash.code_utils.code_extractor import add_global_assignments

    original_code = '''import sys

val = (1, 2)

def foo():
    pass
'''

    # Optimized code with type annotation - the annotation uses Tuple[int, int]
    # but the actual value only depends on 'val'
    optimized_code = '''import sys

val = (1, 2)
x: tuple[int, int] = val
y = x

def foo():
    pass
'''

    result = add_global_assignments(optimized_code, original_code)

    # Verify the code is valid and executes without NameError
    try:
        compiled = compile(result, "<test>", "exec")
        exec(compiled, {})
    except NameError as e:
        msg = f"Annotated assignment test failed with NameError: {e}\n\nGenerated code:\n{result}"
        raise AssertionError(msg) from e

    # Verify x assignment is in the result
    assert "x: tuple[int, int] = val" in result, "Annotated assignment should be present"

    # Verify correct ordering: val must come before x, x must come before y
    val_pos = result.index("val = (1, 2)")
    x_pos = result.index("x: tuple[int, int] = val")
    y_pos = result.index("y = x")
    assert val_pos < x_pos < y_pos, "Variables should be in dependency order"


# =============================================================================
# Real-world standardize_quotes optimization tests
# These tests verify the fixes work for the actual optimization scenarios
# =============================================================================


def test_standardize_quotes_optimization_candidate_6_pattern() -> None:
    """Test optimization pattern from candidate 6 that caused NameError: name '_double_chars' is not defined.

    This pattern uses tuple comprehensions at module level that depend on unicode_to_char,
    then uses those tuples in for-loops to build translation tables.
    """
    from codeflash.code_utils.code_extractor import add_global_assignments

    # Original standardize_quotes code structure
    original_code = '''def standardize_quotes(text: str) -> str:
    """Converts all unicode quotes to standard ASCII quotes."""
    double_quotes = {
        '"': "U+0022",
        '"': "U+201C",
        '"': "U+201D",
    }
    single_quotes = {
        "'": "U+0027",
        "'": "U+2018",
        "'": "U+2019",
    }

    double_quote_standard = '"'
    single_quote_standard = "'"

    for unicode_val in double_quotes.values():
        unicode_char = unicode_to_char(unicode_val)
        if unicode_char in text:
            text = text.replace(unicode_char, double_quote_standard)

    for unicode_val in single_quotes.values():
        unicode_char = unicode_to_char(unicode_val)
        if unicode_char in text:
            text = text.replace(unicode_char, single_quote_standard)

    return text


def unicode_to_char(unicode_val: str) -> str:
    """Converts a Unicode value to a character."""
    return chr(int(unicode_val.replace("U+", ""), 16))
'''

    # Optimization candidate 6 pattern: precompute chars using tuple comprehension
    optimized_code = '''def unicode_to_char(unicode_val: str) -> str:
    """Converts a Unicode value to a character."""
    return chr(int(unicode_val.replace("U+", ""), 16))

double_quotes = {
    '"': "U+0022",
    '"': "U+201C",
    '"': "U+201D",
}
single_quotes = {
    "'": "U+0027",
    "'": "U+2018",
    "'": "U+2019",
}

_double_chars = tuple(unicode_to_char(u) for u in double_quotes.values())
_single_chars = tuple(unicode_to_char(u) for u in single_quotes.values())

_QUOTE_TRANSLATION: dict[int, str] = {}
_double_quote_standard = '"'
_single_quote_standard = "'"

for _ch in _double_chars:
    _QUOTE_TRANSLATION[ord(_ch)] = _double_quote_standard

for _ch in _single_chars:
    _QUOTE_TRANSLATION[ord(_ch)] = _single_quote_standard


def standardize_quotes(text: str) -> str:
    """Converts all unicode quotes to standard ASCII quotes."""
    return text.translate(_QUOTE_TRANSLATION)
'''

    result = add_global_assignments(optimized_code, original_code)

    # Verify the code is valid and executes without NameError
    try:
        compiled = compile(result, "<test>", "exec")
        exec(compiled, {})
    except NameError as e:
        raise AssertionError(
            f"Candidate 6 pattern failed with NameError: {e}\n\nGenerated code:\n{result}"
        ) from e

    # Verify key elements are present
    assert "_double_chars = tuple" in result, "_double_chars assignment should be present"
    assert "for _ch in _double_chars" in result, "First for-loop should be present"
    assert "for _ch in _single_chars" in result, "Second for-loop should be present"

    # Verify ordering: unicode_to_char must come before _double_chars assignment
    func_pos = result.index("def unicode_to_char")
    double_chars_pos = result.index("_double_chars = tuple")
    assert func_pos < double_chars_pos, "unicode_to_char must be defined before _double_chars"


def test_standardize_quotes_optimization_candidate_7_pattern() -> None:
    """Test optimization pattern from candidate 7 that caused NameError: name 'unicode_to_char' is not defined.

    This pattern moves quote dictionaries to module level and builds translation
    table using for-loops that call unicode_to_char.
    """
    from codeflash.code_utils.code_extractor import add_global_assignments

    # Original code
    original_code = '''def standardize_quotes(text: str) -> str:
    """Converts all unicode quotes to standard ASCII quotes."""
    return text


def unicode_to_char(unicode_val: str) -> str:
    """Converts a Unicode value to a character."""
    return chr(int(unicode_val.replace("U+", ""), 16))
'''

    # Optimization candidate 7 pattern: module-level for-loops calling unicode_to_char
    optimized_code = '''def unicode_to_char(unicode_val: str) -> str:
    """Converts a Unicode value to a character."""
    return chr(int(unicode_val.replace("U+", ""), 16))

_DOUBLE_QUOTE_UNICODE_VALUES = [
    "U+0022",
    "U+201C",
    "U+201D",
]

_SINGLE_QUOTE_UNICODE_VALUES = [
    "U+0027",
    "U+2018",
    "U+2019",
]

_QUOTE_TRANSLATION_TABLE = {}
_double_replacement_ord = ord('"')
_single_replacement_ord = ord("'")

for u in _DOUBLE_QUOTE_UNICODE_VALUES:
    ch = unicode_to_char(u)
    _QUOTE_TRANSLATION_TABLE[ord(ch)] = _double_replacement_ord

for u in _SINGLE_QUOTE_UNICODE_VALUES:
    ch = unicode_to_char(u)
    _QUOTE_TRANSLATION_TABLE[ord(ch)] = _single_replacement_ord


def standardize_quotes(text: str) -> str:
    """Converts all unicode quotes to standard ASCII quotes."""
    return text.translate(_QUOTE_TRANSLATION_TABLE)
'''

    result = add_global_assignments(optimized_code, original_code)

    # Verify the code is valid and executes without NameError
    try:
        compiled = compile(result, "<test>", "exec")
        namespace = {}
        exec(compiled, namespace)

        # Verify the translation table is correctly populated by the for-loops
        # This verifies that the for-loops execute correctly and call unicode_to_char
        translation_table = namespace.get("_QUOTE_TRANSLATION_TABLE")
        assert translation_table is not None, "_QUOTE_TRANSLATION_TABLE should be defined"
        # 8220 = U+201C (left double quote), 8221 = U+201D (right double quote)
        # 34 = ord('"') = ASCII double quote
        assert 8220 in translation_table, "Left double quote (U+201C) should be in table"
        assert 8221 in translation_table, "Right double quote (U+201D) should be in table"
        assert translation_table[8220] == 34, "Left double quote should map to ASCII quote"
        assert translation_table[8221] == 34, "Right double quote should map to ASCII quote"

    except NameError as e:
        raise AssertionError(
            f"Candidate 7 pattern failed with NameError: {e}\n\nGenerated code:\n{result}"
        ) from e

    # Verify for-loops are present and ordered correctly
    assert "for u in _DOUBLE_QUOTE_UNICODE_VALUES" in result
    assert "for u in _SINGLE_QUOTE_UNICODE_VALUES" in result

    func_pos = result.index("def unicode_to_char")
    first_loop_pos = result.index("for u in _DOUBLE_QUOTE_UNICODE_VALUES")
    assert func_pos < first_loop_pos, "unicode_to_char must be defined before for-loops"


def test_standardize_quotes_optimization_candidate_9_pattern() -> None:
    """Test optimization pattern from candidate 9 that caused NameError: name 'unicode_to_char' is not defined.

    This pattern is similar to candidate 7 but with slightly different variable names.
    It builds a translation table at module level using for-loops.
    """
    from codeflash.code_utils.code_extractor import add_global_assignments

    # Original code with unicode_to_char defined AFTER standardize_quotes
    original_code = '''def standardize_quotes(text: str) -> str:
    """Converts all unicode quotes to standard ASCII quotes."""
    double_quotes = {'"': "U+0022", '"': "U+201C"}
    single_quotes = {"'": "U+0027", "'": "U+2018"}

    for unicode_val in double_quotes.values():
        unicode_char = unicode_to_char(unicode_val)
        if unicode_char in text:
            text = text.replace(unicode_char, '"')

    for unicode_val in single_quotes.values():
        unicode_char = unicode_to_char(unicode_val)
        if unicode_char in text:
            text = text.replace(unicode_char, "'")

    return text


def unicode_to_char(unicode_val: str) -> str:
    """Converts a Unicode value to a character."""
    return chr(int(unicode_val.replace("U+", ""), 16))
'''

    # Optimization candidate 9 pattern
    # Use proper Unicode escape sequences for the curly quote characters as keys
    optimized_code = '''def unicode_to_char(unicode_val: str) -> str:
    """Converts a Unicode value to a character."""
    return chr(int(unicode_val.replace("U+", ""), 16))

double_quotes = {'"': "U+0022", '\u201c': "U+201C", '\u201d': "U+201D"}
single_quotes = {"'": "U+0027", '\u2018': "U+2018", '\u2019': "U+2019"}

_translation_table = {}

_double_standard_ord = ord('"')
for unicode_val in double_quotes.values():
    ch = unicode_to_char(unicode_val)
    _translation_table[ord(ch)] = _double_standard_ord

_single_standard_ord = ord("'")
for unicode_val in single_quotes.values():
    ch = unicode_to_char(unicode_val)
    _translation_table[ord(ch)] = _single_standard_ord


def standardize_quotes(text: str) -> str:
    """Converts all unicode quotes to standard ASCII quotes."""
    return text.translate(_translation_table)
'''

    result = add_global_assignments(optimized_code, original_code)

    # Verify the code is valid and executes without NameError
    try:
        compiled = compile(result, "<test>", "exec")
        namespace = {}
        exec(compiled, namespace)

        # Verify the translation table is correctly populated by the for-loops
        # This verifies that the for-loops execute correctly and call unicode_to_char
        translation_table = namespace.get("_translation_table")
        assert translation_table is not None, "_translation_table should be defined"
        # 8220 = U+201C (left double quote), 8221 = U+201D (right double quote)
        # 8216 = U+2018 (left single quote), 8217 = U+2019 (right single quote)
        assert 8220 in translation_table, "Left double quote (U+201C) should be in table"
        assert 8221 in translation_table, "Right double quote (U+201D) should be in table"
        assert 8216 in translation_table, "Left single quote (U+2018) should be in table"
        assert 8217 in translation_table, "Right single quote (U+2019) should be in table"
        # Verify mappings to ASCII quotes
        assert translation_table[8220] == 34, "Left double quote should map to ASCII double quote"
        assert translation_table[8221] == 34, "Right double quote should map to ASCII double quote"
        assert translation_table[8216] == 39, "Left single quote should map to ASCII single quote"
        assert translation_table[8217] == 39, "Right single quote should map to ASCII single quote"

    except NameError as e:
        raise AssertionError(
            f"Candidate 9 pattern failed with NameError: {e}\n\nGenerated code:\n{result}"
        ) from e

    # Verify ordering: unicode_to_char must come before module-level for-loops
    # Use \nfor to find module-level (unindented) for-loops, not those inside functions
    func_pos = result.index("def unicode_to_char")
    first_loop = result.index("\nfor unicode_val in double_quotes.values()")
    second_loop = result.index("\nfor unicode_val in single_quotes.values()")

    assert func_pos < first_loop, "unicode_to_char must be defined before first for-loop"
    assert func_pos < second_loop, "unicode_to_char must be defined before second for-loop"
    assert first_loop < second_loop, "For-loops should maintain their relative order"


def test_standardize_quotes_testgen_postprocessing_with_translation_table() -> None:
    """Test end-to-end postprocessing pipeline for standardize_quotes with translation table pattern.

    This simulates the testgen endpoint returning generated tests for an optimization
    that uses module-level for-loops to build a translation table, and verifies
    the postprocessing pipeline (add_global_assignments + test postprocessing) works correctly.
    """
    from codeflash.code_utils.code_extractor import add_global_assignments
    from codeflash.code_utils.edit_generated_tests import remove_functions_from_generated_tests
    from codeflash.models.models import GeneratedTests, GeneratedTestsList

    # Original code (what we're optimizing)
    original_code = '''def standardize_quotes(text: str) -> str:
    """Converts all unicode quotes to standard ASCII quotes."""
    return text


def unicode_to_char(unicode_val: str) -> str:
    """Converts a Unicode value to a character."""
    return chr(int(unicode_val.replace("U+", ""), 16))
'''

    # Simulated testgen optimization response with module-level for-loops
    # This pattern builds a translation table at module level
    optimized_code = '''def unicode_to_char(unicode_val: str) -> str:
    """Converts a Unicode value to a character."""
    return chr(int(unicode_val.replace("U+", ""), 16))

_DOUBLE_QUOTE_CODES = ["U+0022", "U+201C", "U+201D"]
_SINGLE_QUOTE_CODES = ["U+0027", "U+2018", "U+2019"]

_QUOTE_TABLE = {}

for _code in _DOUBLE_QUOTE_CODES:
    _ch = unicode_to_char(_code)
    _QUOTE_TABLE[ord(_ch)] = ord('"')

for _code in _SINGLE_QUOTE_CODES:
    _ch = unicode_to_char(_code)
    _QUOTE_TABLE[ord(_ch)] = ord("'")


def standardize_quotes(text: str) -> str:
    """Converts all unicode quotes to standard ASCII quotes."""
    return text.translate(_QUOTE_TABLE)
'''

    # Simulated generated test code
    generated_test_source = '''import pytest
from module import standardize_quotes

def test_standardize_quotes_basic():
    """Test basic quote standardization."""
    result = standardize_quotes("Hello world")
    assert result == "Hello world"

def test_standardize_quotes_unicode_double():
    """Test unicode double quote conversion."""
    result = standardize_quotes("Say \\u201chello\\u201d")
    assert '"' in result

def test_standardize_quotes_empty():
    """Test empty string."""
    result = standardize_quotes("")
    assert result == ""
'''

    # Step 1: Process optimization code through add_global_assignments
    processed_optimization = add_global_assignments(optimized_code, original_code)

    # Verify optimization code compiles and executes without NameError
    try:
        compiled = compile(processed_optimization, "<test>", "exec")
        namespace = {}
        exec(compiled, namespace)

        # Verify the translation table is populated
        quote_table = namespace.get("_QUOTE_TABLE")
        assert quote_table is not None, "_QUOTE_TABLE should be defined"
        assert 8220 in quote_table, "Left double quote (U+201C) should be in table"
        assert 8221 in quote_table, "Right double quote (U+201D) should be in table"
    except NameError as e:
        raise AssertionError(f"Optimization code failed with NameError: {e}\n\n{processed_optimization}") from e

    # Step 2: Process generated tests through remove_functions_from_generated_tests
    generated_test = GeneratedTests(
        generated_original_test_source=generated_test_source,
        instrumented_behavior_test_source="",
        instrumented_perf_test_source="",
        behavior_file_path=Path("test_standardize_quotes.py"),
        perf_file_path=Path("test_standardize_quotes_perf.py"),
    )
    generated_tests_list = GeneratedTestsList(generated_tests=[generated_test])

    # Simulate removing a failed test
    processed_tests = remove_functions_from_generated_tests(
        generated_tests_list, ["test_standardize_quotes_empty"]
    )

    # Verify the test was removed and others remain
    result_source = processed_tests.generated_tests[0].generated_original_test_source
    assert "test_standardize_quotes_basic" in result_source
    assert "test_standardize_quotes_unicode_double" in result_source
    assert "test_standardize_quotes_empty" not in result_source


def test_standardize_quotes_testgen_postprocessing_with_dict_comprehension() -> None:
    """Test postprocessing pipeline for standardize_quotes with dict comprehension pattern.

    This simulates an optimization that uses dict comprehension with calls to
    a helper function, ensuring the global assignments are correctly ordered.
    """
    from codeflash.code_utils.code_extractor import add_global_assignments
    from codeflash.code_utils.edit_generated_tests import add_runtime_comments_to_generated_tests
    from codeflash.models.models import GeneratedTests, GeneratedTestsList

    # Original code
    original_code = '''def standardize_quotes(text: str) -> str:
    """Converts all unicode quotes to standard ASCII quotes."""
    return text


def unicode_to_char(unicode_val: str) -> str:
    """Converts a Unicode value to a character."""
    return chr(int(unicode_val.replace("U+", ""), 16))
'''

    # Optimization using dict comprehension (depends on unicode_to_char being defined first)
    optimized_code = '''def unicode_to_char(unicode_val: str) -> str:
    """Converts a Unicode value to a character."""
    return chr(int(unicode_val.replace("U+", ""), 16))

_DOUBLE_UNICODES = {"U+0022": '"', "U+201C": '"', "U+201D": '"'}
_SINGLE_UNICODES = {"U+0027": "'", "U+2018": "'", "U+2019": "'"}

# Build translation table using comprehension
_TRANSLATION = {
    ord(unicode_to_char(code)): ord(target)
    for mapping in [_DOUBLE_UNICODES, _SINGLE_UNICODES]
    for code, target in mapping.items()
}


def standardize_quotes(text: str) -> str:
    """Converts all unicode quotes to standard ASCII quotes."""
    return text.translate(_TRANSLATION)
'''

    # Process optimization code
    processed_optimization = add_global_assignments(optimized_code, original_code)

    # Verify code compiles and translation dict is built
    try:
        compiled = compile(processed_optimization, "<test>", "exec")
        namespace = {}
        exec(compiled, namespace)

        translation = namespace.get("_TRANSLATION")
        assert translation is not None, "_TRANSLATION should be defined"
        # Verify the mapping contains unicode quote codes
        assert len(translation) >= 4, "Translation table should have at least 4 entries"
    except NameError as e:
        raise AssertionError(f"Dict comprehension pattern failed with NameError: {e}") from e

    # Create mock generated tests with runtime data
    generated_test_source = '''def test_standardize_double_quotes():
    result = standardize_quotes("\\u201chello\\u201d")
    assert result == '"hello"'

def test_standardize_single_quotes():
    result = standardize_quotes("\\u2018world\\u2019")
    assert result == "'world'"
'''

    generated_test = GeneratedTests(
        generated_original_test_source=generated_test_source,
        instrumented_behavior_test_source="",
        instrumented_perf_test_source="",
        behavior_file_path=Path("test_quotes.py"),
        perf_file_path=Path("test_quotes_perf.py"),
    )
    generated_tests_list = GeneratedTestsList(generated_tests=[generated_test])

    # Mock runtime data for add_runtime_comments_to_generated_tests
    # (Empty dicts since we don't have actual runtime data in this test)
    original_runtimes: dict = {}
    optimized_runtimes: dict = {}

    # Process through runtime comments (should handle empty runtimes gracefully)
    processed_tests = add_runtime_comments_to_generated_tests(
        generated_tests_list, original_runtimes, optimized_runtimes
    )

    # Verify tests are still valid
    result_source = processed_tests.generated_tests[0].generated_original_test_source
    assert "test_standardize_double_quotes" in result_source
    assert "test_standardize_single_quotes" in result_source


def test_standardize_quotes_testgen_full_pipeline_integration() -> None:
    """Test complete integration of testgen postprocessing pipeline for standardize_quotes.

    This test simulates the full flow:
    1. Original code with the function to optimize
    2. LLM-generated optimization with module-level for-loops
    3. Processing through add_global_assignments
    4. Generated tests processing through remove_functions + add_runtime_comments
    5. Final verification that everything compiles and is correct
    """
    from codeflash.code_utils.code_extractor import add_global_assignments
    from codeflash.code_utils.edit_generated_tests import (
        add_runtime_comments_to_generated_tests,
        remove_functions_from_generated_tests,
    )
    from codeflash.models.models import GeneratedTests, GeneratedTestsList

    # Original FTO code
    original_code = '''def standardize_quotes(text: str) -> str:
    """Converts all unicode quotes to standard ASCII quotes.

    Args:
        text: Input text that may contain unicode quotes.

    Returns:
        Text with unicode quotes replaced by ASCII quotes.
    """
    double_quotes = {'"': "U+0022", '\u201c': "U+201C", '\u201d': "U+201D"}
    single_quotes = {"'": "U+0027", '\u2018': "U+2018", '\u2019': "U+2019"}

    for char, unicode_val in double_quotes.items():
        if char != '"':
            text = text.replace(char, '"')

    for char, unicode_val in single_quotes.items():
        if char != "'":
            text = text.replace(char, "'")

    return text
'''

    # LLM-generated optimization with a NEW helper function and module-level call
    # This tests that add_global_assignments transfers new function definitions
    optimized_code = '''def _build_translation_table() -> dict[int, int]:
    """Build translation table for quote standardization."""
    table = {}
    # Double quotes
    for code in [0x0022, 0x201C, 0x201D]:
        table[code] = 0x0022  # Map to ASCII double quote
    # Single quotes
    for code in [0x0027, 0x2018, 0x2019]:
        table[code] = 0x0027  # Map to ASCII single quote
    return table

_QUOTE_TRANSLATION_TABLE = _build_translation_table()


def standardize_quotes(text: str) -> str:
    """Converts all unicode quotes to standard ASCII quotes."""
    return text.translate(_QUOTE_TRANSLATION_TABLE)
'''

    # Step 1: Process optimization through add_global_assignments
    processed_code = add_global_assignments(optimized_code, original_code)

    # Verify optimization code structure - new function should be transferred
    assert "_build_translation_table" in processed_code, "New helper function should be present"
    assert "_QUOTE_TRANSLATION_TABLE = _build_translation_table()" in processed_code

    # Verify code executes without errors
    namespace = {}
    exec(compile(processed_code, "<test>", "exec"), namespace)

    # Verify the translation table is correctly built
    table = namespace["_QUOTE_TRANSLATION_TABLE"]
    assert table[0x201C] == 0x0022, "Left double quote should map to ASCII double quote"
    assert table[0x201D] == 0x0022, "Right double quote should map to ASCII double quote"
    assert table[0x2018] == 0x0027, "Left single quote should map to ASCII single quote"
    assert table[0x2019] == 0x0027, "Right single quote should map to ASCII single quote"

    # Step 2: Create mock generated tests
    generated_test_source = '''import pytest

def test_standardize_quotes_no_change():
    """Test text without unicode quotes."""
    result = standardize_quotes("Hello, World!")
    assert result == "Hello, World!"

def test_standardize_quotes_double_unicode():
    """Test double quote unicode conversion."""
    text = "She said \\u201cHello\\u201d"
    result = standardize_quotes(text)
    assert result == 'She said "Hello"'

def test_standardize_quotes_single_unicode():
    """Test single quote unicode conversion."""
    text = "It\\u2019s a test"
    result = standardize_quotes(text)
    assert result == "It's a test"

def test_standardize_quotes_mixed():
    """Test mixed quote types."""
    text = "\\u201cIt\\u2019s \\u2018great\\u2019\\u201d"
    result = standardize_quotes(text)
    assert result == '"It\\'s \\'great\\'\"'

def test_standardize_quotes_failing():
    """This test will be removed as failed."""
    result = standardize_quotes("test")
    assert result == "wrong"
'''

    generated_test = GeneratedTests(
        generated_original_test_source=generated_test_source,
        instrumented_behavior_test_source="instrumented_behavior_placeholder",
        instrumented_perf_test_source="instrumented_perf_placeholder",
        behavior_file_path=Path("tests/test_standardize_quotes__unit_test_0.py"),
        perf_file_path=Path("tests/test_standardize_quotes__perf_test_0.py"),
    )
    generated_tests_list = GeneratedTestsList(generated_tests=[generated_test])

    # Step 3: Remove failed tests
    tests_to_remove = ["test_standardize_quotes_failing"]
    processed_tests = remove_functions_from_generated_tests(generated_tests_list, tests_to_remove)

    result_source = processed_tests.generated_tests[0].generated_original_test_source
    assert "test_standardize_quotes_no_change" in result_source
    assert "test_standardize_quotes_double_unicode" in result_source
    assert "test_standardize_quotes_single_unicode" in result_source
    assert "test_standardize_quotes_mixed" in result_source
    assert "test_standardize_quotes_failing" not in result_source

    # Step 4: Add runtime comments (with empty data to test graceful handling)
    final_tests = add_runtime_comments_to_generated_tests(processed_tests, {}, {})

    # Verify final output is still valid Python
    final_source = final_tests.generated_tests[0].generated_original_test_source
    compile(final_source, "<test>", "exec")  # Should not raise

    # Count remaining test functions
    test_count = final_source.count("def test_")
    assert test_count == 4, f"Should have 4 tests after removing failed one, got {test_count}"


def test_match_statement_pattern_names():
    """Test that match statement patterns define names correctly."""
    from codeflash.code_utils.code_extractor import get_statement_defined_names

    # Test basic match with MatchAs
    code = """
match value:
    case [x, y]:
        result = x + y
    case {"name": name}:
        other = name
"""
    module = cst.parse_module(code)
    match_stmt = module.body[0]
    defined_names = get_statement_defined_names(match_stmt)

    # Should find x, y from first case, name from second case, result, other from bodies
    assert "x" in defined_names, "Pattern variable 'x' should be defined"
    assert "y" in defined_names, "Pattern variable 'y' should be defined"
    assert "name" in defined_names, "Pattern variable 'name' should be defined"
    assert "result" in defined_names, "Body variable 'result' should be defined"
    assert "other" in defined_names, "Body variable 'other' should be defined"


def test_match_statement_star_pattern():
    """Test that match statement star patterns define names correctly."""
    from codeflash.code_utils.code_extractor import get_statement_defined_names

    code = """
match items:
    case [first, *rest]:
        total = sum(rest)
"""
    module = cst.parse_module(code)
    match_stmt = module.body[0]
    defined_names = get_statement_defined_names(match_stmt)

    assert "first" in defined_names, "Pattern variable 'first' should be defined"
    assert "rest" in defined_names, "Star pattern variable 'rest' should be defined"
    assert "total" in defined_names, "Body variable 'total' should be defined"


def test_match_statement_as_pattern():
    """Test that match statement 'as' patterns define names correctly."""
    from codeflash.code_utils.code_extractor import get_statement_defined_names

    code = """
match obj:
    case [a, b] as both:
        use = both
"""
    module = cst.parse_module(code)
    match_stmt = module.body[0]
    defined_names = get_statement_defined_names(match_stmt)

    assert "a" in defined_names, "Pattern variable 'a' should be defined"
    assert "b" in defined_names, "Pattern variable 'b' should be defined"
    assert "both" in defined_names, "'as' capture variable 'both' should be defined"


def test_match_statement_mapping_rest():
    """Test that match statement mapping rest patterns define names correctly."""
    from codeflash.code_utils.code_extractor import get_statement_defined_names

    code = """
match config:
    case {"known": val, **rest}:
        extra = rest
"""
    module = cst.parse_module(code)
    match_stmt = module.body[0]
    defined_names = get_statement_defined_names(match_stmt)

    assert "val" in defined_names, "Mapping value 'val' should be defined"
    assert "rest" in defined_names, "Mapping rest '**rest' should be defined"


def test_comprehension_variable_not_dependency():
    """Test that comprehension variables aren't external dependencies."""
    from codeflash.code_utils.code_extractor import get_statement_dependencies

    code = "result = [x * 2 for x in items]"
    module = cst.parse_module(code)
    deps = get_statement_dependencies(module.body[0])

    assert "x" not in deps, "Comprehension variable 'x' should not be a dependency"
    assert "items" in deps, "Iterable 'items' should be a dependency"


def test_nested_comprehension_variables_not_dependencies():
    """Test that nested comprehension variables aren't external dependencies."""
    from codeflash.code_utils.code_extractor import get_statement_dependencies

    code = "matrix = [[row[col] for col in range(n)] for row in data]"
    module = cst.parse_module(code)
    deps = get_statement_dependencies(module.body[0])

    assert "row" not in deps, "Outer comprehension variable 'row' should not be a dependency"
    assert "col" not in deps, "Inner comprehension variable 'col' should not be a dependency"
    assert "data" in deps, "Outer iterable 'data' should be a dependency"
    assert "n" in deps, "Inner range argument 'n' should be a dependency"
    assert "range" in deps, "Built-in 'range' should be a dependency"


def test_dict_comprehension_variables_not_dependencies():
    """Test that dict comprehension variables aren't external dependencies."""
    from codeflash.code_utils.code_extractor import get_statement_dependencies

    code = "mapping = {k: v * 2 for k, v in items.items()}"
    module = cst.parse_module(code)
    deps = get_statement_dependencies(module.body[0])

    assert "k" not in deps, "Dict comprehension key variable 'k' should not be a dependency"
    assert "v" not in deps, "Dict comprehension value variable 'v' should not be a dependency"
    assert "items" in deps, "Iterable 'items' should be a dependency"


def test_generator_expression_variable_not_dependency():
    """Test that generator expression variables aren't external dependencies."""
    from codeflash.code_utils.code_extractor import get_statement_dependencies

    code = "gen = (x for x in items if x > 0)"
    module = cst.parse_module(code)
    deps = get_statement_dependencies(module.body[0])

    assert "x" not in deps, "Generator expression variable 'x' should not be a dependency"
    assert "items" in deps, "Iterable 'items' should be a dependency"


def test_lambda_parameter_not_dependency():
    """Test that lambda parameters aren't external dependencies."""
    from codeflash.code_utils.code_extractor import get_statement_dependencies

    code = "handler = lambda x, y: x + y + z"
    module = cst.parse_module(code)
    deps = get_statement_dependencies(module.body[0])

    assert "x" not in deps, "Lambda parameter 'x' should not be a dependency"
    assert "y" not in deps, "Lambda parameter 'y' should not be a dependency"
    assert "z" in deps, "Free variable 'z' should be a dependency"


def test_lambda_star_args_not_dependency():
    """Test that lambda *args and **kwargs aren't external dependencies."""
    from codeflash.code_utils.code_extractor import get_statement_dependencies

    code = "handler = lambda *args, **kwargs: process(args, kwargs, config)"
    module = cst.parse_module(code)
    deps = get_statement_dependencies(module.body[0])

    assert "args" not in deps, "Lambda *args should not be a dependency"
    assert "kwargs" not in deps, "Lambda **kwargs should not be a dependency"
    assert "process" in deps, "Free variable 'process' should be a dependency"
    assert "config" in deps, "Free variable 'config' should be a dependency"


def test_circular_dependency_warning(caplog):
    """Test that circular dependencies produce a warning."""
    import logging

    from codeflash.code_utils.code_extractor import _sort_statements_by_dependencies

    code = """
x = y + 1
y = x + 1
"""
    module = cst.parse_module(code)

    with caplog.at_level(logging.WARNING):
        result = _sort_statements_by_dependencies(list(module.body))

    # Should emit a warning about circular dependency
    assert any("Circular dependency detected" in record.message for record in caplog.records), (
        "Should log a warning about circular dependencies"
    )
    # Should return original order when cycle is detected
    assert len(result) == 2, "Should return all statements"


def test_add_global_assignments_with_match_statement():
    """Test that match statements in optimized code are handled correctly."""
    from codeflash.code_utils.code_extractor import add_global_assignments

    original_code = """def foo():
    pass
"""

    optimized_code = """
match config:
    case {"type": t}:
        result = t
    case _:
        result = "default"

use_result = result

def foo():
    pass
"""

    result = add_global_assignments(optimized_code, original_code)

    # Verify the code contains the match statement
    assert "match config" in result, "Match statement should be in result"
    assert "use_result = result" in result, "Variable using match result should be in result"

    # The code should be syntactically valid (but may have runtime NameErrors
    # due to undefined 'config')
    compiled = compile(result, "<test>", "exec")
    assert compiled is not None


def test_comprehension_in_global_assignment():
    """Test that global assignments with comprehensions work correctly."""
    from codeflash.code_utils.code_extractor import add_global_assignments

    original_code = """def foo():
    pass
"""

    optimized_code = """
items = [1, 2, 3, 4, 5]
doubled = [x * 2 for x in items]
filtered = [x for x in doubled if x > 4]

def foo():
    pass
"""

    result = add_global_assignments(optimized_code, original_code)

    # Verify the code is valid and executes without NameError
    try:
        compiled = compile(result, "<test>", "exec")
        namespace = {}
        exec(compiled, namespace)
        assert namespace["doubled"] == [2, 4, 6, 8, 10]
        assert namespace["filtered"] == [6, 8, 10]
    except NameError as e:
        msg = f"Comprehension test failed with NameError: {e}\n\nGenerated code:\n{result}"
        raise AssertionError(msg) from e


def test_lambda_in_global_assignment():
    """Test that global assignments with lambdas work correctly."""
    from codeflash.code_utils.code_extractor import add_global_assignments

    original_code = """def foo():
    pass
"""

    optimized_code = """
multiplier = 2
scale = lambda x: x * multiplier

def foo():
    pass
"""

    result = add_global_assignments(optimized_code, original_code)

    # Verify the code is valid and executes without NameError
    try:
        compiled = compile(result, "<test>", "exec")
        namespace = {}
        exec(compiled, namespace)
        assert namespace["scale"](5) == 10
    except NameError as e:
        msg = f"Lambda test failed with NameError: {e}\n\nGenerated code:\n{result}"
        raise AssertionError(msg) from e
