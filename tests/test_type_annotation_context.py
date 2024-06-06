from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import List

from codeflash.code_utils.code_extractor import get_code
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.optimization.function_context import (
    get_constrained_function_context_and_helper_functions,
)


class CustomType:
    def __init__(self) -> None:
        self.name = None
        self.data: List[int] = []


@dataclass
class CustomDataClass:
    name: str = ""
    data: List[int] = field(default_factory=list)


def function_to_optimize(data: CustomType) -> CustomType:
    name = data.name
    data.data.sort()
    return data


def function_to_optimize2(data: CustomDataClass) -> CustomType:
    name = data.name
    data.data.sort()
    return data


def function_to_optimize3(data: dict[CustomDataClass, list[CustomDataClass]]) -> list[CustomType] | None:
    name = data.name
    data.data.sort()
    return data


def test_function_context_includes_type_annotation() -> None:
    file_path = pathlib.Path(__file__).resolve()
    a, helper_functions, dunder_methods = get_constrained_function_context_and_helper_functions(
        FunctionToOptimize("function_to_optimize", str(file_path), []),
        str(file_path.parent.resolve()),
        """def function_to_optimize(data: CustomType):
    name = data.name
    data.data.sort()
    return data""",
        1000,
    )

    assert len(helper_functions) == 1
    assert helper_functions[0][0].full_name == "test_type_annotation_context.CustomType"


def test_function_context_includes_type_annotation_dataclass() -> None:
    file_path = pathlib.Path(__file__).resolve()
    a, helper_functions, dunder_methods = get_constrained_function_context_and_helper_functions(
        FunctionToOptimize("function_to_optimize2", str(file_path), []),
        str(file_path.parent.resolve()),
        """def function_to_optimize2(data: CustomDataClass) -> CustomType:
    name = data.name
    data.data.sort()
    return data""",
        1000,
    )

    assert len(helper_functions) == 2
    assert helper_functions[0][0].full_name == "test_type_annotation_context.CustomDataClass"
    assert helper_functions[1][0].full_name == "test_type_annotation_context.CustomType"


def test_function_context_works_for_composite_types() -> None:
    file_path = pathlib.Path(__file__).resolve()
    a, helper_functions, dunder_methods = get_constrained_function_context_and_helper_functions(
        FunctionToOptimize("function_to_optimize3", str(file_path), []),
        str(file_path.parent.resolve()),
        """def function_to_optimize3(data: set[CustomDataClass[CustomDataClass, int]]) -> list[CustomType]:
    name = data.name
    data.data.sort()
    return data""",
        1000,
    )

    assert len(helper_functions) == 2
    assert helper_functions[0][0].full_name == "test_type_annotation_context.CustomDataClass"
    assert helper_functions[1][0].full_name == "test_type_annotation_context.CustomType"


def test_function_context_custom_datatype() -> None:
    project_path = pathlib.Path(__file__).parent.parent.resolve() / "code_to_optimize"
    file_path = project_path / "math_utils.py"
    code, contextual_dunder_methods = get_code(
        [FunctionToOptimize("cosine_similarity", str(file_path), [])],
    )
    assert code is not None
    assert contextual_dunder_methods == set()
    a, helper_functions, dunder_methods = get_constrained_function_context_and_helper_functions(
        FunctionToOptimize("cosine_similarity", str(file_path), []),
        str(project_path),
        code,
        1000,
    )

    assert len(helper_functions) == 1
    assert helper_functions[0][0].full_name == "math_utils.Matrix"
