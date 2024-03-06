import pathlib
from dataclasses import dataclass

from codeflash.discovery.functions_to_optimize import FunctionToOptimize, FunctionParent
from codeflash.optimization.function_context import get_function_variables_definitions


def calculate_something(data):
    return data + 1


def simple_function_with_one_dep(data):
    return calculate_something(data)


def test_simple_dependencies():
    file_path = pathlib.Path(__file__).resolve()
    dependent_functions = get_function_variables_definitions(
        FunctionToOptimize("simple_function_with_one_dep", str(file_path), []),
        str(file_path.parent.resolve()),
    )
    assert len(dependent_functions) == 1
    assert (
        dependent_functions[0][0].definition.full_name
        == "test_function_dependencies.calculate_something"
    )


def global_dependency_1(num):
    return num + 1


def global_dependency_2(num):
    return num + 1


def global_dependency_3(num):
    return num + 1


class A:
    def calculate_something_1(self, num):
        return num + 1

    def run(self):
        a = 1
        b = self.calculate_something_1(a)
        c = global_dependency_1(b)
        return c


class B:
    def calculate_something_2(self, num):
        return num + 1

    def run(self):
        a = 1
        b = self.calculate_something_2(a)
        c = global_dependency_2(b)
        return c


class C:
    def calculate_something_3(self, num):
        return num + 1

    def run(self):
        a = 1
        b = self.calculate_something_3(a)
        c = global_dependency_3(b)
        return c


def test_multiple_classes_dependencies():
    # TODO: Check if C.run only gets calculate_something_3 as dependency and likewise for other classes
    file_path = pathlib.Path(__file__).resolve()
    dependent_functions = get_function_variables_definitions(
        FunctionToOptimize("run", str(file_path), [FunctionParent("C", "ClassDef")]),
        str(file_path.parent.resolve()),
    )

    # assert len(dependent_functions) == 2
    assert list(map(lambda x: x[0].full_name, dependent_functions)) == [
        "test_function_dependencies.C.run.calculate_something_3",
        "test_function_dependencies.C.run.global_dependency_3",
    ]


def recursive_dependency_1(num):
    if num == 0:
        return 0
    num_1 = calculate_something(num)
    return recursive_dependency_1(num) + num_1


def test_recursive_dependency():
    file_path = pathlib.Path(__file__).resolve()
    dependent_functions = get_function_variables_definitions(
        FunctionToOptimize("recursive_dependency_1", str(file_path), []),
        str(file_path.parent.resolve()),
    )
    assert len(dependent_functions) == 1
    assert (
        dependent_functions[0][0].definition.full_name
        == "test_function_dependencies.calculate_something"
    )


@dataclass
class MyData:
    MyInt: int


def calculate_something_ann(data):
    return data + 1


def simple_function_with_one_dep_ann(data: MyData):
    return calculate_something_ann(data)


def test_simple_dependencies_ann():
    file_path = pathlib.Path(__file__).resolve()
    dependent_functions = get_function_variables_definitions(
        FunctionToOptimize("simple_function_with_one_dep_ann", str(file_path), []),
        str(file_path.parent.resolve()),
    )
    assert len(dependent_functions) == 2
    assert dependent_functions[0][0].definition.full_name == "test_function_dependencies.MyData"
    assert (
        dependent_functions[1][0].definition.full_name
        == "test_function_dependencies.calculate_something_ann"
    )
