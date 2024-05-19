import pathlib
from argparse import Namespace
from dataclasses import dataclass

import pytest
from codeflash.discovery.functions_to_optimize import FunctionParent, FunctionToOptimize
from codeflash.optimization.function_context import get_function_variables_definitions
from codeflash.optimization.optimizer import Optimizer
from returns.pipeline import is_successful


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
    assert dependent_functions[0][0].definition.full_name == "test_function_dependencies.calculate_something"


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
        "test_function_dependencies.C.calculate_something_3",
        "test_function_dependencies.global_dependency_3",
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
    assert dependent_functions[0][0].definition.full_name == "test_function_dependencies.calculate_something"


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
        dependent_functions[1][0].definition.full_name == "test_function_dependencies.calculate_something_ann"
    )


from collections import defaultdict


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


def test_class_method_dependencies():
    file_path = pathlib.Path(__file__).resolve()
    opt = Optimizer(
        Namespace(
            project_root=str(file_path.parent.resolve()),
            disable_telemetry=False,
            tests_root="tests",
            test_framework="pytest",
            pytest_cmd="pytest",
            experiment_id=None,
        ),
    )
    function_to_optimize = FunctionToOptimize(
        function_name="topologicalSort",
        file_path=str(file_path),
        parents=[FunctionParent(name="Graph", type="ClassDef")],
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
    # The code_context above should have the topologicalSortUtil function in it
    print("hi")
