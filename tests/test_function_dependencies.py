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


def test_simple_dependencies() -> None:
    file_path = pathlib.Path(__file__).resolve()
    helper_functions = get_function_variables_definitions(
        FunctionToOptimize("simple_function_with_one_dep", str(file_path), []),
        str(file_path.parent.resolve()),
    )[0]
    assert len(helper_functions) == 1
    assert helper_functions[0][0].definition.full_name == "test_function_dependencies.calculate_something"


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


def test_multiple_classes_dependencies() -> None:
    # TODO: Check if C.run only gets calculate_something_3 as dependency and likewise for other classes
    file_path = pathlib.Path(__file__).resolve()
    helper_functions = get_function_variables_definitions(
        FunctionToOptimize("run", str(file_path), [FunctionParent("C", "ClassDef")]),
        str(file_path.parent.resolve()),
    )

    # assert len(helper_functions) == 2
    assert list(map(lambda x: x[0].full_name, helper_functions[0])) == [
        "test_function_dependencies.global_dependency_3",
        "test_function_dependencies.C.calculate_something_3",
    ]


def recursive_dependency_1(num):
    if num == 0:
        return 0
    num_1 = calculate_something(num)
    return recursive_dependency_1(num) + num_1


def test_recursive_dependency() -> None:
    file_path = pathlib.Path(__file__).resolve()
    helper_functions = get_function_variables_definitions(
        FunctionToOptimize("recursive_dependency_1", str(file_path), []),
        str(file_path.parent.resolve()),
    )[0]
    assert len(helper_functions) == 1
    assert helper_functions[0][0].definition.full_name == "test_function_dependencies.calculate_something"


@dataclass
class MyData:
    MyInt: int


def calculate_something_ann(data):
    return data + 1


def simple_function_with_one_dep_ann(data: MyData):
    return calculate_something_ann(data)


def test_simple_dependencies_ann() -> None:
    file_path = pathlib.Path(__file__).resolve()
    helper_functions = get_function_variables_definitions(
        FunctionToOptimize("simple_function_with_one_dep_ann", str(file_path), []),
        str(file_path.parent.resolve()),
    )[0]
    assert len(helper_functions) == 2
    assert helper_functions[0][0].definition.full_name == "test_function_dependencies.MyData"
    assert helper_functions[1][0].definition.full_name == "test_function_dependencies.calculate_something_ann"


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


def test_class_method_dependencies() -> None:
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
    assert len(code_context.helper_functions) == 1
    assert (
        code_context.helper_functions[0][0].definition.full_name
        == "test_function_dependencies.Graph.topologicalSortUtil"
    )
    assert code_context.helper_functions[0][0].definition.name == "topologicalSortUtil"
    assert code_context.helper_functions[0][2] == "Graph.topologicalSortUtil"
    assert code_context.contextual_dunder_methods == {("Graph", "__init__")}
    assert (
        code_context.code_to_optimize_with_helpers
        == """from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.graph = defaultdict(list)
        self.V = vertices  # No. of vertices
    def topologicalSort(self):
        visited = [False] * self.V
        stack = []

        for i in range(self.V):
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

        # Print contents of stack
        return stack
    def topologicalSortUtil(self, v, visited, stack):
        visited[v] = True

        for i in self.graph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

        stack.insert(0, v)
"""
    )


def calculate_something_else(data):
    return data + 1


def imalittledecorator(func):
    def wrapper(data):
        return func(data)

    return wrapper


@imalittledecorator
def simple_function_with_decorator_dep(data):
    return calculate_something_else(data)


@pytest.mark.skip(reason="no decorator dependency support")
def test_decorator_dependencies() -> None:
    file_path = pathlib.Path(__file__).resolve()
    helper_functions = get_function_variables_definitions(
        FunctionToOptimize("simple_function_with_decorator_dep", str(file_path), []),
        str(file_path.parent.resolve()),
    )[0]
    assert len(helper_functions) == 2
    assert {helper_functions[0][0].definition.full_name, helper_functions[1][0].definition.full_name} == {
        "test_function_dependencies.calculate_something",
        "test_function_dependencies.imalittledecorator",
    }
