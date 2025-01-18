import pathlib
from argparse import Namespace
from dataclasses import dataclass

import pytest
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.either import is_successful
from codeflash.models.models import FunctionParent
from codeflash.optimization.function_context import get_function_variables_definitions
from codeflash.optimization.optimizer import Optimizer


def calculate_something(data):
    return data + 1


def simple_function_with_one_dep(data):
    return calculate_something(data)


def test_simple_dependencies() -> None:
    file_path = pathlib.Path(__file__).resolve()
    helper_functions = get_function_variables_definitions(
        FunctionToOptimize("simple_function_with_one_dep", str(file_path), []), str(file_path.parent.resolve())
    )[0]
    assert len(helper_functions) == 1
    assert helper_functions[0].jedi_definition.full_name == "test_function_dependencies.calculate_something"


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

    def function_in_list_comprehension(self):
        return [global_dependency_3(1) for x in range(10)]

    def add_two(self, num):
        return num + 2

    def method_in_list_comprehension(self):
        return [self.add_two(1) for x in range(10)]

    def nested_function(self):
        def nested():
            return global_dependency_3(1)

        return nested() + self.add_two(3)


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

    def recursive(self, num):
        if num == 0:
            return 0
        num_1 = self.calculate_something_3(num)
        return self.recursive(num) + num_1


def test_multiple_classes_dependencies() -> None:
    file_path = pathlib.Path(__file__).resolve()
    helper_functions = get_function_variables_definitions(
        FunctionToOptimize("run", str(file_path), [FunctionParent("C", "ClassDef")]), str(file_path.parent.resolve())
    )

    assert len(helper_functions) == 2
    assert list(map(lambda x: x.fully_qualified_name, helper_functions[0])) == [
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
        FunctionToOptimize("recursive_dependency_1", str(file_path), []), str(file_path.parent.resolve())
    )[0]
    assert len(helper_functions) == 1
    assert helper_functions[0].jedi_definition.full_name == "test_function_dependencies.calculate_something"
    assert helper_functions[0].fully_qualified_name == "test_function_dependencies.calculate_something"


@dataclass
class MyData:
    MyInt: int


def calculate_something_ann(data):
    return data + 1


def simple_function_with_one_dep_ann(data: MyData):
    return calculate_something_ann(data)


def list_comprehension_dependency(data: MyData):
    return [calculate_something(data) for x in range(10)]


def test_simple_dependencies_ann() -> None:
    file_path = pathlib.Path(__file__).resolve()
    helper_functions = get_function_variables_definitions(
        FunctionToOptimize("simple_function_with_one_dep_ann", str(file_path), []), str(file_path.parent.resolve())
    )[0]
    assert len(helper_functions) == 2
    assert helper_functions[0].jedi_definition.full_name == "test_function_dependencies.MyData"
    assert helper_functions[1].jedi_definition.full_name == "test_function_dependencies.calculate_something_ann"


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
            project_root=file_path.parent.resolve(),
            disable_telemetry=True,
            tests_root="tests",
            test_framework="pytest",
            pytest_cmd="pytest",
            experiment_id=None,
            test_project_root=file_path.parent.resolve(),
        )
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
    ctx_result = opt.get_code_optimization_context(function_to_optimize, opt.args.project_root, original_code)
    if not is_successful(ctx_result):
        pytest.fail()
    code_context = ctx_result.unwrap()
    # The code_context above should have the topologicalSortUtil function in it
    assert len(code_context.helper_functions) == 1
    assert (
        code_context.helper_functions[0].jedi_definition.full_name
        == "test_function_dependencies.Graph.topologicalSortUtil"
    )
    assert code_context.helper_functions[0].jedi_definition.name == "topologicalSortUtil"
    assert (
        code_context.helper_functions[0].fully_qualified_name == "test_function_dependencies.Graph.topologicalSortUtil"
    )
    assert code_context.helper_functions[0].qualified_name == "Graph.topologicalSortUtil"
    assert (
        code_context.code_to_optimize_with_helpers
        == """from collections import defaultdict

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
        FunctionToOptimize("simple_function_with_decorator_dep", str(file_path), []), str(file_path.parent.resolve())
    )[0]
    assert len(helper_functions) == 2
    assert {helper_functions[0][0].definition.full_name, helper_functions[1][0].definition.full_name} == {
        "test_function_dependencies.calculate_something",
        "test_function_dependencies.imalittledecorator",
    }


def test_recursive_function_context() -> None:
    file_path = pathlib.Path(__file__).resolve()
    opt = Optimizer(
        Namespace(
            project_root=file_path.parent.resolve(),
            disable_telemetry=True,
            tests_root="tests",
            test_framework="pytest",
            pytest_cmd="pytest",
            experiment_id=None,
            test_project_root=file_path.parent.resolve(),
        )
    )
    function_to_optimize = FunctionToOptimize(
        function_name="recursive",
        file_path=str(file_path),
        parents=[FunctionParent(name="C", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )
    with open(file_path) as f:
        original_code = f.read()
    ctx_result = opt.get_code_optimization_context(function_to_optimize, opt.args.project_root, original_code)
    if not is_successful(ctx_result):
        pytest.fail()
    code_context = ctx_result.unwrap()
    assert len(code_context.helper_functions) == 2
    assert code_context.helper_functions[0].fully_qualified_name == "test_function_dependencies.C.calculate_something_3"
    assert code_context.helper_functions[1].fully_qualified_name == "test_function_dependencies.C.recursive"
    assert (
        code_context.code_to_optimize_with_helpers
        == """class C:
    def calculate_something_3(self, num):
        return num + 1
    def recursive(self, num):
        if num == 0:
            return 0
        num_1 = self.calculate_something_3(num)
        return self.recursive(num) + num_1
"""
    )


def test_list_comprehension_dependency() -> None:
    file_path = pathlib.Path(__file__).resolve()
    helper_functions = get_function_variables_definitions(
        FunctionToOptimize("list_comprehension_dependency", str(file_path), []), str(file_path.parent.resolve())
    )[0]
    assert len(helper_functions) == 2
    assert helper_functions[0].jedi_definition.full_name == "test_function_dependencies.MyData"
    assert helper_functions[1].jedi_definition.full_name == "test_function_dependencies.calculate_something"


def test_function_in_method_list_comprehension() -> None:
    file_path = pathlib.Path(__file__).resolve()
    function_to_optimize = FunctionToOptimize(
        function_name="function_in_list_comprehension",
        file_path=str(file_path),
        parents=[FunctionParent(name="A", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    helper_functions = get_function_variables_definitions(function_to_optimize, str(file_path.parent.resolve()))[0]

    assert len(helper_functions) == 1
    assert helper_functions[0].jedi_definition.full_name == "test_function_dependencies.global_dependency_3"


def test_method_in_method_list_comprehension() -> None:
    file_path = pathlib.Path(__file__).resolve()
    function_to_optimize = FunctionToOptimize(
        function_name="method_in_list_comprehension",
        file_path=str(file_path),
        parents=[FunctionParent(name="A", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    helper_functions = get_function_variables_definitions(function_to_optimize, str(file_path.parent.resolve()))[0]

    assert len(helper_functions) == 1
    assert helper_functions[0].jedi_definition.full_name == "test_function_dependencies.A.add_two"


def test_nested_method() -> None:
    file_path = pathlib.Path(__file__).resolve()
    function_to_optimize = FunctionToOptimize(
        function_name="nested_function",
        file_path=str(file_path),
        parents=[FunctionParent(name="A", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    helper_functions = get_function_variables_definitions(function_to_optimize, str(file_path.parent.resolve()))[0]

    # The nested function should be included in the helper functions
    assert len(helper_functions) == 1
    assert helper_functions[0].jedi_definition.full_name == "test_function_dependencies.A.add_two"
