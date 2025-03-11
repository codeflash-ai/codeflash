import pathlib
from dataclasses import dataclass

import pytest

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.either import is_successful
from codeflash.models.models import FunctionParent
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.verification_utils import TestConfig


def calculate_something(data):
    return data + 1


def simple_function_with_one_dep(data):
    return calculate_something(data)


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


def recursive_dependency_1(num):
    if num == 0:
        return 0
    num_1 = calculate_something(num)
    return recursive_dependency_1(num) + num_1

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

    function_to_optimize = FunctionToOptimize(
        function_name="topologicalSort",
        file_path=str(file_path),
        parents=[FunctionParent(name="Graph", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )
    func_optimizer = FunctionOptimizer(
        function_to_optimize=function_to_optimize,
        test_cfg=TestConfig(
            tests_root=file_path,
            tests_project_rootdir=file_path.parent,
            project_root_path=file_path.parent,
            test_framework="pytest",
            pytest_cmd="pytest",
        ),
    )
    with open(file_path) as f:
        original_code = f.read()
    ctx_result = func_optimizer.get_code_optimization_context()
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
        code_context.testgen_context_code
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
        return stack"""
    )

def test_recursive_function_context() -> None:
    file_path = pathlib.Path(__file__).resolve()

    function_to_optimize = FunctionToOptimize(
        function_name="recursive",
        file_path=str(file_path),
        parents=[FunctionParent(name="C", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )
    func_optimizer = FunctionOptimizer(
        function_to_optimize=function_to_optimize,
        test_cfg=TestConfig(
            tests_root=file_path,
            tests_project_rootdir=file_path.parent,
            project_root_path=file_path.parent,
            test_framework="pytest",
            pytest_cmd="pytest",
        ),
    )
    with open(file_path) as f:
        original_code = f.read()

    ctx_result = func_optimizer.get_code_optimization_context()
    if not is_successful(ctx_result):
        pytest.fail()
    code_context = ctx_result.unwrap()
    assert len(code_context.helper_functions) == 2
    assert code_context.helper_functions[0].fully_qualified_name == "test_function_dependencies.C.calculate_something_3"
    assert code_context.helper_functions[1].fully_qualified_name == "test_function_dependencies.C.recursive"
    assert (
        code_context.testgen_context_code
        == """class C:
    def calculate_something_3(self, num):
        return num + 1

    def recursive(self, num):
        if num == 0:
            return 0
        num_1 = self.calculate_something_3(num)
        return self.recursive(num) + num_1"""
    )