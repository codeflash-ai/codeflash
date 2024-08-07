import os.path
import tempfile

from codeflash.discovery.functions_to_optimize import (
    find_all_functions_in_file,
    get_functions_to_optimize,
    inspect_top_level_functions_or_methods,
)
from codeflash.verification.verification_utils import TestConfig


def test_function_eligible_for_optimization() -> None:
    function = """def test_function_eligible_for_optimization():
    a = 5
    return a**2
    """
    functions_found = {}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(function)
        f.flush()
        functions_found = find_all_functions_in_file(f.name)
    assert functions_found[f.name][0].function_name == "test_function_eligible_for_optimization"

    # Has no return statement
    function = """def test_function_not_eligible_for_optimization():
    a = 5
    print(a)
    """
    functions_found = {}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(function)
        f.flush()
        functions_found = find_all_functions_in_file(f.name)
    assert len(functions_found[f.name]) == 0


def test_find_top_level_function_or_method():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(
            """def functionA():
    def functionB():
        return 5
    class E:
        def functionF():
            pass
    return functionA()
class A:
    def functionC():
        def functionD():
            pass
        return 6
class AirbyteEntrypoint(object):
    @staticmethod
    def handle_record_counts(message: AirbyteMessage, stream_message_count: DefaultDict[HashableStreamDescriptor, float]) -> AirbyteMessage:
        match message.type:
            case Type.RECORD:
                stream_message_count[HashableStreamDescriptor(name=message.record.stream, namespace=message.record.namespace)] += 1.0
            case Type.STATE:
                stream_descriptor = message_utils.get_stream_descriptor(message)

                # Set record count from the counter onto the state message
                message.state.sourceStats = message.state.sourceStats or AirbyteStateStats()
                message.state.sourceStats.recordCount = stream_message_count.get(stream_descriptor, 0.0)

                # Reset the counter
                stream_message_count[stream_descriptor] = 0.0
        return message
    @classmethod
    def functionE(cls, num):
        return AirbyteEntrypoint.handle_record_counts(num)
    """,
        )
        f.flush()
        assert inspect_top_level_functions_or_methods(f.name, "functionA").is_top_level
        assert not inspect_top_level_functions_or_methods(f.name, "functionB").is_top_level
        assert inspect_top_level_functions_or_methods(f.name, "functionC", class_name="A").is_top_level
        assert not inspect_top_level_functions_or_methods(f.name, "functionD", class_name="A").is_top_level
        assert not inspect_top_level_functions_or_methods(f.name, "functionF", class_name="E").is_top_level
        assert not inspect_top_level_functions_or_methods(f.name, "functionA").has_args
        staticmethod_func = inspect_top_level_functions_or_methods(
            f.name,
            "handle_record_counts",
            class_name=None,
            line_no=15,
        )
        assert staticmethod_func.is_staticmethod
        assert staticmethod_func.staticmethod_class_name == "AirbyteEntrypoint"
        assert inspect_top_level_functions_or_methods(
            f.name,
            "functionE",
            class_name="AirbyteEntrypoint",
        ).is_classmethod


def test_class_method_discovery():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(
            """class A:
    def functionA():
        return True
    def functionB():
        return False
class X:
    def functionA():
        return True
    def functionB():
        return False
def functionA():
    return True""",
        )
        f.flush()
        test_config = TestConfig(tests_root="tests", project_root_path=".", test_framework="pytest")

        functions, functions_count = get_functions_to_optimize(
            optimize_all=None,
            replay_test=None,
            file=f.name,
            only_get_this_function="A.functionA",
            test_cfg=test_config,
            ignore_paths=[""],
            project_root=os.path.dirname(f.name),
            module_root=os.path.dirname(f.name),
        )
        assert len(functions) == 1
        for file in functions:
            assert functions[file][0].qualified_name == "A.functionA"
            assert functions[file][0].function_name == "functionA"
            assert functions[file][0].top_level_parent_name == "A"

        functions, functions_count = get_functions_to_optimize(
            optimize_all=None,
            replay_test=None,
            file=f.name,
            only_get_this_function="X.functionA",
            test_cfg=test_config,
            ignore_paths=[""],
            project_root=os.path.dirname(f.name),
            module_root=os.path.dirname(f.name),
        )
        assert len(functions) == 1
        for file in functions:
            assert functions[file][0].qualified_name == "X.functionA"
            assert functions[file][0].function_name == "functionA"
            assert functions[file][0].top_level_parent_name == "X"

        functions, functions_count = get_functions_to_optimize(
            optimize_all=None,
            replay_test=None,
            file=f.name,
            only_get_this_function="functionA",
            test_cfg=test_config,
            ignore_paths=[""],
            project_root=os.path.dirname(f.name),
            module_root=os.path.dirname(f.name),
        )
        assert len(functions) == 1
        for file in functions:
            assert functions[file][0].qualified_name == "functionA"
            assert functions[file][0].function_name == "functionA"
