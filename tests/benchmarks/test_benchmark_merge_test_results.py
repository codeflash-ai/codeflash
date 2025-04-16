from codeflash.models.models import FunctionTestInvocation, InvocationId, TestResults, TestType
from codeflash.verification.parse_test_output import merge_test_results


def generate_test_invocations(count=100):
    """Generate a set number of test invocations for benchmarking."""
    test_results_xml = TestResults()
    test_results_bin = TestResults()

    # Generate test invocations in a loop
    for i in range(count):
        iteration_id = str(i * 3 + 5)  # Generate unique iteration IDs

        # XML results - some with None runtime
        test_results_xml.add(
            FunctionTestInvocation(
                id=InvocationId(
                    test_module_path="code_to_optimize.tests.unittest.test_bubble_sort",
                    test_class_name="TestPigLatin",
                    test_function_name="test_sort",
                    function_getting_tested="sorter",
                    iteration_id=iteration_id,
                ),
                file_name="/tmp/tests/unittest/test_bubble_sort__perfinstrumented.py",
                did_pass=True,
                runtime=None if i % 3 == 0 else i * 100,  # Vary runtime values
                test_framework="unittest",
                test_type=TestType.EXISTING_UNIT_TEST,
                return_value=None,
                timed_out=False,
                loop_index=i,
            )
        )

        # Binary results - with actual runtime values
        test_results_bin.add(
            FunctionTestInvocation(
                id=InvocationId(
                    test_module_path="code_to_optimize.tests.unittest.test_bubble_sort",
                    test_class_name="TestPigLatin",
                    test_function_name="test_sort",
                    function_getting_tested="sorter",
                    iteration_id=iteration_id,
                ),
                file_name="/tmp/tests/unittest/test_bubble_sort__perfinstrumented.py",
                did_pass=True,
                runtime=500 + i * 20,  # Generate varying runtime values
                test_framework="unittest",
                test_type=TestType.EXISTING_UNIT_TEST,
                return_value=None,
                timed_out=False,
                loop_index=i,
            )
        )

    return test_results_xml, test_results_bin


def run_merge_benchmark(count=100):
    test_results_xml, test_results_bin = generate_test_invocations(count)

    # Perform the merge operation that will be benchmarked
    merge_test_results(
        xml_test_results=test_results_xml,
        bin_test_results=test_results_bin,
        test_framework="unittest"
    )


def test_benchmark_merge_test_results(benchmark):
    benchmark(run_merge_benchmark, 1000)  # Default to 100 test invocations
