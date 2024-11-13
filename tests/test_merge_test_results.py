from codeflash.verification.parse_test_output import merge_test_results
from codeflash.verification.test_results import FunctionTestInvocation, InvocationId, TestResults, TestType


def test_merge_test_results_1():
    test_results_xml = TestResults()
    test_results_xml.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="code_to_optimize.tests.unittest.test_bubble_sort",
                test_class_name="TestPigLatin",
                test_function_name="test_sort",
                function_getting_tested="sorter",
                iteration_id="5",
            ),
            file_name="/tmp/tests/unittest/test_bubble_sort__perfinstrumented.py",
            did_pass=True,
            runtime=None,
            test_framework="unittest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )
    test_results_xml.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="code_to_optimize.tests.unittest.test_bubble_sort",
                test_class_name="TestPigLatin",
                test_function_name="test_sort",
                function_getting_tested="sorter",
                iteration_id="8",
            ),
            file_name="/tmp/tests/unittest/test_bubble_sort__perfinstrumented.py",
            did_pass=True,
            runtime=458,
            test_framework="unittest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )
    test_results_xml.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="code_to_optimize.tests.unittest.test_bubble_sort",
                test_class_name="TestPigLatin",
                test_function_name="test_sort",
                function_getting_tested="sorter",
                iteration_id="11",
            ),
            file_name="/tmp/tests/unittest/test_bubble_sort__perfinstrumented.py",
            did_pass=True,
            runtime=14125,
            test_framework="unittest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )

    test_results_bin = TestResults()
    test_results_bin.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="code_to_optimize.tests.unittest.test_bubble_sort",
                test_class_name="TestPigLatin",
                test_function_name="test_sort",
                function_getting_tested="sorter",
                iteration_id="5",
            ),
            file_name="/tmp/tests/unittest/test_bubble_sort__perfinstrumented.py",
            did_pass=True,
            runtime=667,
            test_framework="unittest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )
    test_results_bin.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="code_to_optimize.tests.unittest.test_bubble_sort",
                test_class_name="TestPigLatin",
                test_function_name="test_sort",
                function_getting_tested="sorter",
                iteration_id="8",
            ),
            file_name="/tmp/tests/unittest/test_bubble_sort__perfinstrumented.py",
            did_pass=True,
            runtime=458,
            test_framework="unittest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )
    test_results_bin.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="code_to_optimize.tests.unittest.test_bubble_sort",
                test_class_name="TestPigLatin",
                test_function_name="test_sort",
                function_getting_tested="sorter",
                iteration_id="11",
            ),
            file_name="/tmp/tests/unittest/test_bubble_sort__perfinstrumented.py",
            did_pass=True,
            runtime=14125,
            test_framework="unittest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )

    expected_merged_results = TestResults()
    expected_merged_results.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="code_to_optimize.tests.unittest.test_bubble_sort",
                test_class_name="TestPigLatin",
                test_function_name="test_sort",
                function_getting_tested="sorter",
                iteration_id="5",
            ),
            file_name="/tmp/tests/unittest/test_bubble_sort__perfinstrumented.py",
            did_pass=True,
            runtime=667,
            test_framework="unittest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )
    expected_merged_results.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="code_to_optimize.tests.unittest.test_bubble_sort",
                test_class_name="TestPigLatin",
                test_function_name="test_sort",
                function_getting_tested="sorter",
                iteration_id="8",
            ),
            file_name="/tmp/tests/unittest/test_bubble_sort__perfinstrumented.py",
            did_pass=True,
            runtime=458,
            test_framework="unittest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )
    expected_merged_results.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="code_to_optimize.tests.unittest.test_bubble_sort",
                test_class_name="TestPigLatin",
                test_function_name="test_sort",
                function_getting_tested="sorter",
                iteration_id="11",
            ),
            file_name="/tmp/tests/unittest/test_bubble_sort__perfinstrumented.py",
            did_pass=True,
            runtime=14125,
            test_framework="unittest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )
    merged_results = merge_test_results(
        xml_test_results=test_results_xml, bin_test_results=test_results_bin, test_framework="unittest"
    )
    assert merged_results == expected_merged_results

    test_results_xml_single = TestResults()
    test_results_xml_single.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="code_to_optimize.tests.unittest.test_bubble_sort",
                test_class_name="TestPigLatin",
                test_function_name="test_sort",
                function_getting_tested="sorter",
                iteration_id=None,
            ),
            file_name="/tmp/tests/unittest/test_bubble_sort__perfinstrumented.py",
            did_pass=True,
            runtime=None,
            test_framework="unittest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )

    merged_results = merge_test_results(
        xml_test_results=test_results_xml_single, bin_test_results=test_results_bin, test_framework="unittest"
    )

    assert merged_results == expected_merged_results

    merged_results = merge_test_results(
        xml_test_results=test_results_xml_single, bin_test_results=TestResults(), test_framework="unittest"
    )

    assert merged_results == test_results_xml_single

    merged_results = merge_test_results(
        xml_test_results=TestResults(), bin_test_results=test_results_bin, test_framework="unittest"
    )

    assert merged_results == TestResults()  # XML Results should always have better coverage than bin results

    test_results_xml_pytest = TestResults()
    test_results_xml_pytest.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="code_to_optimize.tests.unittest.test_bubble_sort",
                test_class_name=None,
                test_function_name="test_sort",
                function_getting_tested="",
                iteration_id=None,
            ),
            file_name="/tmp/tests/unittest/test_bubble_sort__perfinstrumented.py",
            did_pass=True,
            runtime=None,
            test_framework="pytest",
            test_type=TestType.GENERATED_REGRESSION,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )

    test_results_bin_pytest = TestResults()
    test_results_bin_pytest.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="code_to_optimize.tests.unittest.test_bubble_sort",
                test_class_name=None,
                test_function_name="test_sort",
                function_getting_tested="sorter",
                iteration_id="5",
            ),
            file_name="/tmp/tests/unittest/test_bubble_sort__perfinstrumented.py",
            did_pass=True,
            runtime=667,
            test_framework="pytest",
            test_type=TestType.GENERATED_REGRESSION,
            return_value=[2],
            timed_out=False,
            loop_index=1,
        )
    )
    test_results_bin_pytest.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="code_to_optimize.tests.unittest.test_bubble_sort",
                test_class_name=None,
                test_function_name="test_sort",
                function_getting_tested="sorter",
                iteration_id="8",
            ),
            file_name="/tmp/tests/unittest/test_bubble_sort__perfinstrumented.py",
            did_pass=True,
            runtime=458,
            test_framework="pytest",
            test_type=TestType.GENERATED_REGRESSION,
            return_value=[3],
            timed_out=False,
            loop_index=1,
        )
    )

    merged_results = merge_test_results(
        xml_test_results=test_results_xml_pytest, bin_test_results=test_results_bin_pytest, test_framework="unittest"
    )

    assert merged_results == test_results_bin_pytest
