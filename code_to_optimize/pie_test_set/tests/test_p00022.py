from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00022_0():
    input_content = "7\n-5\n-1\n6\n4\n9\n-6\n-7\n13\n1\n2\n3\n2\n-2\n-1\n1\n2\n3\n2\n1\n-2\n1\n3\n1000\n-200\n201\n0"
    expected_output = "19\n14\n1001"
    run_pie_test_case("../p00022.py", input_content, expected_output)


def test_problem_p00022_1():
    input_content = "7\n-5\n-1\n6\n4\n9\n-6\n-7\n13\n1\n2\n3\n2\n-2\n-1\n1\n2\n3\n2\n1\n-2\n1\n3\n1000\n-200\n201\n0"
    expected_output = "19\n14\n1001"
    run_pie_test_case("../p00022.py", input_content, expected_output)
