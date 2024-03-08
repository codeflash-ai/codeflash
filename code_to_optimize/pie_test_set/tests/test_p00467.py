from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00467_0():
    input_content = "10 5\n0\n0\n5\n6\n-3\n8\n1\n8\n-4\n0\n1\n3\n5\n1\n5\n10 10\n0\n-1\n-1\n4\n4\n-5\n0\n1\n-6\n0\n1\n5\n2\n4\n6\n5\n5\n4\n1\n6\n0 0"
    expected_output = "5\n6"
    run_pie_test_case("../p00467.py", input_content, expected_output)


def test_problem_p00467_1():
    input_content = "10 5\n0\n0\n5\n6\n-3\n8\n1\n8\n-4\n0\n1\n3\n5\n1\n5\n10 10\n0\n-1\n-1\n4\n4\n-5\n0\n1\n-6\n0\n1\n5\n2\n4\n6\n5\n5\n4\n1\n6\n0 0"
    expected_output = "5\n6"
    run_pie_test_case("../p00467.py", input_content, expected_output)
