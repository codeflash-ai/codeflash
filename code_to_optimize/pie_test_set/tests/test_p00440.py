from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00440_0():
    input_content = "7 5\n6\n2\n4\n7\n1\n7 5\n6\n2\n0\n4\n7\n0 0"
    expected_output = "2\n4"
    run_pie_test_case("../p00440.py", input_content, expected_output)


def test_problem_p00440_1():
    input_content = "7 5\n6\n2\n4\n7\n1\n7 5\n6\n2\n0\n4\n7\n0 0"
    expected_output = "2\n4"
    run_pie_test_case("../p00440.py", input_content, expected_output)
