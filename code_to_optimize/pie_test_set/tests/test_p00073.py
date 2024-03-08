from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00073_0():
    input_content = "6\n4\n7\n9\n0\n0"
    expected_output = "96.000000\n184.192455"
    run_pie_test_case("../p00073.py", input_content, expected_output)


def test_problem_p00073_1():
    input_content = "6\n4\n7\n9\n0\n0"
    expected_output = "96.000000\n184.192455"
    run_pie_test_case("../p00073.py", input_content, expected_output)
