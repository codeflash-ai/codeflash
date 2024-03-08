from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00169_0():
    input_content = "1\n7 7 7\n7 7 8\n12 1\n10 1 1\n0"
    expected_output = "11\n21\n0\n21\n12"
    run_pie_test_case("../p00169.py", input_content, expected_output)


def test_problem_p00169_1():
    input_content = "1\n7 7 7\n7 7 8\n12 1\n10 1 1\n0"
    expected_output = "11\n21\n0\n21\n12"
    run_pie_test_case("../p00169.py", input_content, expected_output)
