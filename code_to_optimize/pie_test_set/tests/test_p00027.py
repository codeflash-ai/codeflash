from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00027_0():
    input_content = "1 1\n2 29\n0 0"
    expected_output = "Thursday\nSunday"
    run_pie_test_case("../p00027.py", input_content, expected_output)


def test_problem_p00027_1():
    input_content = "1 1\n2 29\n0 0"
    expected_output = "Thursday\nSunday"
    run_pie_test_case("../p00027.py", input_content, expected_output)
