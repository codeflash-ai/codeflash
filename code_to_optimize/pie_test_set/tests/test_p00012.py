from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00012_0():
    input_content = "0.0 0.0 2.0 0.0 2.0 2.0 1.5 0.5\n0.0 0.0 1.0 4.0 5.0 3.0 -1.0 3.0"
    expected_output = "YES\nNO"
    run_pie_test_case("../p00012.py", input_content, expected_output)


def test_problem_p00012_1():
    input_content = "0.0 0.0 2.0 0.0 2.0 2.0 1.5 0.5\n0.0 0.0 1.0 4.0 5.0 3.0 -1.0 3.0"
    expected_output = "YES\nNO"
    run_pie_test_case("../p00012.py", input_content, expected_output)
