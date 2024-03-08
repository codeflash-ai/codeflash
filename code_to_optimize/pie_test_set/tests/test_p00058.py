from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00058_0():
    input_content = "1.0 1.0 2.0 2.0 0.0 0.0 1.0 -1.0\n0.0 0.0 2.0 0.0 -1.0 2.0 2.0 2.0\n10.0 6.0 3.4 5.2 6.8 9.5 4.3 2.1\n2.5 3.5 2.5 4.5 -3.3 -2.3 6.8 -2.3"
    expected_output = "YES\nNO\nNO\nYES"
    run_pie_test_case("../p00058.py", input_content, expected_output)


def test_problem_p00058_1():
    input_content = "1.0 1.0 2.0 2.0 0.0 0.0 1.0 -1.0\n0.0 0.0 2.0 0.0 -1.0 2.0 2.0 2.0\n10.0 6.0 3.4 5.2 6.8 9.5 4.3 2.1\n2.5 3.5 2.5 4.5 -3.3 -2.3 6.8 -2.3"
    expected_output = "YES\nNO\nNO\nYES"
    run_pie_test_case("../p00058.py", input_content, expected_output)
