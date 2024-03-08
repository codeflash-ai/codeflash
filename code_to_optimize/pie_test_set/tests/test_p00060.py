from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00060_0():
    input_content = "1 2 3\n5 6 9\n8 9 10"
    expected_output = "YES\nYES\nNO"
    run_pie_test_case("../p00060.py", input_content, expected_output)


def test_problem_p00060_1():
    input_content = "1 2 3\n5 6 9\n8 9 10"
    expected_output = "YES\nYES\nNO"
    run_pie_test_case("../p00060.py", input_content, expected_output)
