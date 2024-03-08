from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00003_0():
    input_content = "3\n4 3 5\n4 3 6\n8 8 8"
    expected_output = "YES\nNO\nNO"
    run_pie_test_case("../p00003.py", input_content, expected_output)


def test_problem_p00003_1():
    input_content = "3\n4 3 5\n4 3 6\n8 8 8"
    expected_output = "YES\nNO\nNO"
    run_pie_test_case("../p00003.py", input_content, expected_output)
