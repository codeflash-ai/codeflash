from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02416_0():
    input_content = "123\n55\n1000\n0"
    expected_output = "6\n10\n1"
    run_pie_test_case("../p02416.py", input_content, expected_output)


def test_problem_p02416_1():
    input_content = "123\n55\n1000\n0"
    expected_output = "6\n10\n1"
    run_pie_test_case("../p02416.py", input_content, expected_output)
