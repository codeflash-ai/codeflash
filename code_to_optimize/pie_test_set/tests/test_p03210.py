from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03210_0():
    input_content = "5"
    expected_output = "YES"
    run_pie_test_case("../p03210.py", input_content, expected_output)


def test_problem_p03210_1():
    input_content = "6"
    expected_output = "NO"
    run_pie_test_case("../p03210.py", input_content, expected_output)


def test_problem_p03210_2():
    input_content = "5"
    expected_output = "YES"
    run_pie_test_case("../p03210.py", input_content, expected_output)
