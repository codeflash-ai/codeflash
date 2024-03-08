from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04043_0():
    input_content = "5 5 7"
    expected_output = "YES"
    run_pie_test_case("../p04043.py", input_content, expected_output)


def test_problem_p04043_1():
    input_content = "5 5 7"
    expected_output = "YES"
    run_pie_test_case("../p04043.py", input_content, expected_output)


def test_problem_p04043_2():
    input_content = "7 7 5"
    expected_output = "NO"
    run_pie_test_case("../p04043.py", input_content, expected_output)
