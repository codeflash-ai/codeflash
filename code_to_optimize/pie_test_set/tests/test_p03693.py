from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03693_0():
    input_content = "4 3 2"
    expected_output = "YES"
    run_pie_test_case("../p03693.py", input_content, expected_output)


def test_problem_p03693_1():
    input_content = "4 3 2"
    expected_output = "YES"
    run_pie_test_case("../p03693.py", input_content, expected_output)


def test_problem_p03693_2():
    input_content = "2 3 4"
    expected_output = "NO"
    run_pie_test_case("../p03693.py", input_content, expected_output)
