from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03211_0():
    input_content = "1234567876"
    expected_output = "34"
    run_pie_test_case("../p03211.py", input_content, expected_output)


def test_problem_p03211_1():
    input_content = "35753"
    expected_output = "0"
    run_pie_test_case("../p03211.py", input_content, expected_output)


def test_problem_p03211_2():
    input_content = "1234567876"
    expected_output = "34"
    run_pie_test_case("../p03211.py", input_content, expected_output)


def test_problem_p03211_3():
    input_content = "1111111111"
    expected_output = "642"
    run_pie_test_case("../p03211.py", input_content, expected_output)
