from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03227_0():
    input_content = "abc"
    expected_output = "cba"
    run_pie_test_case("../p03227.py", input_content, expected_output)


def test_problem_p03227_1():
    input_content = "abc"
    expected_output = "cba"
    run_pie_test_case("../p03227.py", input_content, expected_output)


def test_problem_p03227_2():
    input_content = "ac"
    expected_output = "ac"
    run_pie_test_case("../p03227.py", input_content, expected_output)
