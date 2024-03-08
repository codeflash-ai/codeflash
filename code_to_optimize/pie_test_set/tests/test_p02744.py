from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02744_0():
    input_content = "1"
    expected_output = "a"
    run_pie_test_case("../p02744.py", input_content, expected_output)


def test_problem_p02744_1():
    input_content = "1"
    expected_output = "a"
    run_pie_test_case("../p02744.py", input_content, expected_output)


def test_problem_p02744_2():
    input_content = "2"
    expected_output = "aa\nab"
    run_pie_test_case("../p02744.py", input_content, expected_output)
