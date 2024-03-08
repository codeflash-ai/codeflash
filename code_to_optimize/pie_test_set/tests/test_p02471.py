from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02471_0():
    input_content = "4 12"
    expected_output = "1 0"
    run_pie_test_case("../p02471.py", input_content, expected_output)


def test_problem_p02471_1():
    input_content = "3 8"
    expected_output = "3 -1"
    run_pie_test_case("../p02471.py", input_content, expected_output)


def test_problem_p02471_2():
    input_content = "4 12"
    expected_output = "1 0"
    run_pie_test_case("../p02471.py", input_content, expected_output)
