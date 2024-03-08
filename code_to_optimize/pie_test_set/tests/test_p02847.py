from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02847_0():
    input_content = "SAT"
    expected_output = "1"
    run_pie_test_case("../p02847.py", input_content, expected_output)


def test_problem_p02847_1():
    input_content = "SUN"
    expected_output = "7"
    run_pie_test_case("../p02847.py", input_content, expected_output)


def test_problem_p02847_2():
    input_content = "SAT"
    expected_output = "1"
    run_pie_test_case("../p02847.py", input_content, expected_output)
