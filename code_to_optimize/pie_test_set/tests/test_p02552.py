from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02552_0():
    input_content = "1"
    expected_output = "0"
    run_pie_test_case("../p02552.py", input_content, expected_output)


def test_problem_p02552_1():
    input_content = "1"
    expected_output = "0"
    run_pie_test_case("../p02552.py", input_content, expected_output)


def test_problem_p02552_2():
    input_content = "0"
    expected_output = "1"
    run_pie_test_case("../p02552.py", input_content, expected_output)
