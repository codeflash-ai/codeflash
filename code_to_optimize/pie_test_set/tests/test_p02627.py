from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02627_0():
    input_content = "B"
    expected_output = "A"
    run_pie_test_case("../p02627.py", input_content, expected_output)


def test_problem_p02627_1():
    input_content = "a"
    expected_output = "a"
    run_pie_test_case("../p02627.py", input_content, expected_output)


def test_problem_p02627_2():
    input_content = "B"
    expected_output = "A"
    run_pie_test_case("../p02627.py", input_content, expected_output)
