from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02745_0():
    input_content = "a?c\nder\ncod"
    expected_output = "7"
    run_pie_test_case("../p02745.py", input_content, expected_output)


def test_problem_p02745_1():
    input_content = "a?c\nder\ncod"
    expected_output = "7"
    run_pie_test_case("../p02745.py", input_content, expected_output)


def test_problem_p02745_2():
    input_content = "atcoder\natcoder\n???????"
    expected_output = "7"
    run_pie_test_case("../p02745.py", input_content, expected_output)
