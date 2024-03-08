from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02743_0():
    input_content = "2 3 9"
    expected_output = "No"
    run_pie_test_case("../p02743.py", input_content, expected_output)


def test_problem_p02743_1():
    input_content = "2 3 10"
    expected_output = "Yes"
    run_pie_test_case("../p02743.py", input_content, expected_output)


def test_problem_p02743_2():
    input_content = "2 3 9"
    expected_output = "No"
    run_pie_test_case("../p02743.py", input_content, expected_output)
