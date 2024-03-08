from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02594_0():
    input_content = "25"
    expected_output = "No"
    run_pie_test_case("../p02594.py", input_content, expected_output)


def test_problem_p02594_1():
    input_content = "25"
    expected_output = "No"
    run_pie_test_case("../p02594.py", input_content, expected_output)


def test_problem_p02594_2():
    input_content = "30"
    expected_output = "Yes"
    run_pie_test_case("../p02594.py", input_content, expected_output)
