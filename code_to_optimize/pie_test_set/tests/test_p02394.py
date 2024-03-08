from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02394_0():
    input_content = "5 4 2 2 1"
    expected_output = "Yes"
    run_pie_test_case("../p02394.py", input_content, expected_output)


def test_problem_p02394_1():
    input_content = "5 4 2 2 1"
    expected_output = "Yes"
    run_pie_test_case("../p02394.py", input_content, expected_output)


def test_problem_p02394_2():
    input_content = "5 4 2 4 1"
    expected_output = "No"
    run_pie_test_case("../p02394.py", input_content, expected_output)
