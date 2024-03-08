from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02259_0():
    input_content = "5\n5 3 2 4 1"
    expected_output = "1 2 3 4 5\n8"
    run_pie_test_case("../p02259.py", input_content, expected_output)


def test_problem_p02259_1():
    input_content = "5\n5 3 2 4 1"
    expected_output = "1 2 3 4 5\n8"
    run_pie_test_case("../p02259.py", input_content, expected_output)


def test_problem_p02259_2():
    input_content = "6\n5 2 4 6 1 3"
    expected_output = "1 2 3 4 5 6\n9"
    run_pie_test_case("../p02259.py", input_content, expected_output)
