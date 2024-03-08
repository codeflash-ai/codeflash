from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02392_0():
    input_content = "1 3 8"
    expected_output = "Yes"
    run_pie_test_case("../p02392.py", input_content, expected_output)


def test_problem_p02392_1():
    input_content = "1 3 8"
    expected_output = "Yes"
    run_pie_test_case("../p02392.py", input_content, expected_output)


def test_problem_p02392_2():
    input_content = "3 8 1"
    expected_output = "No"
    run_pie_test_case("../p02392.py", input_content, expected_output)
