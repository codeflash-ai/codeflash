from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00470_0():
    input_content = "3 4\n15 15\n0 0"
    expected_output = "5\n43688"
    run_pie_test_case("../p00470.py", input_content, expected_output)


def test_problem_p00470_1():
    input_content = "3 4\n15 15\n0 0"
    expected_output = "5\n43688"
    run_pie_test_case("../p00470.py", input_content, expected_output)
