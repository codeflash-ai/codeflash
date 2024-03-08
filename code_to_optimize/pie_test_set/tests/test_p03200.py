from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03200_0():
    input_content = "BBW"
    expected_output = "2"
    run_pie_test_case("../p03200.py", input_content, expected_output)


def test_problem_p03200_1():
    input_content = "BWBWBW"
    expected_output = "6"
    run_pie_test_case("../p03200.py", input_content, expected_output)


def test_problem_p03200_2():
    input_content = "BBW"
    expected_output = "2"
    run_pie_test_case("../p03200.py", input_content, expected_output)
