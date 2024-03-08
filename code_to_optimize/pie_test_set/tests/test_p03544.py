from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03544_0():
    input_content = "5"
    expected_output = "11"
    run_pie_test_case("../p03544.py", input_content, expected_output)


def test_problem_p03544_1():
    input_content = "5"
    expected_output = "11"
    run_pie_test_case("../p03544.py", input_content, expected_output)


def test_problem_p03544_2():
    input_content = "86"
    expected_output = "939587134549734843"
    run_pie_test_case("../p03544.py", input_content, expected_output)
