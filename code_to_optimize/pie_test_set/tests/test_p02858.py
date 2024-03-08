from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02858_0():
    input_content = "2 2 1"
    expected_output = "9"
    run_pie_test_case("../p02858.py", input_content, expected_output)


def test_problem_p02858_1():
    input_content = "869 120 1001"
    expected_output = "672919729"
    run_pie_test_case("../p02858.py", input_content, expected_output)


def test_problem_p02858_2():
    input_content = "2 2 1"
    expected_output = "9"
    run_pie_test_case("../p02858.py", input_content, expected_output)
