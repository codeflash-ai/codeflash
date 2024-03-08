from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01741_0():
    input_content = "1.000"
    expected_output = "2.000000000000"
    run_pie_test_case("../p01741.py", input_content, expected_output)


def test_problem_p01741_1():
    input_content = "1.000"
    expected_output = "2.000000000000"
    run_pie_test_case("../p01741.py", input_content, expected_output)


def test_problem_p01741_2():
    input_content = "2.345"
    expected_output = "3.316330803765"
    run_pie_test_case("../p01741.py", input_content, expected_output)
