from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02303_0():
    input_content = "2\n0.0 0.0\n1.0 0.0"
    expected_output = "1.000000"
    run_pie_test_case("../p02303.py", input_content, expected_output)


def test_problem_p02303_1():
    input_content = "3\n0.0 0.0\n2.0 0.0\n1.0 1.0"
    expected_output = "1.41421356237"
    run_pie_test_case("../p02303.py", input_content, expected_output)


def test_problem_p02303_2():
    input_content = "2\n0.0 0.0\n1.0 0.0"
    expected_output = "1.000000"
    run_pie_test_case("../p02303.py", input_content, expected_output)
