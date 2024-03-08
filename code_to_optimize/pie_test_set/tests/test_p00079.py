from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00079_0():
    input_content = "0.0,0.0\n0.0,1.0\n1.0,1.0\n2.0,0.0\n1.0,-1.0"
    expected_output = "2.500000"
    run_pie_test_case("../p00079.py", input_content, expected_output)


def test_problem_p00079_1():
    input_content = "0.0,0.0\n0.0,1.0\n1.0,1.0\n2.0,0.0\n1.0,-1.0"
    expected_output = "2.500000"
    run_pie_test_case("../p00079.py", input_content, expected_output)
