from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03879_0():
    input_content = "0 0\n1 1\n2 0"
    expected_output = "0.292893218813"
    run_pie_test_case("../p03879.py", input_content, expected_output)


def test_problem_p03879_1():
    input_content = "3 1\n1 5\n4 9"
    expected_output = "0.889055514217"
    run_pie_test_case("../p03879.py", input_content, expected_output)


def test_problem_p03879_2():
    input_content = "0 0\n1 1\n2 0"
    expected_output = "0.292893218813"
    run_pie_test_case("../p03879.py", input_content, expected_output)
