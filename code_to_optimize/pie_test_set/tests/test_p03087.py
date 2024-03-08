from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03087_0():
    input_content = "8 3\nACACTACG\n3 7\n2 3\n1 8"
    expected_output = "2\n0\n3"
    run_pie_test_case("../p03087.py", input_content, expected_output)


def test_problem_p03087_1():
    input_content = "8 3\nACACTACG\n3 7\n2 3\n1 8"
    expected_output = "2\n0\n3"
    run_pie_test_case("../p03087.py", input_content, expected_output)
