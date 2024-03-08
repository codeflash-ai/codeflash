from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00253_0():
    input_content = "5\n1 2 3 6 4 5\n6\n1 3 6 9 12 15 18\n4\n5 7 9 11 12\n0"
    expected_output = "6\n1\n12"
    run_pie_test_case("../p00253.py", input_content, expected_output)


def test_problem_p00253_1():
    input_content = "5\n1 2 3 6 4 5\n6\n1 3 6 9 12 15 18\n4\n5 7 9 11 12\n0"
    expected_output = "6\n1\n12"
    run_pie_test_case("../p00253.py", input_content, expected_output)
