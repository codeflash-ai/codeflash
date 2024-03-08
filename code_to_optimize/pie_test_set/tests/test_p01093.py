from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01093_0():
    input_content = "5\n10 10 10 10 10\n5\n1 5 8 9 11\n7\n11 34 83 47 59 29 70\n0"
    expected_output = "0\n1\n5"
    run_pie_test_case("../p01093.py", input_content, expected_output)


def test_problem_p01093_1():
    input_content = "5\n10 10 10 10 10\n5\n1 5 8 9 11\n7\n11 34 83 47 59 29 70\n0"
    expected_output = "0\n1\n5"
    run_pie_test_case("../p01093.py", input_content, expected_output)
