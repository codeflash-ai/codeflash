from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02304_0():
    input_content = "6\n2 2 2 5\n1 3 5 3\n4 1 4 4\n5 2 7 2\n6 1 6 3\n6 5 6 7"
    expected_output = "3"
    run_pie_test_case("../p02304.py", input_content, expected_output)


def test_problem_p02304_1():
    input_content = "6\n2 2 2 5\n1 3 5 3\n4 1 4 4\n5 2 7 2\n6 1 6 3\n6 5 6 7"
    expected_output = "3"
    run_pie_test_case("../p02304.py", input_content, expected_output)
