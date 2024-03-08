from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02384_0():
    input_content = "1 2 3 4 5 6\n3\n6 5\n1 3\n3 2"
    expected_output = "3\n5\n6"
    run_pie_test_case("../p02384.py", input_content, expected_output)


def test_problem_p02384_1():
    input_content = "1 2 3 4 5 6\n3\n6 5\n1 3\n3 2"
    expected_output = "3\n5\n6"
    run_pie_test_case("../p02384.py", input_content, expected_output)
