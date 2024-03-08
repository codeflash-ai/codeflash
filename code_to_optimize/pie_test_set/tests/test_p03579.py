from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03579_0():
    input_content = "6 5\n1 2\n2 3\n3 4\n4 5\n5 6"
    expected_output = "4"
    run_pie_test_case("../p03579.py", input_content, expected_output)


def test_problem_p03579_1():
    input_content = "6 5\n1 2\n2 3\n3 4\n4 5\n5 6"
    expected_output = "4"
    run_pie_test_case("../p03579.py", input_content, expected_output)


def test_problem_p03579_2():
    input_content = "5 5\n1 2\n2 3\n3 1\n5 4\n5 1"
    expected_output = "5"
    run_pie_test_case("../p03579.py", input_content, expected_output)
