from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03785_0():
    input_content = "5 3 5\n1\n2\n3\n6\n12"
    expected_output = "3"
    run_pie_test_case("../p03785.py", input_content, expected_output)


def test_problem_p03785_1():
    input_content = "5 3 5\n1\n2\n3\n6\n12"
    expected_output = "3"
    run_pie_test_case("../p03785.py", input_content, expected_output)


def test_problem_p03785_2():
    input_content = "6 3 3\n7\n6\n2\n8\n10\n6"
    expected_output = "3"
    run_pie_test_case("../p03785.py", input_content, expected_output)
