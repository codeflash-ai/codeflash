from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03845_0():
    input_content = "3\n2 1 4\n2\n1 1\n2 3"
    expected_output = "6\n9"
    run_pie_test_case("../p03845.py", input_content, expected_output)


def test_problem_p03845_1():
    input_content = "3\n2 1 4\n2\n1 1\n2 3"
    expected_output = "6\n9"
    run_pie_test_case("../p03845.py", input_content, expected_output)


def test_problem_p03845_2():
    input_content = "5\n7 2 3 8 5\n3\n4 2\n1 7\n4 13"
    expected_output = "19\n25\n30"
    run_pie_test_case("../p03845.py", input_content, expected_output)
