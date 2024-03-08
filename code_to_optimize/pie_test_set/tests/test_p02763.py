from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02763_0():
    input_content = "7\nabcdbbd\n6\n2 3 6\n1 5 z\n2 1 1\n1 4 a\n1 7 d\n2 1 7"
    expected_output = "3\n1\n5"
    run_pie_test_case("../p02763.py", input_content, expected_output)


def test_problem_p02763_1():
    input_content = "7\nabcdbbd\n6\n2 3 6\n1 5 z\n2 1 1\n1 4 a\n1 7 d\n2 1 7"
    expected_output = "3\n1\n5"
    run_pie_test_case("../p02763.py", input_content, expected_output)
