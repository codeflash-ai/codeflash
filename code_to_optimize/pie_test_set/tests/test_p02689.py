from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02689_0():
    input_content = "4 3\n1 2 3 4\n1 3\n2 3\n2 4"
    expected_output = "2"
    run_pie_test_case("../p02689.py", input_content, expected_output)


def test_problem_p02689_1():
    input_content = "6 5\n8 6 9 1 2 1\n1 3\n4 2\n4 3\n4 6\n4 6"
    expected_output = "3"
    run_pie_test_case("../p02689.py", input_content, expected_output)


def test_problem_p02689_2():
    input_content = "4 3\n1 2 3 4\n1 3\n2 3\n2 4"
    expected_output = "2"
    run_pie_test_case("../p02689.py", input_content, expected_output)
