from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03333_0():
    input_content = "3\n-5 1\n3 7\n-4 -2"
    expected_output = "10"
    run_pie_test_case("../p03333.py", input_content, expected_output)


def test_problem_p03333_1():
    input_content = "3\n1 2\n3 4\n5 6"
    expected_output = "12"
    run_pie_test_case("../p03333.py", input_content, expected_output)


def test_problem_p03333_2():
    input_content = "3\n-5 1\n3 7\n-4 -2"
    expected_output = "10"
    run_pie_test_case("../p03333.py", input_content, expected_output)


def test_problem_p03333_3():
    input_content = "5\n-2 0\n-2 0\n7 8\n9 10\n-2 -1"
    expected_output = "34"
    run_pie_test_case("../p03333.py", input_content, expected_output)
