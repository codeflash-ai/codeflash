from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03641_0():
    input_content = "4\n3 2 4 1"
    expected_output = "3 1 2 4"
    run_pie_test_case("../p03641.py", input_content, expected_output)


def test_problem_p03641_1():
    input_content = "2\n1 2"
    expected_output = "1 2"
    run_pie_test_case("../p03641.py", input_content, expected_output)


def test_problem_p03641_2():
    input_content = "8\n4 6 3 2 8 5 7 1"
    expected_output = "3 1 2 7 4 6 8 5"
    run_pie_test_case("../p03641.py", input_content, expected_output)


def test_problem_p03641_3():
    input_content = "4\n3 2 4 1"
    expected_output = "3 1 2 4"
    run_pie_test_case("../p03641.py", input_content, expected_output)
