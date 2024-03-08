from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02641_0():
    input_content = "6 5\n4 7 10 6 5"
    expected_output = "8"
    run_pie_test_case("../p02641.py", input_content, expected_output)


def test_problem_p02641_1():
    input_content = "6 5\n4 7 10 6 5"
    expected_output = "8"
    run_pie_test_case("../p02641.py", input_content, expected_output)


def test_problem_p02641_2():
    input_content = "10 5\n4 7 10 6 5"
    expected_output = "9"
    run_pie_test_case("../p02641.py", input_content, expected_output)


def test_problem_p02641_3():
    input_content = "100 0"
    expected_output = "100"
    run_pie_test_case("../p02641.py", input_content, expected_output)
