from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02649_0():
    input_content = "3 3 0 3\n1 2 3"
    expected_output = "2"
    run_pie_test_case("../p02649.py", input_content, expected_output)


def test_problem_p02649_1():
    input_content = "3 3 0 3\n1 2 3"
    expected_output = "2"
    run_pie_test_case("../p02649.py", input_content, expected_output)


def test_problem_p02649_2():
    input_content = "5 3 1 7\n3 4 9 1 5"
    expected_output = "2"
    run_pie_test_case("../p02649.py", input_content, expected_output)


def test_problem_p02649_3():
    input_content = "5 4 0 15\n3 4 9 1 5"
    expected_output = "3"
    run_pie_test_case("../p02649.py", input_content, expected_output)
