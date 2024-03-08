from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02948_0():
    input_content = "3 4\n4 3\n4 1\n2 2"
    expected_output = "5"
    run_pie_test_case("../p02948.py", input_content, expected_output)


def test_problem_p02948_1():
    input_content = "5 3\n1 2\n1 3\n1 4\n2 1\n2 3"
    expected_output = "10"
    run_pie_test_case("../p02948.py", input_content, expected_output)


def test_problem_p02948_2():
    input_content = "1 1\n2 1"
    expected_output = "0"
    run_pie_test_case("../p02948.py", input_content, expected_output)


def test_problem_p02948_3():
    input_content = "3 4\n4 3\n4 1\n2 2"
    expected_output = "5"
    run_pie_test_case("../p02948.py", input_content, expected_output)
