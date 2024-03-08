from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02268_0():
    input_content = "5\n1 2 3 4 5\n3\n3 4 1"
    expected_output = "3"
    run_pie_test_case("../p02268.py", input_content, expected_output)


def test_problem_p02268_1():
    input_content = "5\n1 1 2 2 3\n2\n1 2"
    expected_output = "2"
    run_pie_test_case("../p02268.py", input_content, expected_output)


def test_problem_p02268_2():
    input_content = "3\n1 2 3\n1\n5"
    expected_output = "0"
    run_pie_test_case("../p02268.py", input_content, expected_output)


def test_problem_p02268_3():
    input_content = "5\n1 2 3 4 5\n3\n3 4 1"
    expected_output = "3"
    run_pie_test_case("../p02268.py", input_content, expected_output)
