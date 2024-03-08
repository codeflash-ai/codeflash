from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02748_0():
    input_content = "2 3 1\n3 3\n3 3 3\n1 2 1"
    expected_output = "5"
    run_pie_test_case("../p02748.py", input_content, expected_output)


def test_problem_p02748_1():
    input_content = "1 1 2\n10\n10\n1 1 5\n1 1 10"
    expected_output = "10"
    run_pie_test_case("../p02748.py", input_content, expected_output)


def test_problem_p02748_2():
    input_content = "2 2 1\n3 5\n3 5\n2 2 2"
    expected_output = "6"
    run_pie_test_case("../p02748.py", input_content, expected_output)


def test_problem_p02748_3():
    input_content = "2 3 1\n3 3\n3 3 3\n1 2 1"
    expected_output = "5"
    run_pie_test_case("../p02748.py", input_content, expected_output)
