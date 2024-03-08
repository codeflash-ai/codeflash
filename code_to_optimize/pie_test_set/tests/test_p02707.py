from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02707_0():
    input_content = "5\n1 1 2 2"
    expected_output = "2\n2\n0\n0\n0"
    run_pie_test_case("../p02707.py", input_content, expected_output)


def test_problem_p02707_1():
    input_content = "7\n1 2 3 4 5 6"
    expected_output = "1\n1\n1\n1\n1\n1\n0"
    run_pie_test_case("../p02707.py", input_content, expected_output)


def test_problem_p02707_2():
    input_content = "5\n1 1 2 2"
    expected_output = "2\n2\n0\n0\n0"
    run_pie_test_case("../p02707.py", input_content, expected_output)


def test_problem_p02707_3():
    input_content = "10\n1 1 1 1 1 1 1 1 1"
    expected_output = "9\n0\n0\n0\n0\n0\n0\n0\n0\n0"
    run_pie_test_case("../p02707.py", input_content, expected_output)
