from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02737_0():
    input_content = "2\n1\n1 2\n1\n1 2\n1\n1 2"
    expected_output = "46494701"
    run_pie_test_case("../p02737.py", input_content, expected_output)


def test_problem_p02737_1():
    input_content = "3\n3\n1 3\n1 2\n3 2\n2\n2 1\n2 3\n1\n2 1"
    expected_output = "883188316"
    run_pie_test_case("../p02737.py", input_content, expected_output)


def test_problem_p02737_2():
    input_content = "100000\n1\n1 2\n1\n99999 100000\n1\n1 100000"
    expected_output = "318525248"
    run_pie_test_case("../p02737.py", input_content, expected_output)


def test_problem_p02737_3():
    input_content = "2\n1\n1 2\n1\n1 2\n1\n1 2"
    expected_output = "46494701"
    run_pie_test_case("../p02737.py", input_content, expected_output)
