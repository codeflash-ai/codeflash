from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03312_0():
    input_content = "5\n3 2 4 1 2"
    expected_output = "2"
    run_pie_test_case("../p03312.py", input_content, expected_output)


def test_problem_p03312_1():
    input_content = "10\n10 71 84 33 6 47 23 25 52 64"
    expected_output = "36"
    run_pie_test_case("../p03312.py", input_content, expected_output)


def test_problem_p03312_2():
    input_content = "7\n1 2 3 1000000000 4 5 6"
    expected_output = "999999994"
    run_pie_test_case("../p03312.py", input_content, expected_output)


def test_problem_p03312_3():
    input_content = "5\n3 2 4 1 2"
    expected_output = "2"
    run_pie_test_case("../p03312.py", input_content, expected_output)
