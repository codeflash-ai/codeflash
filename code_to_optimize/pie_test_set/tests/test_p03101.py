from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03101_0():
    input_content = "3 2\n2 1"
    expected_output = "1"
    run_pie_test_case("../p03101.py", input_content, expected_output)


def test_problem_p03101_1():
    input_content = "2 4\n2 4"
    expected_output = "0"
    run_pie_test_case("../p03101.py", input_content, expected_output)


def test_problem_p03101_2():
    input_content = "5 5\n2 3"
    expected_output = "6"
    run_pie_test_case("../p03101.py", input_content, expected_output)


def test_problem_p03101_3():
    input_content = "3 2\n2 1"
    expected_output = "1"
    run_pie_test_case("../p03101.py", input_content, expected_output)
