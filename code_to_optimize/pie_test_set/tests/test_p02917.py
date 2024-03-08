from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02917_0():
    input_content = "3\n2 5"
    expected_output = "9"
    run_pie_test_case("../p02917.py", input_content, expected_output)


def test_problem_p02917_1():
    input_content = "2\n3"
    expected_output = "6"
    run_pie_test_case("../p02917.py", input_content, expected_output)


def test_problem_p02917_2():
    input_content = "6\n0 153 10 10 23"
    expected_output = "53"
    run_pie_test_case("../p02917.py", input_content, expected_output)


def test_problem_p02917_3():
    input_content = "3\n2 5"
    expected_output = "9"
    run_pie_test_case("../p02917.py", input_content, expected_output)
