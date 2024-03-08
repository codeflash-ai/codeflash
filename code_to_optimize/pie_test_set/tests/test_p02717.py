from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02717_0():
    input_content = "1 2 3"
    expected_output = "3 1 2"
    run_pie_test_case("../p02717.py", input_content, expected_output)


def test_problem_p02717_1():
    input_content = "100 100 100"
    expected_output = "100 100 100"
    run_pie_test_case("../p02717.py", input_content, expected_output)


def test_problem_p02717_2():
    input_content = "41 59 31"
    expected_output = "31 41 59"
    run_pie_test_case("../p02717.py", input_content, expected_output)


def test_problem_p02717_3():
    input_content = "1 2 3"
    expected_output = "3 1 2"
    run_pie_test_case("../p02717.py", input_content, expected_output)
