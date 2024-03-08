from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02881_0():
    input_content = "10"
    expected_output = "5"
    run_pie_test_case("../p02881.py", input_content, expected_output)


def test_problem_p02881_1():
    input_content = "10000000019"
    expected_output = "10000000018"
    run_pie_test_case("../p02881.py", input_content, expected_output)


def test_problem_p02881_2():
    input_content = "50"
    expected_output = "13"
    run_pie_test_case("../p02881.py", input_content, expected_output)


def test_problem_p02881_3():
    input_content = "10"
    expected_output = "5"
    run_pie_test_case("../p02881.py", input_content, expected_output)
