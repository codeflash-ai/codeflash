from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02992_0():
    input_content = "3 2"
    expected_output = "5"
    run_pie_test_case("../p02992.py", input_content, expected_output)


def test_problem_p02992_1():
    input_content = "10 3"
    expected_output = "147"
    run_pie_test_case("../p02992.py", input_content, expected_output)


def test_problem_p02992_2():
    input_content = "314159265 35"
    expected_output = "457397712"
    run_pie_test_case("../p02992.py", input_content, expected_output)


def test_problem_p02992_3():
    input_content = "3 2"
    expected_output = "5"
    run_pie_test_case("../p02992.py", input_content, expected_output)
