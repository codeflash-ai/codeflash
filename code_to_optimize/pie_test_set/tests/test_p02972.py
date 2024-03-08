from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02972_0():
    input_content = "3\n1 0 0"
    expected_output = "1\n1"
    run_pie_test_case("../p02972.py", input_content, expected_output)


def test_problem_p02972_1():
    input_content = "5\n0 0 0 0 0"
    expected_output = "0"
    run_pie_test_case("../p02972.py", input_content, expected_output)


def test_problem_p02972_2():
    input_content = "3\n1 0 0"
    expected_output = "1\n1"
    run_pie_test_case("../p02972.py", input_content, expected_output)
