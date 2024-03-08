from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02865_0():
    input_content = "4"
    expected_output = "1"
    run_pie_test_case("../p02865.py", input_content, expected_output)


def test_problem_p02865_1():
    input_content = "4"
    expected_output = "1"
    run_pie_test_case("../p02865.py", input_content, expected_output)


def test_problem_p02865_2():
    input_content = "999999"
    expected_output = "499999"
    run_pie_test_case("../p02865.py", input_content, expected_output)
