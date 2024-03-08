from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02835_0():
    input_content = "5 7 9"
    expected_output = "win"
    run_pie_test_case("../p02835.py", input_content, expected_output)


def test_problem_p02835_1():
    input_content = "5 7 9"
    expected_output = "win"
    run_pie_test_case("../p02835.py", input_content, expected_output)


def test_problem_p02835_2():
    input_content = "13 7 2"
    expected_output = "bust"
    run_pie_test_case("../p02835.py", input_content, expected_output)
