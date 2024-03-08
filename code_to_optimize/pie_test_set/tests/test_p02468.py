from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02468_0():
    input_content = "2 3"
    expected_output = "8"
    run_pie_test_case("../p02468.py", input_content, expected_output)


def test_problem_p02468_1():
    input_content = "2 3"
    expected_output = "8"
    run_pie_test_case("../p02468.py", input_content, expected_output)


def test_problem_p02468_2():
    input_content = "5 8"
    expected_output = "390625"
    run_pie_test_case("../p02468.py", input_content, expected_output)
