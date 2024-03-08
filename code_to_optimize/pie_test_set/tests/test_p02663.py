from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02663_0():
    input_content = "10 0 15 0 30"
    expected_output = "270"
    run_pie_test_case("../p02663.py", input_content, expected_output)


def test_problem_p02663_1():
    input_content = "10 0 15 0 30"
    expected_output = "270"
    run_pie_test_case("../p02663.py", input_content, expected_output)


def test_problem_p02663_2():
    input_content = "10 0 12 0 120"
    expected_output = "0"
    run_pie_test_case("../p02663.py", input_content, expected_output)
