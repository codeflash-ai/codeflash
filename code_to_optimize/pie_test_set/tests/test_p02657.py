from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02657_0():
    input_content = "2 5"
    expected_output = "10"
    run_pie_test_case("../p02657.py", input_content, expected_output)


def test_problem_p02657_1():
    input_content = "2 5"
    expected_output = "10"
    run_pie_test_case("../p02657.py", input_content, expected_output)


def test_problem_p02657_2():
    input_content = "100 100"
    expected_output = "10000"
    run_pie_test_case("../p02657.py", input_content, expected_output)
