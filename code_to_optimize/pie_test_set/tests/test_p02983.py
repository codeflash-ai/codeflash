from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02983_0():
    input_content = "2020 2040"
    expected_output = "2"
    run_pie_test_case("../p02983.py", input_content, expected_output)


def test_problem_p02983_1():
    input_content = "4 5"
    expected_output = "20"
    run_pie_test_case("../p02983.py", input_content, expected_output)


def test_problem_p02983_2():
    input_content = "2020 2040"
    expected_output = "2"
    run_pie_test_case("../p02983.py", input_content, expected_output)
