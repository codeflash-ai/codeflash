from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02700_0():
    input_content = "10 9 10 10"
    expected_output = "No"
    run_pie_test_case("../p02700.py", input_content, expected_output)


def test_problem_p02700_1():
    input_content = "46 4 40 5"
    expected_output = "Yes"
    run_pie_test_case("../p02700.py", input_content, expected_output)


def test_problem_p02700_2():
    input_content = "10 9 10 10"
    expected_output = "No"
    run_pie_test_case("../p02700.py", input_content, expected_output)
