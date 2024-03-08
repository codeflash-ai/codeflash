from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02768_0():
    input_content = "4 1 3"
    expected_output = "7"
    run_pie_test_case("../p02768.py", input_content, expected_output)


def test_problem_p02768_1():
    input_content = "4 1 3"
    expected_output = "7"
    run_pie_test_case("../p02768.py", input_content, expected_output)


def test_problem_p02768_2():
    input_content = "1000000000 141421 173205"
    expected_output = "34076506"
    run_pie_test_case("../p02768.py", input_content, expected_output)
