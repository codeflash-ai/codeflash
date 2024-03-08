from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02875_0():
    input_content = "2"
    expected_output = "7"
    run_pie_test_case("../p02875.py", input_content, expected_output)


def test_problem_p02875_1():
    input_content = "2"
    expected_output = "7"
    run_pie_test_case("../p02875.py", input_content, expected_output)


def test_problem_p02875_2():
    input_content = "1000000"
    expected_output = "210055358"
    run_pie_test_case("../p02875.py", input_content, expected_output)


def test_problem_p02875_3():
    input_content = "10"
    expected_output = "50007"
    run_pie_test_case("../p02875.py", input_content, expected_output)
