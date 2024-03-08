from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02817_0():
    input_content = "oder atc"
    expected_output = "atcoder"
    run_pie_test_case("../p02817.py", input_content, expected_output)


def test_problem_p02817_1():
    input_content = "humu humu"
    expected_output = "humuhumu"
    run_pie_test_case("../p02817.py", input_content, expected_output)


def test_problem_p02817_2():
    input_content = "oder atc"
    expected_output = "atcoder"
    run_pie_test_case("../p02817.py", input_content, expected_output)
