from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02782_0():
    input_content = "1 1 2 2"
    expected_output = "14"
    run_pie_test_case("../p02782.py", input_content, expected_output)


def test_problem_p02782_1():
    input_content = "314 159 2653 589"
    expected_output = "602215194"
    run_pie_test_case("../p02782.py", input_content, expected_output)


def test_problem_p02782_2():
    input_content = "1 1 2 2"
    expected_output = "14"
    run_pie_test_case("../p02782.py", input_content, expected_output)
