from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02330_0():
    input_content = "2 2 1 9\n5 1"
    expected_output = "1"
    run_pie_test_case("../p02330.py", input_content, expected_output)


def test_problem_p02330_1():
    input_content = "5 2 7 19\n3 5 4 2 2"
    expected_output = "5"
    run_pie_test_case("../p02330.py", input_content, expected_output)


def test_problem_p02330_2():
    input_content = "2 2 1 9\n5 1"
    expected_output = "1"
    run_pie_test_case("../p02330.py", input_content, expected_output)
