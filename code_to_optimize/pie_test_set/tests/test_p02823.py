from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02823_0():
    input_content = "5 2 4"
    expected_output = "1"
    run_pie_test_case("../p02823.py", input_content, expected_output)


def test_problem_p02823_1():
    input_content = "5 2 3"
    expected_output = "2"
    run_pie_test_case("../p02823.py", input_content, expected_output)


def test_problem_p02823_2():
    input_content = "5 2 4"
    expected_output = "1"
    run_pie_test_case("../p02823.py", input_content, expected_output)
