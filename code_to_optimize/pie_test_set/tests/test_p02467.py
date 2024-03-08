from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02467_0():
    input_content = "12"
    expected_output = "12: 2 2 3"
    run_pie_test_case("../p02467.py", input_content, expected_output)


def test_problem_p02467_1():
    input_content = "12"
    expected_output = "12: 2 2 3"
    run_pie_test_case("../p02467.py", input_content, expected_output)


def test_problem_p02467_2():
    input_content = "126"
    expected_output = "126: 2 3 3 7"
    run_pie_test_case("../p02467.py", input_content, expected_output)
