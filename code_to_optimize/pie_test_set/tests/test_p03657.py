from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03657_0():
    input_content = "4 5"
    expected_output = "Possible"
    run_pie_test_case("../p03657.py", input_content, expected_output)


def test_problem_p03657_1():
    input_content = "4 5"
    expected_output = "Possible"
    run_pie_test_case("../p03657.py", input_content, expected_output)


def test_problem_p03657_2():
    input_content = "1 1"
    expected_output = "Impossible"
    run_pie_test_case("../p03657.py", input_content, expected_output)
