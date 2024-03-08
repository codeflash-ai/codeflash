from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03307_0():
    input_content = "3"
    expected_output = "6"
    run_pie_test_case("../p03307.py", input_content, expected_output)


def test_problem_p03307_1():
    input_content = "999999999"
    expected_output = "1999999998"
    run_pie_test_case("../p03307.py", input_content, expected_output)


def test_problem_p03307_2():
    input_content = "3"
    expected_output = "6"
    run_pie_test_case("../p03307.py", input_content, expected_output)


def test_problem_p03307_3():
    input_content = "10"
    expected_output = "10"
    run_pie_test_case("../p03307.py", input_content, expected_output)
