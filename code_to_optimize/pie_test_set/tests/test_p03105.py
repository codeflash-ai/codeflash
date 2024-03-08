from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03105_0():
    input_content = "2 11 4"
    expected_output = "4"
    run_pie_test_case("../p03105.py", input_content, expected_output)


def test_problem_p03105_1():
    input_content = "3 9 5"
    expected_output = "3"
    run_pie_test_case("../p03105.py", input_content, expected_output)


def test_problem_p03105_2():
    input_content = "2 11 4"
    expected_output = "4"
    run_pie_test_case("../p03105.py", input_content, expected_output)


def test_problem_p03105_3():
    input_content = "100 1 10"
    expected_output = "0"
    run_pie_test_case("../p03105.py", input_content, expected_output)
