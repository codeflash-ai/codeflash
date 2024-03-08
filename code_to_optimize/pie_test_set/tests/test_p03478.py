from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03478_0():
    input_content = "20 2 5"
    expected_output = "84"
    run_pie_test_case("../p03478.py", input_content, expected_output)


def test_problem_p03478_1():
    input_content = "10 1 2"
    expected_output = "13"
    run_pie_test_case("../p03478.py", input_content, expected_output)


def test_problem_p03478_2():
    input_content = "20 2 5"
    expected_output = "84"
    run_pie_test_case("../p03478.py", input_content, expected_output)


def test_problem_p03478_3():
    input_content = "100 4 16"
    expected_output = "4554"
    run_pie_test_case("../p03478.py", input_content, expected_output)
