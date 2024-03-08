from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03573_0():
    input_content = "5 7 5"
    expected_output = "7"
    run_pie_test_case("../p03573.py", input_content, expected_output)


def test_problem_p03573_1():
    input_content = "5 7 5"
    expected_output = "7"
    run_pie_test_case("../p03573.py", input_content, expected_output)


def test_problem_p03573_2():
    input_content = "-100 100 100"
    expected_output = "-100"
    run_pie_test_case("../p03573.py", input_content, expected_output)


def test_problem_p03573_3():
    input_content = "1 1 7"
    expected_output = "7"
    run_pie_test_case("../p03573.py", input_content, expected_output)
