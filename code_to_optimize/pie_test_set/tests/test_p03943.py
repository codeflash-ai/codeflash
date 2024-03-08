from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03943_0():
    input_content = "10 30 20"
    expected_output = "Yes"
    run_pie_test_case("../p03943.py", input_content, expected_output)


def test_problem_p03943_1():
    input_content = "56 25 31"
    expected_output = "Yes"
    run_pie_test_case("../p03943.py", input_content, expected_output)


def test_problem_p03943_2():
    input_content = "10 30 20"
    expected_output = "Yes"
    run_pie_test_case("../p03943.py", input_content, expected_output)


def test_problem_p03943_3():
    input_content = "30 30 100"
    expected_output = "No"
    run_pie_test_case("../p03943.py", input_content, expected_output)
