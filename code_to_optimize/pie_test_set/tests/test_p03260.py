from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03260_0():
    input_content = "3 1"
    expected_output = "Yes"
    run_pie_test_case("../p03260.py", input_content, expected_output)


def test_problem_p03260_1():
    input_content = "3 1"
    expected_output = "Yes"
    run_pie_test_case("../p03260.py", input_content, expected_output)


def test_problem_p03260_2():
    input_content = "1 2"
    expected_output = "No"
    run_pie_test_case("../p03260.py", input_content, expected_output)


def test_problem_p03260_3():
    input_content = "2 2"
    expected_output = "No"
    run_pie_test_case("../p03260.py", input_content, expected_output)
