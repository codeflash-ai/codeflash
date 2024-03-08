from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03351_0():
    input_content = "4 7 9 3"
    expected_output = "Yes"
    run_pie_test_case("../p03351.py", input_content, expected_output)


def test_problem_p03351_1():
    input_content = "1 100 2 10"
    expected_output = "Yes"
    run_pie_test_case("../p03351.py", input_content, expected_output)


def test_problem_p03351_2():
    input_content = "100 10 1 2"
    expected_output = "No"
    run_pie_test_case("../p03351.py", input_content, expected_output)


def test_problem_p03351_3():
    input_content = "10 10 10 1"
    expected_output = "Yes"
    run_pie_test_case("../p03351.py", input_content, expected_output)


def test_problem_p03351_4():
    input_content = "4 7 9 3"
    expected_output = "Yes"
    run_pie_test_case("../p03351.py", input_content, expected_output)
