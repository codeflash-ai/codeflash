from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03861_0():
    input_content = "4 8 2"
    expected_output = "3"
    run_pie_test_case("../p03861.py", input_content, expected_output)


def test_problem_p03861_1():
    input_content = "4 8 2"
    expected_output = "3"
    run_pie_test_case("../p03861.py", input_content, expected_output)


def test_problem_p03861_2():
    input_content = "9 9 2"
    expected_output = "0"
    run_pie_test_case("../p03861.py", input_content, expected_output)


def test_problem_p03861_3():
    input_content = "1 1000000000000000000 3"
    expected_output = "333333333333333333"
    run_pie_test_case("../p03861.py", input_content, expected_output)


def test_problem_p03861_4():
    input_content = "0 5 1"
    expected_output = "6"
    run_pie_test_case("../p03861.py", input_content, expected_output)
