from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03587_0():
    input_content = "111100"
    expected_output = "4"
    run_pie_test_case("../p03587.py", input_content, expected_output)


def test_problem_p03587_1():
    input_content = "001001"
    expected_output = "2"
    run_pie_test_case("../p03587.py", input_content, expected_output)


def test_problem_p03587_2():
    input_content = "000000"
    expected_output = "0"
    run_pie_test_case("../p03587.py", input_content, expected_output)


def test_problem_p03587_3():
    input_content = "111100"
    expected_output = "4"
    run_pie_test_case("../p03587.py", input_content, expected_output)
