from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03863_0():
    input_content = "aba"
    expected_output = "Second"
    run_pie_test_case("../p03863.py", input_content, expected_output)


def test_problem_p03863_1():
    input_content = "abc"
    expected_output = "First"
    run_pie_test_case("../p03863.py", input_content, expected_output)


def test_problem_p03863_2():
    input_content = "aba"
    expected_output = "Second"
    run_pie_test_case("../p03863.py", input_content, expected_output)


def test_problem_p03863_3():
    input_content = "abcab"
    expected_output = "First"
    run_pie_test_case("../p03863.py", input_content, expected_output)
