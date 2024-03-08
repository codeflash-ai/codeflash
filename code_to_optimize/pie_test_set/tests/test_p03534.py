from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03534_0():
    input_content = "abac"
    expected_output = "YES"
    run_pie_test_case("../p03534.py", input_content, expected_output)


def test_problem_p03534_1():
    input_content = "aba"
    expected_output = "NO"
    run_pie_test_case("../p03534.py", input_content, expected_output)


def test_problem_p03534_2():
    input_content = "abac"
    expected_output = "YES"
    run_pie_test_case("../p03534.py", input_content, expected_output)


def test_problem_p03534_3():
    input_content = "babacccabab"
    expected_output = "YES"
    run_pie_test_case("../p03534.py", input_content, expected_output)
