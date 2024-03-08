from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03392_0():
    input_content = "abc"
    expected_output = "3"
    run_pie_test_case("../p03392.py", input_content, expected_output)


def test_problem_p03392_1():
    input_content = "babacabac"
    expected_output = "6310"
    run_pie_test_case("../p03392.py", input_content, expected_output)


def test_problem_p03392_2():
    input_content = "abbac"
    expected_output = "65"
    run_pie_test_case("../p03392.py", input_content, expected_output)


def test_problem_p03392_3():
    input_content = "abc"
    expected_output = "3"
    run_pie_test_case("../p03392.py", input_content, expected_output)


def test_problem_p03392_4():
    input_content = "ababacbcacbacacbcbbcbbacbaccacbacbacba"
    expected_output = "148010497"
    run_pie_test_case("../p03392.py", input_content, expected_output)
