from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03018_0():
    input_content = "ABCABC"
    expected_output = "3"
    run_pie_test_case("../p03018.py", input_content, expected_output)


def test_problem_p03018_1():
    input_content = "ABCACCBABCBCAABCB"
    expected_output = "6"
    run_pie_test_case("../p03018.py", input_content, expected_output)


def test_problem_p03018_2():
    input_content = "C"
    expected_output = "0"
    run_pie_test_case("../p03018.py", input_content, expected_output)


def test_problem_p03018_3():
    input_content = "ABCABC"
    expected_output = "3"
    run_pie_test_case("../p03018.py", input_content, expected_output)
