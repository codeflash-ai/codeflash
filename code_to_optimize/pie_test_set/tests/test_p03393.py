from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03393_0():
    input_content = "atcoder"
    expected_output = "atcoderb"
    run_pie_test_case("../p03393.py", input_content, expected_output)


def test_problem_p03393_1():
    input_content = "zyxwvutsrqponmlkjihgfedcba"
    expected_output = "-1"
    run_pie_test_case("../p03393.py", input_content, expected_output)


def test_problem_p03393_2():
    input_content = "atcoder"
    expected_output = "atcoderb"
    run_pie_test_case("../p03393.py", input_content, expected_output)


def test_problem_p03393_3():
    input_content = "abcdefghijklmnopqrstuvwzyx"
    expected_output = "abcdefghijklmnopqrstuvx"
    run_pie_test_case("../p03393.py", input_content, expected_output)


def test_problem_p03393_4():
    input_content = "abc"
    expected_output = "abcd"
    run_pie_test_case("../p03393.py", input_content, expected_output)
