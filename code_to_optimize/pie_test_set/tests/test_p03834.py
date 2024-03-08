from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03834_0():
    input_content = "happy,newyear,enjoy"
    expected_output = "happy newyear enjoy"
    run_pie_test_case("../p03834.py", input_content, expected_output)


def test_problem_p03834_1():
    input_content = "haiku,atcoder,tasks"
    expected_output = "haiku atcoder tasks"
    run_pie_test_case("../p03834.py", input_content, expected_output)


def test_problem_p03834_2():
    input_content = "abcde,fghihgf,edcba"
    expected_output = "abcde fghihgf edcba"
    run_pie_test_case("../p03834.py", input_content, expected_output)


def test_problem_p03834_3():
    input_content = "happy,newyear,enjoy"
    expected_output = "happy newyear enjoy"
    run_pie_test_case("../p03834.py", input_content, expected_output)
