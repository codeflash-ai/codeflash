from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03998_0():
    input_content = "aca\naccc\nca"
    expected_output = "A"
    run_pie_test_case("../p03998.py", input_content, expected_output)


def test_problem_p03998_1():
    input_content = "abcb\naacb\nbccc"
    expected_output = "C"
    run_pie_test_case("../p03998.py", input_content, expected_output)


def test_problem_p03998_2():
    input_content = "aca\naccc\nca"
    expected_output = "A"
    run_pie_test_case("../p03998.py", input_content, expected_output)
