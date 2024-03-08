from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04044_0():
    input_content = "3 3\ndxx\naxx\ncxx"
    expected_output = "axxcxxdxx"
    run_pie_test_case("../p04044.py", input_content, expected_output)


def test_problem_p04044_1():
    input_content = "3 3\ndxx\naxx\ncxx"
    expected_output = "axxcxxdxx"
    run_pie_test_case("../p04044.py", input_content, expected_output)
