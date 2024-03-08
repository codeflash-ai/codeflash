from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03024_0():
    input_content = "oxoxoxoxoxoxox"
    expected_output = "YES"
    run_pie_test_case("../p03024.py", input_content, expected_output)


def test_problem_p03024_1():
    input_content = "oxoxoxoxoxoxox"
    expected_output = "YES"
    run_pie_test_case("../p03024.py", input_content, expected_output)


def test_problem_p03024_2():
    input_content = "xxxxxxxx"
    expected_output = "NO"
    run_pie_test_case("../p03024.py", input_content, expected_output)
