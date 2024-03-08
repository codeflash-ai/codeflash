from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03971_0():
    input_content = "10 2 3\nabccabaabb"
    expected_output = "Yes\nYes\nNo\nNo\nYes\nYes\nYes\nNo\nNo\nNo"
    run_pie_test_case("../p03971.py", input_content, expected_output)


def test_problem_p03971_1():
    input_content = "12 5 2\ncabbabaacaba"
    expected_output = "No\nYes\nYes\nYes\nYes\nNo\nYes\nYes\nNo\nYes\nNo\nNo"
    run_pie_test_case("../p03971.py", input_content, expected_output)


def test_problem_p03971_2():
    input_content = "5 2 2\nccccc"
    expected_output = "No\nNo\nNo\nNo\nNo"
    run_pie_test_case("../p03971.py", input_content, expected_output)


def test_problem_p03971_3():
    input_content = "10 2 3\nabccabaabb"
    expected_output = "Yes\nYes\nNo\nNo\nYes\nYes\nYes\nNo\nNo\nNo"
    run_pie_test_case("../p03971.py", input_content, expected_output)
