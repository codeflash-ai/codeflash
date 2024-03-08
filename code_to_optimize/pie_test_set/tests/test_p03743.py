from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03743_0():
    input_content = "4 10\n3 4 3 3\n2\n4 3"
    expected_output = "NO\nYES"
    run_pie_test_case("../p03743.py", input_content, expected_output)


def test_problem_p03743_1():
    input_content = "5 9\n4 4 2 3 2\n5\n1 4 2 3 5"
    expected_output = "YES\nYES\nYES\nYES\nYES"
    run_pie_test_case("../p03743.py", input_content, expected_output)


def test_problem_p03743_2():
    input_content = "6 15\n4 3 5 4 2 1\n6\n1 2 3 4 5 6"
    expected_output = "NO\nNO\nYES\nNO\nNO\nYES"
    run_pie_test_case("../p03743.py", input_content, expected_output)


def test_problem_p03743_3():
    input_content = "4 10\n3 4 3 3\n2\n4 3"
    expected_output = "NO\nYES"
    run_pie_test_case("../p03743.py", input_content, expected_output)
