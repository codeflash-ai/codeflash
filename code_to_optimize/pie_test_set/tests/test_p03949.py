from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03949_0():
    input_content = "5\n1 2\n3 1\n4 3\n3 5\n2\n2 6\n5 7"
    expected_output = "Yes\n5\n6\n6\n5\n7"
    run_pie_test_case("../p03949.py", input_content, expected_output)


def test_problem_p03949_1():
    input_content = "4\n1 2\n2 3\n3 4\n1\n1 0"
    expected_output = "Yes\n0\n-1\n-2\n-3"
    run_pie_test_case("../p03949.py", input_content, expected_output)


def test_problem_p03949_2():
    input_content = "5\n1 2\n3 1\n4 3\n3 5\n3\n2 6\n4 3\n5 7"
    expected_output = "No"
    run_pie_test_case("../p03949.py", input_content, expected_output)


def test_problem_p03949_3():
    input_content = "5\n1 2\n3 1\n4 3\n3 5\n2\n2 6\n5 7"
    expected_output = "Yes\n5\n6\n6\n5\n7"
    run_pie_test_case("../p03949.py", input_content, expected_output)
