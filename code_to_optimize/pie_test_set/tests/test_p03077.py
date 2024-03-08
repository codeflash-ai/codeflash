from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03077_0():
    input_content = "5\n3\n2\n4\n3\n5"
    expected_output = "7"
    run_pie_test_case("../p03077.py", input_content, expected_output)


def test_problem_p03077_1():
    input_content = "5\n3\n2\n4\n3\n5"
    expected_output = "7"
    run_pie_test_case("../p03077.py", input_content, expected_output)


def test_problem_p03077_2():
    input_content = "10\n123\n123\n123\n123\n123"
    expected_output = "5"
    run_pie_test_case("../p03077.py", input_content, expected_output)


def test_problem_p03077_3():
    input_content = "10000000007\n2\n3\n5\n7\n11"
    expected_output = "5000000008"
    run_pie_test_case("../p03077.py", input_content, expected_output)
