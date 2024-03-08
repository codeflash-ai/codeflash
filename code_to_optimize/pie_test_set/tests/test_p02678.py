from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02678_0():
    input_content = "4 4\n1 2\n2 3\n3 4\n4 2"
    expected_output = "Yes\n1\n2\n2"
    run_pie_test_case("../p02678.py", input_content, expected_output)


def test_problem_p02678_1():
    input_content = "6 9\n3 4\n6 1\n2 4\n5 3\n4 6\n1 5\n6 2\n4 5\n5 6"
    expected_output = "Yes\n6\n5\n5\n1\n1"
    run_pie_test_case("../p02678.py", input_content, expected_output)


def test_problem_p02678_2():
    input_content = "4 4\n1 2\n2 3\n3 4\n4 2"
    expected_output = "Yes\n1\n2\n2"
    run_pie_test_case("../p02678.py", input_content, expected_output)
