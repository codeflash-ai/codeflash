from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03821_0():
    input_content = "3\n3 5\n2 7\n9 4"
    expected_output = "7"
    run_pie_test_case("../p03821.py", input_content, expected_output)


def test_problem_p03821_1():
    input_content = "7\n3 1\n4 1\n5 9\n2 6\n5 3\n5 8\n9 7"
    expected_output = "22"
    run_pie_test_case("../p03821.py", input_content, expected_output)


def test_problem_p03821_2():
    input_content = "3\n3 5\n2 7\n9 4"
    expected_output = "7"
    run_pie_test_case("../p03821.py", input_content, expected_output)
