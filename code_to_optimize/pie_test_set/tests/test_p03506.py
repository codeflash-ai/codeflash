from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03506_0():
    input_content = "3 3\n5 7\n8 11\n3 9"
    expected_output = "2\n1\n3"
    run_pie_test_case("../p03506.py", input_content, expected_output)


def test_problem_p03506_1():
    input_content = "3 3\n5 7\n8 11\n3 9"
    expected_output = "2\n1\n3"
    run_pie_test_case("../p03506.py", input_content, expected_output)


def test_problem_p03506_2():
    input_content = "100000 2\n1 2\n3 4"
    expected_output = "1\n1"
    run_pie_test_case("../p03506.py", input_content, expected_output)
