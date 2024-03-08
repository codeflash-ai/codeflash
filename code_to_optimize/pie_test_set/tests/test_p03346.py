from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03346_0():
    input_content = "4\n1\n3\n2\n4"
    expected_output = "2"
    run_pie_test_case("../p03346.py", input_content, expected_output)


def test_problem_p03346_1():
    input_content = "4\n1\n3\n2\n4"
    expected_output = "2"
    run_pie_test_case("../p03346.py", input_content, expected_output)


def test_problem_p03346_2():
    input_content = "6\n3\n2\n5\n1\n4\n6"
    expected_output = "4"
    run_pie_test_case("../p03346.py", input_content, expected_output)


def test_problem_p03346_3():
    input_content = "8\n6\n3\n1\n2\n7\n4\n8\n5"
    expected_output = "5"
    run_pie_test_case("../p03346.py", input_content, expected_output)
