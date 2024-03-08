from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03857_0():
    input_content = "4 3 1\n1 2\n2 3\n3 4\n2 3"
    expected_output = "1 2 2 1"
    run_pie_test_case("../p03857.py", input_content, expected_output)


def test_problem_p03857_1():
    input_content = "7 4 4\n1 2\n2 3\n2 5\n6 7\n3 5\n4 5\n3 4\n6 7"
    expected_output = "1 1 2 1 2 2 2"
    run_pie_test_case("../p03857.py", input_content, expected_output)


def test_problem_p03857_2():
    input_content = "4 3 1\n1 2\n2 3\n3 4\n2 3"
    expected_output = "1 2 2 1"
    run_pie_test_case("../p03857.py", input_content, expected_output)


def test_problem_p03857_3():
    input_content = "4 2 2\n1 2\n2 3\n1 4\n2 3"
    expected_output = "1 2 2 1"
    run_pie_test_case("../p03857.py", input_content, expected_output)
