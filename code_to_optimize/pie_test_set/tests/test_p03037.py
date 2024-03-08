from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03037_0():
    input_content = "4 2\n1 3\n2 4"
    expected_output = "2"
    run_pie_test_case("../p03037.py", input_content, expected_output)


def test_problem_p03037_1():
    input_content = "100000 1\n1 100000"
    expected_output = "100000"
    run_pie_test_case("../p03037.py", input_content, expected_output)


def test_problem_p03037_2():
    input_content = "4 2\n1 3\n2 4"
    expected_output = "2"
    run_pie_test_case("../p03037.py", input_content, expected_output)


def test_problem_p03037_3():
    input_content = "10 3\n3 6\n5 7\n6 9"
    expected_output = "1"
    run_pie_test_case("../p03037.py", input_content, expected_output)
