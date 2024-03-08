from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03944_0():
    input_content = "5 4 2\n2 1 1\n3 3 4"
    expected_output = "9"
    run_pie_test_case("../p03944.py", input_content, expected_output)


def test_problem_p03944_1():
    input_content = "5 4 3\n2 1 1\n3 3 4\n1 4 2"
    expected_output = "0"
    run_pie_test_case("../p03944.py", input_content, expected_output)


def test_problem_p03944_2():
    input_content = "5 4 2\n2 1 1\n3 3 4"
    expected_output = "9"
    run_pie_test_case("../p03944.py", input_content, expected_output)


def test_problem_p03944_3():
    input_content = "10 10 5\n1 6 1\n4 1 3\n6 9 4\n9 4 2\n3 1 3"
    expected_output = "64"
    run_pie_test_case("../p03944.py", input_content, expected_output)
