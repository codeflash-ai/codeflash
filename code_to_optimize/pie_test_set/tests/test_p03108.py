from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03108_0():
    input_content = "4 5\n1 2\n3 4\n1 3\n2 3\n1 4"
    expected_output = "0\n0\n4\n5\n6"
    run_pie_test_case("../p03108.py", input_content, expected_output)


def test_problem_p03108_1():
    input_content = "6 5\n2 3\n1 2\n5 6\n3 4\n4 5"
    expected_output = "8\n9\n12\n14\n15"
    run_pie_test_case("../p03108.py", input_content, expected_output)


def test_problem_p03108_2():
    input_content = "4 5\n1 2\n3 4\n1 3\n2 3\n1 4"
    expected_output = "0\n0\n4\n5\n6"
    run_pie_test_case("../p03108.py", input_content, expected_output)


def test_problem_p03108_3():
    input_content = "2 1\n1 2"
    expected_output = "1"
    run_pie_test_case("../p03108.py", input_content, expected_output)
