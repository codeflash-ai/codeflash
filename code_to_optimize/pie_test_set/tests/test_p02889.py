from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02889_0():
    input_content = "3 2 5\n1 2 3\n2 3 3\n2\n3 2\n1 3"
    expected_output = "0\n1"
    run_pie_test_case("../p02889.py", input_content, expected_output)


def test_problem_p02889_1():
    input_content = "4 0 1\n1\n2 1"
    expected_output = "-1"
    run_pie_test_case("../p02889.py", input_content, expected_output)


def test_problem_p02889_2():
    input_content = "5 4 4\n1 2 2\n2 3 2\n3 4 3\n4 5 2\n20\n2 1\n3 1\n4 1\n5 1\n1 2\n3 2\n4 2\n5 2\n1 3\n2 3\n4 3\n5 3\n1 4\n2 4\n3 4\n5 4\n1 5\n2 5\n3 5\n4 5"
    expected_output = "0\n0\n1\n2\n0\n0\n1\n2\n0\n0\n0\n1\n1\n1\n0\n0\n2\n2\n1\n0"
    run_pie_test_case("../p02889.py", input_content, expected_output)


def test_problem_p02889_3():
    input_content = "3 2 5\n1 2 3\n2 3 3\n2\n3 2\n1 3"
    expected_output = "0\n1"
    run_pie_test_case("../p02889.py", input_content, expected_output)
