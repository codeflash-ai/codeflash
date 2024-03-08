from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03021_0():
    input_content = "7\n0010101\n1 2\n2 3\n1 4\n4 5\n1 6\n6 7"
    expected_output = "3"
    run_pie_test_case("../p03021.py", input_content, expected_output)


def test_problem_p03021_1():
    input_content = "7\n0010101\n1 2\n2 3\n1 4\n4 5\n1 6\n6 7"
    expected_output = "3"
    run_pie_test_case("../p03021.py", input_content, expected_output)


def test_problem_p03021_2():
    input_content = "7\n0010110\n1 2\n2 3\n1 4\n4 5\n1 6\n6 7"
    expected_output = "-1"
    run_pie_test_case("../p03021.py", input_content, expected_output)


def test_problem_p03021_3():
    input_content = "2\n01\n1 2"
    expected_output = "0"
    run_pie_test_case("../p03021.py", input_content, expected_output)
