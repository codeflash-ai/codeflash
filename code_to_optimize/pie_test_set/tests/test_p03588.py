from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03588_0():
    input_content = "3\n4 7\n2 9\n6 2"
    expected_output = "8"
    run_pie_test_case("../p03588.py", input_content, expected_output)


def test_problem_p03588_1():
    input_content = "3\n4 7\n2 9\n6 2"
    expected_output = "8"
    run_pie_test_case("../p03588.py", input_content, expected_output)


def test_problem_p03588_2():
    input_content = "2\n1 1000000000\n1000000000 1"
    expected_output = "1000000001"
    run_pie_test_case("../p03588.py", input_content, expected_output)


def test_problem_p03588_3():
    input_content = "5\n1 10\n3 6\n5 2\n4 4\n2 8"
    expected_output = "7"
    run_pie_test_case("../p03588.py", input_content, expected_output)
