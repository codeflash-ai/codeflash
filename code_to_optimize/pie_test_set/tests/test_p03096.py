from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03096_0():
    input_content = "5\n1\n2\n1\n2\n2"
    expected_output = "3"
    run_pie_test_case("../p03096.py", input_content, expected_output)


def test_problem_p03096_1():
    input_content = "6\n4\n2\n5\n4\n2\n4"
    expected_output = "5"
    run_pie_test_case("../p03096.py", input_content, expected_output)


def test_problem_p03096_2():
    input_content = "5\n1\n2\n1\n2\n2"
    expected_output = "3"
    run_pie_test_case("../p03096.py", input_content, expected_output)


def test_problem_p03096_3():
    input_content = "7\n1\n3\n1\n2\n3\n3\n2"
    expected_output = "5"
    run_pie_test_case("../p03096.py", input_content, expected_output)
