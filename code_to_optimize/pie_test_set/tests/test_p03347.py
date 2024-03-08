from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03347_0():
    input_content = "4\n0\n1\n1\n2"
    expected_output = "3"
    run_pie_test_case("../p03347.py", input_content, expected_output)


def test_problem_p03347_1():
    input_content = "3\n1\n2\n1"
    expected_output = "-1"
    run_pie_test_case("../p03347.py", input_content, expected_output)


def test_problem_p03347_2():
    input_content = "9\n0\n1\n1\n0\n1\n2\n2\n1\n2"
    expected_output = "8"
    run_pie_test_case("../p03347.py", input_content, expected_output)


def test_problem_p03347_3():
    input_content = "4\n0\n1\n1\n2"
    expected_output = "3"
    run_pie_test_case("../p03347.py", input_content, expected_output)
