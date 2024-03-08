from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04034_0():
    input_content = "3 2\n1 2\n2 3"
    expected_output = "2"
    run_pie_test_case("../p04034.py", input_content, expected_output)


def test_problem_p04034_1():
    input_content = "3 2\n1 2\n2 3"
    expected_output = "2"
    run_pie_test_case("../p04034.py", input_content, expected_output)


def test_problem_p04034_2():
    input_content = "4 4\n1 2\n2 3\n4 1\n3 4"
    expected_output = "3"
    run_pie_test_case("../p04034.py", input_content, expected_output)


def test_problem_p04034_3():
    input_content = "3 3\n1 2\n2 3\n2 3"
    expected_output = "1"
    run_pie_test_case("../p04034.py", input_content, expected_output)
