from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03958_0():
    input_content = "7 3\n3 2 2"
    expected_output = "0"
    run_pie_test_case("../p03958.py", input_content, expected_output)


def test_problem_p03958_1():
    input_content = "7 3\n3 2 2"
    expected_output = "0"
    run_pie_test_case("../p03958.py", input_content, expected_output)


def test_problem_p03958_2():
    input_content = "6 3\n1 4 1"
    expected_output = "1"
    run_pie_test_case("../p03958.py", input_content, expected_output)


def test_problem_p03958_3():
    input_content = "100 1\n100"
    expected_output = "99"
    run_pie_test_case("../p03958.py", input_content, expected_output)
