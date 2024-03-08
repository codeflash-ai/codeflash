from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03637_0():
    input_content = "3\n1 10 100"
    expected_output = "Yes"
    run_pie_test_case("../p03637.py", input_content, expected_output)


def test_problem_p03637_1():
    input_content = "6\n2 7 1 8 2 8"
    expected_output = "Yes"
    run_pie_test_case("../p03637.py", input_content, expected_output)


def test_problem_p03637_2():
    input_content = "3\n1 4 1"
    expected_output = "Yes"
    run_pie_test_case("../p03637.py", input_content, expected_output)


def test_problem_p03637_3():
    input_content = "3\n1 10 100"
    expected_output = "Yes"
    run_pie_test_case("../p03637.py", input_content, expected_output)


def test_problem_p03637_4():
    input_content = "2\n1 1"
    expected_output = "No"
    run_pie_test_case("../p03637.py", input_content, expected_output)


def test_problem_p03637_5():
    input_content = "4\n1 2 3 4"
    expected_output = "No"
    run_pie_test_case("../p03637.py", input_content, expected_output)
