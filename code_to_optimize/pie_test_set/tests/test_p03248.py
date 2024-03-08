from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03248_0():
    input_content = "1111"
    expected_output = "-1"
    run_pie_test_case("../p03248.py", input_content, expected_output)


def test_problem_p03248_1():
    input_content = "1110"
    expected_output = "1 2\n2 3\n3 4"
    run_pie_test_case("../p03248.py", input_content, expected_output)


def test_problem_p03248_2():
    input_content = "1111"
    expected_output = "-1"
    run_pie_test_case("../p03248.py", input_content, expected_output)


def test_problem_p03248_3():
    input_content = "1010"
    expected_output = "1 2\n1 3\n1 4"
    run_pie_test_case("../p03248.py", input_content, expected_output)
