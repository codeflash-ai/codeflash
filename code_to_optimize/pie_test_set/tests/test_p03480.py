from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03480_0():
    input_content = "010"
    expected_output = "2"
    run_pie_test_case("../p03480.py", input_content, expected_output)


def test_problem_p03480_1():
    input_content = "100000000"
    expected_output = "8"
    run_pie_test_case("../p03480.py", input_content, expected_output)


def test_problem_p03480_2():
    input_content = "010"
    expected_output = "2"
    run_pie_test_case("../p03480.py", input_content, expected_output)


def test_problem_p03480_3():
    input_content = "00001111"
    expected_output = "4"
    run_pie_test_case("../p03480.py", input_content, expected_output)
