from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03292_0():
    input_content = "1 6 3"
    expected_output = "5"
    run_pie_test_case("../p03292.py", input_content, expected_output)


def test_problem_p03292_1():
    input_content = "1 6 3"
    expected_output = "5"
    run_pie_test_case("../p03292.py", input_content, expected_output)


def test_problem_p03292_2():
    input_content = "100 100 100"
    expected_output = "0"
    run_pie_test_case("../p03292.py", input_content, expected_output)


def test_problem_p03292_3():
    input_content = "11 5 5"
    expected_output = "6"
    run_pie_test_case("../p03292.py", input_content, expected_output)
