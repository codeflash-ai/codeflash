from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03549_0():
    input_content = "1 1"
    expected_output = "3800"
    run_pie_test_case("../p03549.py", input_content, expected_output)


def test_problem_p03549_1():
    input_content = "100 5"
    expected_output = "608000"
    run_pie_test_case("../p03549.py", input_content, expected_output)


def test_problem_p03549_2():
    input_content = "10 2"
    expected_output = "18400"
    run_pie_test_case("../p03549.py", input_content, expected_output)


def test_problem_p03549_3():
    input_content = "1 1"
    expected_output = "3800"
    run_pie_test_case("../p03549.py", input_content, expected_output)
