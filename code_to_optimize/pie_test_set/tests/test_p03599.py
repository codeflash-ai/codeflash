from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03599_0():
    input_content = "1 2 10 20 15 200"
    expected_output = "110 10"
    run_pie_test_case("../p03599.py", input_content, expected_output)


def test_problem_p03599_1():
    input_content = "1 2 1 2 100 1000"
    expected_output = "200 100"
    run_pie_test_case("../p03599.py", input_content, expected_output)


def test_problem_p03599_2():
    input_content = "1 2 10 20 15 200"
    expected_output = "110 10"
    run_pie_test_case("../p03599.py", input_content, expected_output)


def test_problem_p03599_3():
    input_content = "17 19 22 26 55 2802"
    expected_output = "2634 934"
    run_pie_test_case("../p03599.py", input_content, expected_output)
