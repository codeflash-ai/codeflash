from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00991_0():
    input_content = "4 4 0 0 3 3"
    expected_output = "2"
    run_pie_test_case("../p00991.py", input_content, expected_output)


def test_problem_p00991_1():
    input_content = "500 500 0 0 200 200"
    expected_output = "34807775"
    run_pie_test_case("../p00991.py", input_content, expected_output)


def test_problem_p00991_2():
    input_content = "4 4 0 0 1 1"
    expected_output = "2"
    run_pie_test_case("../p00991.py", input_content, expected_output)


def test_problem_p00991_3():
    input_content = "4 4 0 0 3 3"
    expected_output = "2"
    run_pie_test_case("../p00991.py", input_content, expected_output)


def test_problem_p00991_4():
    input_content = "2 3 0 0 1 2"
    expected_output = "4"
    run_pie_test_case("../p00991.py", input_content, expected_output)
