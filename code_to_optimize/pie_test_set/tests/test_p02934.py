from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02934_0():
    input_content = "2\n10 30"
    expected_output = "7.5"
    run_pie_test_case("../p02934.py", input_content, expected_output)


def test_problem_p02934_1():
    input_content = "2\n10 30"
    expected_output = "7.5"
    run_pie_test_case("../p02934.py", input_content, expected_output)


def test_problem_p02934_2():
    input_content = "1\n1000"
    expected_output = "1000"
    run_pie_test_case("../p02934.py", input_content, expected_output)


def test_problem_p02934_3():
    input_content = "3\n200 200 200"
    expected_output = "66.66666666666667"
    run_pie_test_case("../p02934.py", input_content, expected_output)
