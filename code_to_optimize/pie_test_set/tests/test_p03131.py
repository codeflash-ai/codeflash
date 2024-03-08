from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03131_0():
    input_content = "4 2 6"
    expected_output = "7"
    run_pie_test_case("../p03131.py", input_content, expected_output)


def test_problem_p03131_1():
    input_content = "4 2 6"
    expected_output = "7"
    run_pie_test_case("../p03131.py", input_content, expected_output)


def test_problem_p03131_2():
    input_content = "7 3 4"
    expected_output = "8"
    run_pie_test_case("../p03131.py", input_content, expected_output)


def test_problem_p03131_3():
    input_content = "314159265 35897932 384626433"
    expected_output = "48518828981938099"
    run_pie_test_case("../p03131.py", input_content, expected_output)
