from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03160_0():
    input_content = "4\n10 30 40 20"
    expected_output = "30"
    run_pie_test_case("../p03160.py", input_content, expected_output)


def test_problem_p03160_1():
    input_content = "6\n30 10 60 10 60 50"
    expected_output = "40"
    run_pie_test_case("../p03160.py", input_content, expected_output)


def test_problem_p03160_2():
    input_content = "4\n10 30 40 20"
    expected_output = "30"
    run_pie_test_case("../p03160.py", input_content, expected_output)


def test_problem_p03160_3():
    input_content = "2\n10 10"
    expected_output = "0"
    run_pie_test_case("../p03160.py", input_content, expected_output)
