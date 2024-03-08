from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03244_0():
    input_content = "4\n3 1 3 2"
    expected_output = "1"
    run_pie_test_case("../p03244.py", input_content, expected_output)


def test_problem_p03244_1():
    input_content = "4\n3 1 3 2"
    expected_output = "1"
    run_pie_test_case("../p03244.py", input_content, expected_output)


def test_problem_p03244_2():
    input_content = "4\n1 1 1 1"
    expected_output = "2"
    run_pie_test_case("../p03244.py", input_content, expected_output)


def test_problem_p03244_3():
    input_content = "6\n105 119 105 119 105 119"
    expected_output = "0"
    run_pie_test_case("../p03244.py", input_content, expected_output)
