from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03029_0():
    input_content = "1 3"
    expected_output = "3"
    run_pie_test_case("../p03029.py", input_content, expected_output)


def test_problem_p03029_1():
    input_content = "1 3"
    expected_output = "3"
    run_pie_test_case("../p03029.py", input_content, expected_output)


def test_problem_p03029_2():
    input_content = "32 21"
    expected_output = "58"
    run_pie_test_case("../p03029.py", input_content, expected_output)


def test_problem_p03029_3():
    input_content = "0 1"
    expected_output = "0"
    run_pie_test_case("../p03029.py", input_content, expected_output)
