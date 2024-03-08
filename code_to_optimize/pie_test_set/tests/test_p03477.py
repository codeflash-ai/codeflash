from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03477_0():
    input_content = "3 8 7 1"
    expected_output = "Left"
    run_pie_test_case("../p03477.py", input_content, expected_output)


def test_problem_p03477_1():
    input_content = "3 8 7 1"
    expected_output = "Left"
    run_pie_test_case("../p03477.py", input_content, expected_output)


def test_problem_p03477_2():
    input_content = "1 7 6 4"
    expected_output = "Right"
    run_pie_test_case("../p03477.py", input_content, expected_output)


def test_problem_p03477_3():
    input_content = "3 4 5 2"
    expected_output = "Balanced"
    run_pie_test_case("../p03477.py", input_content, expected_output)
