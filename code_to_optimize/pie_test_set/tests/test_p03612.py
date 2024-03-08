from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03612_0():
    input_content = "5\n1 4 3 5 2"
    expected_output = "2"
    run_pie_test_case("../p03612.py", input_content, expected_output)


def test_problem_p03612_1():
    input_content = "5\n1 4 3 5 2"
    expected_output = "2"
    run_pie_test_case("../p03612.py", input_content, expected_output)


def test_problem_p03612_2():
    input_content = "9\n1 2 4 9 5 8 7 3 6"
    expected_output = "3"
    run_pie_test_case("../p03612.py", input_content, expected_output)


def test_problem_p03612_3():
    input_content = "2\n1 2"
    expected_output = "1"
    run_pie_test_case("../p03612.py", input_content, expected_output)


def test_problem_p03612_4():
    input_content = "2\n2 1"
    expected_output = "0"
    run_pie_test_case("../p03612.py", input_content, expected_output)
