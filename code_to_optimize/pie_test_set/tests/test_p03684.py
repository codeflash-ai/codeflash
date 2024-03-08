from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03684_0():
    input_content = "3\n1 5\n3 9\n7 8"
    expected_output = "3"
    run_pie_test_case("../p03684.py", input_content, expected_output)


def test_problem_p03684_1():
    input_content = "6\n8 3\n4 9\n12 19\n18 1\n13 5\n7 6"
    expected_output = "8"
    run_pie_test_case("../p03684.py", input_content, expected_output)


def test_problem_p03684_2():
    input_content = "3\n1 5\n3 9\n7 8"
    expected_output = "3"
    run_pie_test_case("../p03684.py", input_content, expected_output)
