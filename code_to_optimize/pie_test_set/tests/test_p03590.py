from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03590_0():
    input_content = "3 5\n3 3\n4 4\n2 5"
    expected_output = "8"
    run_pie_test_case("../p03590.py", input_content, expected_output)


def test_problem_p03590_1():
    input_content = "3 6\n3 3\n4 4\n2 5"
    expected_output = "9"
    run_pie_test_case("../p03590.py", input_content, expected_output)


def test_problem_p03590_2():
    input_content = "7 14\n10 5\n7 4\n11 4\n9 8\n3 6\n6 2\n8 9"
    expected_output = "32"
    run_pie_test_case("../p03590.py", input_content, expected_output)


def test_problem_p03590_3():
    input_content = "3 5\n3 3\n4 4\n2 5"
    expected_output = "8"
    run_pie_test_case("../p03590.py", input_content, expected_output)
