from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03598_0():
    input_content = "1\n10\n2"
    expected_output = "4"
    run_pie_test_case("../p03598.py", input_content, expected_output)


def test_problem_p03598_1():
    input_content = "1\n10\n2"
    expected_output = "4"
    run_pie_test_case("../p03598.py", input_content, expected_output)


def test_problem_p03598_2():
    input_content = "2\n9\n3 6"
    expected_output = "12"
    run_pie_test_case("../p03598.py", input_content, expected_output)


def test_problem_p03598_3():
    input_content = "5\n20\n11 12 9 17 12"
    expected_output = "74"
    run_pie_test_case("../p03598.py", input_content, expected_output)
