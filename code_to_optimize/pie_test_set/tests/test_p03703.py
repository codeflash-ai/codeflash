from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03703_0():
    input_content = "3 6\n7\n5\n7"
    expected_output = "5"
    run_pie_test_case("../p03703.py", input_content, expected_output)


def test_problem_p03703_1():
    input_content = "1 2\n1"
    expected_output = "0"
    run_pie_test_case("../p03703.py", input_content, expected_output)


def test_problem_p03703_2():
    input_content = "3 6\n7\n5\n7"
    expected_output = "5"
    run_pie_test_case("../p03703.py", input_content, expected_output)


def test_problem_p03703_3():
    input_content = "7 26\n10\n20\n30\n40\n30\n20\n10"
    expected_output = "13"
    run_pie_test_case("../p03703.py", input_content, expected_output)
