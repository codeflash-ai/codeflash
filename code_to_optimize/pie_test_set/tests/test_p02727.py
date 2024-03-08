from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02727_0():
    input_content = "1 2 2 2 1\n2 4\n5 1\n3"
    expected_output = "12"
    run_pie_test_case("../p02727.py", input_content, expected_output)


def test_problem_p02727_1():
    input_content = "2 2 2 2 2\n8 6\n9 1\n2 1"
    expected_output = "25"
    run_pie_test_case("../p02727.py", input_content, expected_output)


def test_problem_p02727_2():
    input_content = "1 2 2 2 1\n2 4\n5 1\n3"
    expected_output = "12"
    run_pie_test_case("../p02727.py", input_content, expected_output)


def test_problem_p02727_3():
    input_content = "2 2 4 4 4\n11 12 13 14\n21 22 23 24\n1 2 3 4"
    expected_output = "74"
    run_pie_test_case("../p02727.py", input_content, expected_output)
