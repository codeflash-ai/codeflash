from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02666_0():
    input_content = "4\n2 1 -1 3"
    expected_output = "8"
    run_pie_test_case("../p02666.py", input_content, expected_output)


def test_problem_p02666_1():
    input_content = "4\n2 1 -1 3"
    expected_output = "8"
    run_pie_test_case("../p02666.py", input_content, expected_output)


def test_problem_p02666_2():
    input_content = "2\n2 1"
    expected_output = "1"
    run_pie_test_case("../p02666.py", input_content, expected_output)


def test_problem_p02666_3():
    input_content = "10\n2 6 9 -1 6 9 -1 -1 -1 -1"
    expected_output = "527841"
    run_pie_test_case("../p02666.py", input_content, expected_output)
