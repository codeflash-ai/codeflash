from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02319_0():
    input_content = "4 5\n4 2\n5 2\n2 1\n8 3"
    expected_output = "13"
    run_pie_test_case("../p02319.py", input_content, expected_output)


def test_problem_p02319_1():
    input_content = "4 5\n4 2\n5 2\n2 1\n8 3"
    expected_output = "13"
    run_pie_test_case("../p02319.py", input_content, expected_output)


def test_problem_p02319_2():
    input_content = "2 20\n5 9\n4 10"
    expected_output = "9"
    run_pie_test_case("../p02319.py", input_content, expected_output)
