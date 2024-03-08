from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02796_0():
    input_content = "4\n2 4\n4 3\n9 3\n100 5"
    expected_output = "3"
    run_pie_test_case("../p02796.py", input_content, expected_output)


def test_problem_p02796_1():
    input_content = "5\n10 1\n2 1\n4 1\n6 1\n8 1"
    expected_output = "5"
    run_pie_test_case("../p02796.py", input_content, expected_output)


def test_problem_p02796_2():
    input_content = "2\n8 20\n1 10"
    expected_output = "1"
    run_pie_test_case("../p02796.py", input_content, expected_output)


def test_problem_p02796_3():
    input_content = "4\n2 4\n4 3\n9 3\n100 5"
    expected_output = "3"
    run_pie_test_case("../p02796.py", input_content, expected_output)
