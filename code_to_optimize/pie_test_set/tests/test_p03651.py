from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03651_0():
    input_content = "3 7\n9 3 4"
    expected_output = "POSSIBLE"
    run_pie_test_case("../p03651.py", input_content, expected_output)


def test_problem_p03651_1():
    input_content = "4 11\n11 3 7 15"
    expected_output = "POSSIBLE"
    run_pie_test_case("../p03651.py", input_content, expected_output)


def test_problem_p03651_2():
    input_content = "3 5\n6 9 3"
    expected_output = "IMPOSSIBLE"
    run_pie_test_case("../p03651.py", input_content, expected_output)


def test_problem_p03651_3():
    input_content = "5 12\n10 2 8 6 4"
    expected_output = "IMPOSSIBLE"
    run_pie_test_case("../p03651.py", input_content, expected_output)


def test_problem_p03651_4():
    input_content = "3 7\n9 3 4"
    expected_output = "POSSIBLE"
    run_pie_test_case("../p03651.py", input_content, expected_output)
